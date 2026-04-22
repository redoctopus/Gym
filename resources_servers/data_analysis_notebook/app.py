# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import base64
import json
import multiprocessing
import queue
import re
import shutil
from asyncio import Semaphore, get_running_loop
from functools import partial
from pathlib import Path
from typing import Any, Literal, Optional

import nbformat
from jupyter_client.utils import ensure_async
from jupyter_core.utils import run_sync
from nbclient import NotebookClient
from nbclient.exceptions import CellTimeoutError
from nbformat.notebooknode import NotebookNode
from pydantic import BaseModel, Field, ValidationError

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.openai_utils import NeMoGymResponse


_THINKING_SPLIT_RE = re.compile(r"</think>", re.DOTALL)
_XML_THINKING_RE = re.compile(r"<thinking>.*?</thinking>", re.DOTALL | re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


class DataAnalysisNotebookResourcesServerConfig(BaseResourcesServerConfig):
    max_concurrent_executions: int = 2
    execute_timeout_secs: int = 120
    wall_clock_margin_secs: int = 90
    image_compare_mode: Literal["exact", "none"] = "exact"


class VerifierMetadata(BaseModel):
    """Fields expected under verifier_metadata for this server."""

    reference_notebook: dict[str, Any]
    data_paths: Optional[list[dict[str, str]]] = None


class DataAnalysisNotebookVerifyRequest(BaseVerifyRequest):
    verifier_metadata: dict[str, Any]


class DataAnalysisNotebookVerifyResponse(BaseVerifyResponse):
    match: bool = False
    reference_execution_error: Optional[str] = None
    predicted_execution_error: Optional[str] = None
    comparison_detail: Optional[str] = None
    reference_merged_output: Optional[dict[str, Any]] = Field(
        default=None,
        description=(
            "Merged text_merged (stdout + text/plain in output order), stderr_merged, pngs, and errors "
            "from executing reference code cells (same structure as the 'reference' object in "
            "comparison_detail when present)."
        ),
    )
    num_reference_code_cells: int = 0
    num_predicted_code_cells: int = 0


def _verify_response(
    body: DataAnalysisNotebookVerifyRequest,
    *,
    reward: float,
    match: bool = False,
    comparison_detail: Optional[str] = None,
    reference_execution_error: Optional[str] = None,
    predicted_execution_error: Optional[str] = None,
    reference_merged_output: Optional[dict[str, Any]] = None,
    num_reference_code_cells: int = 0,
    num_predicted_code_cells: int = 0,
) -> DataAnalysisNotebookVerifyResponse:
    return DataAnalysisNotebookVerifyResponse(
        **body.model_dump(),
        reward=reward,
        match=match,
        comparison_detail=comparison_detail,
        reference_execution_error=reference_execution_error,
        predicted_execution_error=predicted_execution_error,
        reference_merged_output=reference_merged_output,
        num_reference_code_cells=num_reference_code_cells,
        num_predicted_code_cells=num_predicted_code_cells,
    )


def _strip_thinking(text: str) -> str:
    if "</think>" in text:
        text = _THINKING_SPLIT_RE.split(text)[-1]
    text = _XML_THINKING_RE.sub("", text)
    return text.strip()


def _assistant_output_text(response: NeMoGymResponse) -> str:
    parts: list[str] = []
    for output in response.output:
        if output.type != "message" or output.role != "assistant":
            continue
        for content in output.content:
            if content.type == "output_text":
                parts.append(content.text)
    return _strip_thinking("".join(parts))


def extract_predicted_code_cells(text: str) -> list[str]:
    """Ordered Python cells from ```python ... ``` fences."""
    if not text or not text.strip():
        return []
    cells: list[str] = []
    for m in _CODE_FENCE_RE.finditer(text):
        block = m.group(1).strip()
        if block:
            cells.append(block)
    return cells


def extract_reference_code_sources(reference_notebook: dict[str, Any]) -> list[str]:
    nb = nbformat.from_dict(reference_notebook)
    sources: list[str] = []
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        src = str(src).strip()
        if src:
            sources.append(src)
    return sources


def _stage_data_paths(target: Path, data_paths: Optional[list[dict[str, str]]]) -> Optional[str]:
    """Copy files or directories from the local filesystem into the sandbox cwd.

    Each entry: ``{"source": "/host/path/or/relative", "path": "relative/path/in/sandbox"}``.
    ``source`` is resolved on the machine running the resources server; ``path`` is where
    notebook code should open the file (relative to the execution working directory).
    """
    if not data_paths:
        return None
    try:
        for item in data_paths:
            src = Path(item["source"]).expanduser().resolve()
            rel_dest = item["path"]
            dest = target / rel_dest
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not src.exists():
                return f"data_paths: source does not exist: {src}"
            if src.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(src, dest)
            else:
                shutil.copy2(src, dest)
    except (KeyError, OSError, ValueError, TypeError) as e:
        return f"data_paths staging failed: {e}"
    return None


def _canonicalize_cell_outputs(cell: NotebookNode) -> list[dict[str, Any]]:
    """Per-output records from one executed code cell."""
    records: list[dict[str, Any]] = []
    for out in cell.get("outputs", []):
        otype = out.get("output_type")
        if otype == "stream":
            text = out.get("text", "")
            if isinstance(text, list):
                text = "".join(text)
            name = out.get("name", "stdout")
            records.append({"kind": "stream", "name": name, "text": _normalize_text(text)})
        elif otype == "error":
            tb = out.get("traceback", [])
            if isinstance(tb, list):
                tb = [_normalize_text(str(line)) for line in tb]
            records.append(
                {
                    "kind": "error",
                    "ename": str(out.get("ename", "")),
                    "evalue": _normalize_text(str(out.get("evalue", ""))),
                    "traceback": tb,
                }
            )
        elif otype in ("display_data", "execute_result"):
            data = dict(out.get("data", {}))
            plain = data.get("text/plain")
            if plain is not None:
                if not isinstance(plain, str):
                    plain = str(plain)
                records.append({"kind": "plain", "text": _normalize_text(plain)})
            png = data.get("image/png")
            if png is not None:
                png_b64 = png if isinstance(png, str) else base64.b64encode(png).decode("ascii")
                records.append({"kind": "png", "data": png_b64})
    return records


def _normalize_text(s: str) -> str:
    return "\n".join(line.rstrip() for line in s.replace("\r\n", "\n").strip().split("\n")).strip()


def merged_output_signature(nb: NotebookNode, image_mode: Literal["exact", "none"]) -> dict[str, Any]:
    """Flatten all code-cell outputs into one comparable structure."""
    text_chunks: list[str] = []
    stderr_chunks: list[str] = []
    pngs: list[str] = []
    errors: list[dict[str, Any]] = []

    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        for rec in _canonicalize_cell_outputs(cell):
            if rec["kind"] == "stream":
                if rec["name"] == "stderr":
                    stderr_chunks.append(rec["text"])
                else:
                    text_chunks.append(rec["text"])
            elif rec["kind"] == "plain":
                text_chunks.append(rec["text"])
            elif rec["kind"] == "png":
                if image_mode == "exact":
                    pngs.append(rec["data"])
            elif rec["kind"] == "error":
                errors.append({"ename": rec["ename"], "evalue": rec["evalue"], "traceback": rec["traceback"]})

    return {
        "text_merged": _normalize_text("\n".join(text_chunks)),
        "stderr_merged": _normalize_text("\n".join(stderr_chunks)),
        "pngs": pngs,
        "errors": errors,
    }


def merged_signatures_match(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return a == b


# Lines of prior stdout/plain/stderr to include when a code cell times out.
_PREVIOUS_OUTPUT_PREVIEW_LINES = 5
_REFERENCE_EXEC_ERR_MAX_CHARS = 5000


def _previous_output_excerpt(nb: NotebookNode, failing_code_cell_index: int, max_lines: int) -> str:
    """Last lines of merged output from code cells before ``failing_code_cell_index``."""
    if failing_code_cell_index <= 0 or not nb.cells:
        return "(no output from previous code cells)"
    pnb = nbformat.v4.new_notebook()
    pnb.metadata = dict(nb.get("metadata") or {})
    pnb.cells = nb.cells[:failing_code_cell_index]
    try:
        sig = merged_output_signature(pnb, "none")
    except Exception:  # pragma: no cover - defensive
        return "(could not collect output from previous code cells)"
    parts: list[str] = []
    t = (sig.get("text_merged") or "").strip()
    s = (sig.get("stderr_merged") or "").strip()
    if t:
        parts.append(t)
    if s:
        parts.append(s)
    blob = "\n".join(parts) if parts else "(empty output from previous code cells)"
    lines = blob.split("\n")
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "...\n" + "\n".join(lines[-max_lines:])


def _format_cell_timeout_error(
    code_cell_index: int,
    nb: NotebookNode,
    original_exc: CellTimeoutError,
) -> str:
    excerpt = _previous_output_excerpt(nb, code_cell_index, _PREVIOUS_OUTPUT_PREVIEW_LINES)
    return (
        f"Code cell index {code_cell_index} (0-based) timed out after execute_timeout_secs "
        f"({_PREVIOUS_OUTPUT_PREVIEW_LINES} lines of prior output at most, stdout+stderr):\n"
        f"{excerpt}\n"
        f"---\n{original_exc!s}"
    )


async def _async_execute_code_cells(
    cwd: Path,
    sources: list[str],
    timeout: int,
) -> tuple[Optional[NotebookNode], Optional[str]]:
    import os

    os.environ["MPLBACKEND"] = "Agg"
    os.chdir(cwd)
    nb = nbformat.v4.new_notebook()
    for src in sources:
        nb.cells.append(nbformat.v4.new_code_cell(source=src))
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(cwd)}},
    )
    client.reset_execution_trackers()
    async with client.async_setup_kernel():
        assert client.kc is not None
        client.log.info("Executing notebook with kernel: %s" % client.kernel_name)
        msg_id = await ensure_async(client.kc.kernel_info())
        info_msg = await client.async_wait_for_reply(msg_id)
        if info_msg is not None:
            if "language_info" in info_msg["content"]:
                client.nb.metadata["language_info"] = info_msg["content"]["language_info"]
            else:
                raise RuntimeError(
                    'Kernel info received message content has no "language_info" key. '
                    "Content is:\n" + str(info_msg["content"])
                )
        for index, cell in enumerate(client.nb.cells):
            try:
                await client.async_execute_cell(cell, index, execution_count=client.code_cells_executed + 1)
            except CellTimeoutError as e:
                return None, _format_cell_timeout_error(index, client.nb, e)
        client.set_widgets_metadata()
    return client.nb, None


@run_sync
async def _execute_notebook_with_timeout_error_detail(
    cwd: Path, sources: list[str], timeout: int
) -> tuple[Optional[NotebookNode], Optional[str]]:
    """Run cells like nbclient.execute(), but return structured timeout errors (see async impl)."""
    return await _async_execute_code_cells(cwd, sources, timeout)


def _notebook_worker(cwd_str: str, sources: list[str], timeout: int, result_queue: multiprocessing.Queue) -> None:
    from pathlib import Path

    try:
        cwd = Path(cwd_str)
        nb, err = _execute_notebook_with_timeout_error_detail(cwd, sources, timeout)
        if err is not None:
            result_queue.put(("err", err))
            return
        assert nb is not None
        result_queue.put(("ok", nbformat.writes(nb)))
    except Exception as e:
        result_queue.put(("err", f"{type(e).__name__}: {e}"))


def execute_notebook_cells_process(
    cwd: Path,
    sources: list[str],
    timeout: int,
    wall_margin: int,
) -> tuple[Optional[NotebookNode], Optional[str]]:
    """Run code cells in an isolated process so cwd and kernels do not clash across async requests."""
    ctx = multiprocessing.get_context("spawn")
    q: multiprocessing.Queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=_notebook_worker, args=(str(cwd), sources, timeout, q))
    proc.start()
    limit = timeout * max(len(sources), 1) + wall_margin
    proc.join(timeout=limit)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()
            proc.join(2)
        return None, f"Notebook execution exceeded wall-clock limit ({limit}s)"

    try:
        status, payload = q.get_nowait()
    except queue.Empty:
        return None, "Worker produced no result"

    if status == "ok":
        return nbformat.reads(payload, as_version=4), None
    return None, str(payload)


def _run_staging_and_execute(
    sources: list[str],
    data_paths: Optional[list[dict[str, str]]],
    timeout: int,
    wall_margin: int,
) -> tuple[Optional[NotebookNode], Optional[str]]:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        cwd = Path(tmp)
        err = _stage_data_paths(cwd, data_paths)
        if err:
            return None, err
        return execute_notebook_cells_process(cwd, sources, timeout, wall_margin)


class DataAnalysisNotebookResourcesServer(SimpleResourcesServer):
    config: DataAnalysisNotebookResourcesServerConfig

    def model_post_init(self, context: Any) -> None:
        self._semaphore: Semaphore = Semaphore(value=self.config.max_concurrent_executions)

    async def verify(self, body: DataAnalysisNotebookVerifyRequest) -> DataAnalysisNotebookVerifyResponse:
        if not body.verifier_metadata:
            return _verify_response(
                body,
                reward=0.0,
                comparison_detail="verifier_metadata is missing",
            )
        try:
            meta = VerifierMetadata.model_validate(body.verifier_metadata)
        except ValidationError as e:
            return _verify_response(
                body,
                reward=0.0,
                comparison_detail=f"invalid verifier_metadata: {e}"[:2000],
            )
        ref_sources = extract_reference_code_sources(meta.reference_notebook)
        pred_text = _assistant_output_text(body.response)
        pred_sources = extract_predicted_code_cells(pred_text)

        if not ref_sources:
            return _verify_response(
                body,
                reward=0.0,
                comparison_detail="reference notebook has no code cells",
                num_predicted_code_cells=len(pred_sources),
            )

        if not pred_sources:
            return _verify_response(
                body,
                reward=0.0,
                comparison_detail="model output has no python code fences",
                num_reference_code_cells=len(ref_sources),
            )

        loop = get_running_loop()
        timeout = self.config.execute_timeout_secs
        margin = self.config.wall_clock_margin_secs
        image_mode = self.config.image_compare_mode

        run_staged = partial(
            _run_staging_and_execute,
            data_paths=meta.data_paths,
            timeout=timeout,
            wall_margin=margin,
        )
        async with self._semaphore:
            ref_nb, ref_err = await loop.run_in_executor(None, partial(run_staged, ref_sources))
            pred_nb, pred_err = await loop.run_in_executor(None, partial(run_staged, pred_sources))

        if ref_err:
            return _verify_response(
                body,
                reward=0.0,
                reference_execution_error=ref_err[:_REFERENCE_EXEC_ERR_MAX_CHARS],
                num_reference_code_cells=len(ref_sources),
                num_predicted_code_cells=len(pred_sources),
            )

        assert ref_nb is not None
        sig_ref = merged_output_signature(ref_nb, image_mode)

        if pred_err:
            return _verify_response(
                body,
                reward=0.0,
                predicted_execution_error=pred_err[:_REFERENCE_EXEC_ERR_MAX_CHARS],
                reference_merged_output=sig_ref,
                num_reference_code_cells=len(ref_sources),
                num_predicted_code_cells=len(pred_sources),
            )

        assert pred_nb is not None
        sig_pred = merged_output_signature(pred_nb, image_mode)
        ok = merged_signatures_match(sig_ref, sig_pred)
        detail = None
        if not ok:
            detail = json.dumps({"reference": sig_ref, "predicted": sig_pred}, default=str)[:4000]

        return _verify_response(
            body,
            reward=1.0 if ok else 0.0,
            match=ok,
            comparison_detail=detail,
            reference_merged_output=sig_ref,
            num_reference_code_cells=len(ref_sources),
            num_predicted_code_cells=len(pred_sources),
        )


if __name__ == "__main__":
    DataAnalysisNotebookResourcesServer.run_webserver()
