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
from pathlib import Path
from typing import Any, Literal, Optional

import nbformat
from nbclient import NotebookClient
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
    data_files: Optional[dict[str, str]] = None
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
    return [m.group(1).strip() for m in _CODE_FENCE_RE.finditer(text) if m.group(1).strip()]


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


def _stage_data_files(target: Path, data_files: Optional[dict[str, str]]) -> Optional[str]:
    if not data_files:
        return None
    try:
        for rel, content in data_files.items():
            path = target / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, str) and content.startswith("base64:"):
                path.write_bytes(base64.b64decode(content[7:]))
            else:
                path.write_text(content if isinstance(content, str) else str(content), encoding="utf-8")
    except (OSError, ValueError, TypeError) as e:
        return f"data_files staging failed: {e}"
    return None


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
                if isinstance(png, str):
                    png_b64 = png
                else:
                    png_b64 = base64.b64encode(png).decode("ascii")
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


def _notebook_worker(cwd_str: str, sources: list[str], timeout: int, result_queue: multiprocessing.Queue) -> None:
    import os
    from pathlib import Path

    cwd = Path(cwd_str)
    os.environ["MPLBACKEND"] = "Agg"
    try:
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
        client.execute()
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
    data_files: Optional[dict[str, str]],
    data_paths: Optional[list[dict[str, str]]],
    timeout: int,
    wall_margin: int,
) -> tuple[Optional[NotebookNode], Optional[str]]:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        cwd = Path(tmp)
        err = _stage_data_files(cwd, data_files)
        if err:
            return None, err
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
            return DataAnalysisNotebookVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                match=False,
                comparison_detail="verifier_metadata is missing",
                num_reference_code_cells=0,
                num_predicted_code_cells=0,
            )
        try:
            meta = VerifierMetadata.model_validate(body.verifier_metadata)
        except ValidationError as e:
            return DataAnalysisNotebookVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                match=False,
                comparison_detail=f"invalid verifier_metadata: {e}"[:2000],
                num_reference_code_cells=0,
                num_predicted_code_cells=0,
            )
        ref_sources = extract_reference_code_sources(meta.reference_notebook)
        pred_text = _assistant_output_text(body.response)
        pred_sources = extract_predicted_code_cells(pred_text)

        if not ref_sources:
            return DataAnalysisNotebookVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                match=False,
                comparison_detail="reference notebook has no code cells",
                num_reference_code_cells=0,
                num_predicted_code_cells=len(pred_sources),
            )

        if not pred_sources:
            return DataAnalysisNotebookVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                match=False,
                comparison_detail="model output has no python code fences",
                num_reference_code_cells=len(ref_sources),
                num_predicted_code_cells=0,
            )

        loop = get_running_loop()
        timeout = self.config.execute_timeout_secs
        margin = self.config.wall_clock_margin_secs
        image_mode = self.config.image_compare_mode

        async with self._semaphore:
            ref_nb, ref_err = await loop.run_in_executor(
                None,
                lambda: _run_staging_and_execute(
                    ref_sources,
                    meta.data_files,
                    meta.data_paths,
                    timeout,
                    margin,
                ),
            )
            pred_nb, pred_err = await loop.run_in_executor(
                None,
                lambda: _run_staging_and_execute(
                    pred_sources,
                    meta.data_files,
                    meta.data_paths,
                    timeout,
                    margin,
                ),
            )

        if ref_err:
            return DataAnalysisNotebookVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                match=False,
                reference_execution_error=ref_err[:2000],
                num_reference_code_cells=len(ref_sources),
                num_predicted_code_cells=len(pred_sources),
            )

        assert ref_nb is not None
        sig_ref = merged_output_signature(ref_nb, image_mode)

        if pred_err:
            return DataAnalysisNotebookVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                match=False,
                predicted_execution_error=pred_err[:2000],
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

        return DataAnalysisNotebookVerifyResponse(
            **body.model_dump(),
            reward=1.0 if ok else 0.0,
            match=ok,
            comparison_detail=detail,
            reference_merged_output=sig_ref,
            num_reference_code_cells=len(ref_sources),
            num_predicted_code_cells=len(pred_sources),
        )


if __name__ == "__main__":
    DataAnalysisNotebookResourcesServer.run_webserver()
