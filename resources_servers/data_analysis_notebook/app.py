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

import asyncio
import base64
import json
import logging
import multiprocessing
import os
import queue
import re
import shutil
import tempfile
import uuid
from asyncio import Semaphore, get_running_loop
from datetime import datetime
from functools import partial
from pathlib import Path
from time import sleep
from typing import Any, Literal, Optional, TextIO

import nbformat
import requests
from judge import (
    NotebookJudgeEvaluation,
    comparison_detail_for_judge,
    fill_notebook_judge_prompt,
    inputs_to_task_text,
    last_assistant_output_text,
    parse_notebook_judge_verdict,
    redact_signature_for_judge,
    truncate_for_judge,
)
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
from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import get_first_server_config_dict
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import get_response_json


logger = logging.getLogger(__name__)


_THINKING_SPLIT_RE = re.compile(r"</think>", re.DOTALL)
_XML_THINKING_RE = re.compile(r"<thinking>.*?</thinking>", re.DOTALL | re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


class DataAnalysisNotebookResourcesServerConfig(BaseResourcesServerConfig):
    max_concurrent_executions: int = 2
    execute_timeout_secs: int = 120
    wall_clock_margin_secs: int = 90
    # When true, skip staging and subprocess execution for reference and predicted code; the judge
    # sees task text, reference source, predicted code, and empty execution signatures only.
    skip_notebook_execution: bool = False
    image_compare_mode: Literal["exact", "none"] = "exact"
    # When set, each /verify with logging enabled writes
    # `<YYYY-mm-dd_HH-MM-SS_ffffff>_{stem}_reference.log` and the same prefix for `_predicted.log`
    # directly under this directory (server machine). Stem defaults from task / data_paths /
    # notebook metadata unless overridden in metadata. VerifierMetadata may override the directory.
    execution_log_directory: Optional[str] = None
    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    judge_endpoint_max_concurrency: int = 32
    judge_prompt_template_fpath: str = "prompt_templates/notebook_task_judge.txt"
    judge_max_output_chars: int = 12000
    judge_probe_on_startup: bool = True


class VerifierMetadata(BaseModel):
    """Fields expected under verifier_metadata for this server."""

    reference_notebook: dict[str, Any]
    data_paths: Optional[list[dict[str, str]]] = None
    # Server-side path; if set, overrides `execution_log_directory` from resources server config.
    execution_log_directory: Optional[str] = None
    # Suffix in log filenames after the datetime prefix. If unset or invalid, derived from task
    # text, data_paths filenames, and notebook metadata.
    execution_log_stem: Optional[str] = None
    # If set, overrides `skip_notebook_execution` from resources server config for this request.
    skip_notebook_execution: Optional[bool] = None


class DataAnalysisNotebookVerifyRequest(BaseVerifyRequest):
    verifier_metadata: dict[str, Any]


class DataAnalysisNotebookVerifyResponse(BaseVerifyResponse):
    match: bool = False
    reference_execution_error: Optional[str] = None
    predicted_execution_error: Optional[str] = None
    comparison_detail: str = Field(
        ...,
        min_length=1,
        description="Human-readable summary of verification outcome; always set (never null).",
    )
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
    judge_evaluations: Optional[list[NotebookJudgeEvaluation]] = None


def _verify_response(
    body: DataAnalysisNotebookVerifyRequest,
    *,
    reward: float,
    comparison_detail: str,
    match: bool = False,
    reference_execution_error: Optional[str] = None,
    predicted_execution_error: Optional[str] = None,
    reference_merged_output: Optional[dict[str, Any]] = None,
    num_reference_code_cells: int = 0,
    num_predicted_code_cells: int = 0,
    judge_evaluations: Optional[list[NotebookJudgeEvaluation]] = None,
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
        judge_evaluations=judge_evaluations,
    )


# Task log header: first line of the user message only, at most this many characters.
_MAX_TASK_QUESTION_LOG_EXCERPT_CHARS = 500
# Log filename stem: total cap; per-fragment caps when composing from task / paths / metadata.
_MAX_EXECUTION_LOG_STEM_CHARS = 80
_MAX_EXECUTION_LOG_STEM_TASK_FRAGMENT_CHARS = 56
_MAX_EXECUTION_LOG_STEM_DATA_FRAGMENT_CHARS = 28
_MAX_EXECUTION_LOG_STEM_NOTEBOOK_FRAGMENT_CHARS = 40

_UNSAFE_LOG_STEM_CHARS_RE = re.compile(r'[/\\:*?"<>|\x00-\x1f]')
_LOG_STEM_WS_RE = re.compile(r"\s+")


def _message_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    parts: list[str] = []
    for p in content:
        t = p.get("text", "") if isinstance(p, dict) else getattr(p, "text", None)
        if isinstance(t, str) and t:
            parts.append(t)
    return "\n".join(parts)


def _first_line_excerpt_for_log(s: str) -> str:
    """First line of ``s`` only, strip each line, then cap length."""
    s = s.strip()
    if not s:
        return ""
    first = s.splitlines()[0].strip()
    if len(first) <= _MAX_TASK_QUESTION_LOG_EXCERPT_CHARS:
        return first
    return first[: _MAX_TASK_QUESTION_LOG_EXCERPT_CHARS - 3] + "..."


def _task_question_excerpt_for_log(params: NeMoGymResponseCreateParamsNonStreaming) -> str:
    """First line of the (last) user `input` for execution log file headers (length-capped)."""
    inp = params.input
    if isinstance(inp, str):
        return _first_line_excerpt_for_log(inp)
    for m in reversed(inp):
        if getattr(m, "type", None) != "message":
            continue
        if getattr(m, "role", None) != "user":
            continue
        raw = _message_content_to_text(getattr(m, "content", None))
        ex = _first_line_excerpt_for_log(raw)
        if ex:
            return ex
    return ""


def _assistant_output_text(response: NeMoGymResponse) -> str:
    parts: list[str] = []
    for output in response.output:
        if output.type != "message" or output.role != "assistant":
            continue
        for content in output.content:
            if content.type == "output_text":
                parts.append(content.text)
    text = _THINKING_SPLIT_RE.split("".join(parts))[-1]
    text = _XML_THINKING_RE.sub("", text)
    return text.strip()


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


def _normalize_cell_source(src: Any) -> str:
    if isinstance(src, list):
        src = "".join(src)
    return str(src).strip()


def extract_reference_code_sources(reference_notebook: dict[str, Any]) -> list[str]:
    nb = nbformat.from_dict(reference_notebook)
    sources: list[str] = []
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        src = _normalize_cell_source(cell.get("source", ""))
        if src:
            sources.append(src)
    return sources


def _notebook_from_code_sources(sources: list[str]) -> NotebookNode:
    """Minimal nbformat notebook with code cells and no outputs (for empty merged signatures)."""
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_code_cell(s) for s in sources]
    return nb


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
                (stderr_chunks if rec["name"] == "stderr" else text_chunks).append(rec["text"])
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


# Lines of prior stdout/plain/stderr to include when a code cell times out.
_PREVIOUS_OUTPUT_PREVIEW_LINES = 5
_EXEC_ERR_MAX_CHARS = 5000
_MAX_CELL_SOURCE_LOG_CHARS = 8_000
_MAX_LOG_BLOCK_CHARS = 120_000


def _signature_json_for_log(nb: NotebookNode, image_mode: Literal["exact", "none"]) -> str:
    sig = merged_output_signature(nb, image_mode)
    try:
        sig_json = json.dumps(sig, indent=2, default=str)
    except (TypeError, ValueError) as e:  # pragma: no cover - defensive
        sig_json = f"<could not json-encode signature: {e!s}>"
    if len(sig_json) > _MAX_LOG_BLOCK_CHARS:
        sig_json = sig_json[:_MAX_LOG_BLOCK_CHARS] + "\n... [truncated] ...\n"
    return sig_json


def _write_code_cell_log_block(
    log_fh: TextIO,
    code_cell_index: int,
    src: str,
    header_line: str,
    image_mode: Literal["exact", "none"],
    *,
    executed_cell: Optional[NotebookNode] = None,
) -> None:
    if len(src) > _MAX_CELL_SOURCE_LOG_CHARS:
        src = src[:_MAX_CELL_SOURCE_LOG_CHARS] + "\n... [truncated] ...\n"
    one_cell_nb = nbformat.v4.new_notebook()
    one_cell_nb.metadata = {}
    one_cell_nb.cells = [executed_cell if executed_cell is not None else nbformat.v4.new_code_cell(src)]
    sig_json = _signature_json_for_log(one_cell_nb, image_mode)
    log_fh.write(f"---\n{header_line}\n")
    log_fh.write("## Source\n")
    log_fh.write(src)
    log_fh.write("\n## Output (this cell only, PNG omitted from JSON)\n")
    log_fh.write(sig_json)
    log_fh.write("\n")
    log_fh.flush()


def _write_execution_log_session_header(
    log_fh: TextIO,
    log_task_excerpt: Optional[str],
    cwd: Optional[Path],
    *,
    execution_skipped: bool = False,
) -> None:
    if log_task_excerpt:
        log_fh.write("# Task question\n")
        log_fh.write(log_task_excerpt.strip() + "\n\n")
    if execution_skipped:
        log_fh.write(
            "# Notebook execution was skipped for this verify (skip_notebook_execution); "
            "sources below were not run in a kernel.\n\n"
        )
    else:
        assert cwd is not None
        log_fh.write(f"Execution log (cwd={cwd})\n")
    log_fh.flush()


def _write_skip_mode_execution_logs(
    ref_log_path: Optional[Path],
    pred_log_path: Optional[Path],
    log_task_excerpt: Optional[str],
    ref_sources: list[str],
    pred_sources: list[str],
    image_mode: Literal["exact", "none"],
) -> None:
    """Append reference / predicted code-cell sources to optional log files when execution is skipped."""

    def _write_cells(path: Optional[Path], sources: list[str]) -> None:
        if path is None:
            return
        with open(path, "a", encoding="utf-8", buffering=1) as log_fh:
            _write_execution_log_session_header(log_fh, log_task_excerpt, None, execution_skipped=True)
            for i, raw_src in enumerate(sources):
                _write_code_cell_log_block(
                    log_fh,
                    i,
                    raw_src,
                    f"# Code cell {i} (0-based) — not executed (verify skipped notebook execution)",
                    image_mode,
                )

    _write_cells(ref_log_path, ref_sources)
    _write_cells(pred_log_path, pred_sources)


def _append_completed_cell_to_execution_log(log_fh: TextIO, code_cell_index: int, nb: NotebookNode) -> None:
    """Append this code cell’s source and output only to the log (one shot per cell, flushed)."""
    if code_cell_index < 0 or code_cell_index >= len(nb.cells):
        return
    cell = nb.cells[code_cell_index]
    src = _normalize_cell_source(cell.get("source", ""))
    _write_code_cell_log_block(
        log_fh,
        code_cell_index,
        src,
        f"# Code cell {code_cell_index} (0-based) completed",
        "none",
        executed_cell=cell,
    )


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
    log_path: Optional[Path] = None,
    log_task_excerpt: Optional[str] = None,
) -> tuple[Optional[NotebookNode], Optional[str]]:
    log_fh: Optional[TextIO] = None
    if log_path is not None:
        log_fh = open(log_path, "a", encoding="utf-8", buffering=1)

    try:
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
            if log_fh is not None:
                _write_execution_log_session_header(log_fh, log_task_excerpt, cwd)
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
                    if log_fh is not None:
                        log_fh.write(f"---\n# Code cell {index} (0-based) CellTimeoutError\n")
                        log_fh.write(_format_cell_timeout_error(index, client.nb, e))
                        log_fh.write("\n")
                        log_fh.flush()
                    return None, _format_cell_timeout_error(index, client.nb, e)
                if log_fh is not None:
                    _append_completed_cell_to_execution_log(log_fh, index, client.nb)
            client.set_widgets_metadata()
        return client.nb, None
    finally:
        if log_fh is not None:
            log_fh.close()


@run_sync
async def _execute_notebook_with_timeout_error_detail(
    cwd: Path,
    sources: list[str],
    timeout: int,
    log_path: Optional[Path] = None,
    log_task_excerpt: Optional[str] = None,
) -> tuple[Optional[NotebookNode], Optional[str]]:
    """Run cells like nbclient.execute(), but return structured timeout errors (see async impl)."""
    return await _async_execute_code_cells(cwd, sources, timeout, log_path=log_path, log_task_excerpt=log_task_excerpt)


def _notebook_worker(
    cwd_str: str,
    sources: list[str],
    timeout: int,
    result_queue: multiprocessing.Queue,
    log_path_str: Optional[str],
    log_task_excerpt: Optional[str],
) -> None:
    try:
        cwd = Path(cwd_str)
        log_p = Path(log_path_str) if log_path_str else None
        nb, err = _execute_notebook_with_timeout_error_detail(
            cwd, sources, timeout, log_path=log_p, log_task_excerpt=log_task_excerpt
        )
        if err is not None:
            result_queue.put(("err", err))
            return
        assert nb is not None
        result_queue.put(("ok", nbformat.writes(nb)))
    except Exception as e:
        result_queue.put(("err", f"{type(e).__name__}: {e}"))


def _terminate_process(proc: multiprocessing.Process, *, join_timeout: float = 5) -> None:
    if not proc.is_alive():
        proc.join(timeout=join_timeout)
        return
    proc.terminate()
    proc.join(5)
    if proc.is_alive():
        proc.kill()
        proc.join(2)


def execute_notebook_cells_process(
    cwd: Path,
    sources: list[str],
    timeout: int,
    wall_margin: int,
    log_path: Optional[Path] = None,
    log_task_excerpt: Optional[str] = None,
) -> tuple[Optional[NotebookNode], Optional[str]]:
    """Run code cells in an isolated process so cwd and kernels do not clash across async requests."""
    ctx = multiprocessing.get_context("spawn")
    q: multiprocessing.Queue = ctx.Queue(maxsize=1)
    lps: Optional[str] = str(log_path) if log_path is not None else None
    proc = ctx.Process(
        target=_notebook_worker,
        args=(str(cwd), sources, timeout, q, lps, log_task_excerpt),
    )
    proc.start()
    limit = timeout * max(len(sources), 1) + wall_margin
    # IMPORTANT: Drain the queue BEFORE join(). If the pipe buffer gets filled the child is deadlocked
    # in Queue.put(), while the parent waits in join() until timeout.
    try:
        status, payload = q.get(timeout=limit)
    except queue.Empty:
        if proc.is_alive():
            _terminate_process(proc)
            return None, f"Notebook execution exceeded wall-clock limit ({limit}s)"
        proc.join(timeout=5)
        return None, "Worker produced no result"

    _terminate_process(proc)

    if status == "ok":
        return nbformat.reads(payload, as_version=4), None
    return None, str(payload)


def _run_staging_and_execute(
    sources: list[str],
    data_paths: Optional[list[dict[str, str]]],
    timeout: int,
    wall_margin: int,
    log_path: Optional[Path] = None,
    log_task_excerpt: Optional[str] = None,
) -> tuple[Optional[NotebookNode], Optional[str]]:
    with tempfile.TemporaryDirectory() as tmp:
        cwd = Path(tmp)
        err = _stage_data_paths(cwd, data_paths)
        if err:
            return None, err
        return execute_notebook_cells_process(
            cwd, sources, timeout, wall_margin, log_path=log_path, log_task_excerpt=log_task_excerpt
        )


def _sanitize_execution_log_stem_fragment(s: str, max_len: int) -> str:
    """Single filesystem-safe path segment fragment (no slashes); empty if nothing usable remains."""
    s = s.strip()
    if not s:
        return ""
    s = _LOG_STEM_WS_RE.sub("_", s)
    s = _UNSAFE_LOG_STEM_CHARS_RE.sub("_", s)
    s = re.sub(r"[^\w\-.]+", "_", s, flags=re.UNICODE)
    s = re.sub(r"_+", "_", s).strip("._-")
    if not s or s in (".", ".."):
        return ""
    if len(s) > max_len:
        s = s[:max_len].rstrip("._-")
    return s


def _data_paths_stem_hint(data_paths: Optional[list[dict[str, str]]], max_len: int) -> str:
    if not data_paths:
        return ""
    for entry in data_paths:
        if not isinstance(entry, dict):
            continue
        rel = (entry.get("path") or "").strip()
        src = (entry.get("source") or "").strip()
        for candidate in (rel, src):
            if not candidate:
                continue
            base = Path(candidate).name
            frag = _sanitize_execution_log_stem_fragment(base, max_len)
            if frag:
                return frag
    return ""


def _reference_notebook_stem_hint(nb: dict[str, Any], max_len: int) -> str:
    md = nb.get("metadata")
    if not isinstance(md, dict):
        return ""
    title = md.get("title")
    if isinstance(title, str) and title.strip():
        return _sanitize_execution_log_stem_fragment(title.strip(), max_len)
    return ""


def _resolve_execution_log_stem(
    explicit: Optional[str],
    params: NeMoGymResponseCreateParamsNonStreaming,
    data_paths: Optional[list[dict[str, str]]],
    reference_notebook: dict[str, Any],
) -> str:
    """Path-safe ``stem`` segment used in ``<datetime>_{stem}_reference.log`` (and ``_predicted``)."""
    raw_explicit = (explicit or "").strip()
    if (
        raw_explicit
        and "/" not in raw_explicit
        and "\\" not in raw_explicit
        and ".." not in raw_explicit
        and raw_explicit not in (".", "..")
    ):
        stem = _sanitize_execution_log_stem_fragment(raw_explicit, _MAX_EXECUTION_LOG_STEM_CHARS)
        if stem:
            return stem

    parts: list[str] = []
    task_ex = _task_question_excerpt_for_log(params)
    if task_ex:
        frag = _sanitize_execution_log_stem_fragment(task_ex, _MAX_EXECUTION_LOG_STEM_TASK_FRAGMENT_CHARS)
        if frag:
            parts.append(frag)
    data_hint = _data_paths_stem_hint(data_paths, _MAX_EXECUTION_LOG_STEM_DATA_FRAGMENT_CHARS)
    if data_hint:
        parts.append(data_hint)
    if not parts:
        nb_hint = _reference_notebook_stem_hint(reference_notebook, _MAX_EXECUTION_LOG_STEM_NOTEBOOK_FRAGMENT_CHARS)
        if nb_hint:
            parts.append(nb_hint)
    if not parts:
        return uuid.uuid4().hex
    stem = _sanitize_execution_log_stem_fragment("_".join(parts), _MAX_EXECUTION_LOG_STEM_CHARS)
    return stem if stem else uuid.uuid4().hex


def _prepare_execution_log_paths(
    log_dir: Optional[str],
    stem: str,
) -> tuple[Optional[Path], Optional[Path]]:
    """If ``log_dir`` is set, return (reference, predicted) log paths under that root with datetime prefix."""
    if not (log_dir or "").strip():
        return None, None
    base = Path(log_dir).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    safe_stem = stem.strip() or uuid.uuid4().hex
    for i in range(8):
        unique = f"{uuid.uuid4().hex[:8]}_" if i else ""
        prefix = f"{stamp}_{unique}{safe_stem}"
        ref = base / f"{prefix}_reference.log"
        pred = base / f"{prefix}_predicted.log"
        if not ref.exists() and not pred.exists():
            return ref, pred
    stamp2 = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    prefix2 = f"{stamp2}_{uuid.uuid4().hex[:8]}_{safe_stem}"
    return base / f"{prefix2}_reference.log", base / f"{prefix2}_predicted.log"


def _setup_execution_logging(
    meta: VerifierMetadata,
    params: NeMoGymResponseCreateParamsNonStreaming,
    config: DataAnalysisNotebookResourcesServerConfig,
    *,
    task_excerpt_if_pred_log: bool,
) -> tuple[Optional[Path], Optional[Path], Optional[str]]:
    log_dir_raw = meta.execution_log_directory or config.execution_log_directory
    log_dir = log_dir_raw if (log_dir_raw or "").strip() else None
    log_stem = _resolve_execution_log_stem(
        meta.execution_log_stem,
        params,
        meta.data_paths,
        meta.reference_notebook,
    )
    ref_log_path, pred_log_path = _prepare_execution_log_paths(log_dir, log_stem)
    if task_excerpt_if_pred_log:
        log_task = _task_question_excerpt_for_log(params) if (ref_log_path or pred_log_path) else None
    else:
        log_task = _task_question_excerpt_for_log(params) if ref_log_path else None
    return ref_log_path, pred_log_path, log_task


def _judge_inputs_from_verify(
    params: NeMoGymResponseCreateParamsNonStreaming,
    pred_sources: list[str],
) -> tuple[str, str]:
    return inputs_to_task_text(params), "\n\n---\n\n".join(pred_sources)


def _truncate_exec_error(err: Optional[str]) -> Optional[str]:
    return err[:_EXEC_ERR_MAX_CHARS] if err else None


def _finalize_judge_verification(
    body: DataAnalysisNotebookVerifyRequest,
    ok: Optional[bool],
    evaluation: NotebookJudgeEvaluation,
    *,
    num_reference_code_cells: int,
    num_predicted_code_cells: int,
    reference_merged_output: Optional[dict[str, Any]] = None,
    reference_execution_error: Optional[str] = None,
    predicted_execution_error: Optional[str] = None,
) -> DataAnalysisNotebookVerifyResponse:
    if ok is None and evaluation.verdict_label == "judge_parsing_error":
        logger.error(
            "Notebook judge did not return a parseable verdict (verdict_label=%s). reward=0.0.",
            evaluation.verdict_label,
        )
    return _verify_response(
        body,
        reward=1.0 if ok is True else 0.0,
        match=ok is True,
        comparison_detail=comparison_detail_for_judge(ok, evaluation),
        reference_merged_output=reference_merged_output,
        reference_execution_error=reference_execution_error,
        predicted_execution_error=predicted_execution_error,
        num_reference_code_cells=num_reference_code_cells,
        num_predicted_code_cells=num_predicted_code_cells,
        judge_evaluations=[evaluation],
    )


class DataAnalysisNotebookResourcesServer(SimpleResourcesServer):
    config: DataAnalysisNotebookResourcesServerConfig

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._semaphore: Semaphore = Semaphore(value=self.config.max_concurrent_executions)
        self._judge_semaphore = asyncio.Semaphore(value=self.config.judge_endpoint_max_concurrency)
        tpl_path = Path(self.config.judge_prompt_template_fpath)
        if not tpl_path.is_absolute():
            tpl_path = Path(__file__).resolve().parent / tpl_path
        with open(tpl_path, encoding="utf-8") as f:
            self._judge_prompt_template = f.read().strip()

    def setup_webserver(self):
        if self.config.judge_probe_on_startup:
            self._ensure_judge_backend_ready()
        return super().setup_webserver()

    def _ensure_judge_backend_ready(self) -> None:
        judge_name = self.config.judge_model_server.name
        logger.info("Waiting for judge model server '%s' to become reachable...", judge_name)
        while self.server_client.poll_for_status(judge_name) != "success":
            sleep(10)

        judge_config = get_first_server_config_dict(self.server_client.global_config_dict, judge_name)
        judge_url = self.server_client._build_server_base_url(judge_config)
        logger.info("Verifying judge backend through '%s' at %s ...", judge_name, judge_url)
        while True:
            try:
                requests.post(f"{judge_url}/v1/responses", json={"input": []}, timeout=10)
                break
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                logger.warning("Judge backend not yet reachable through '%s', retrying in 10s...", judge_name)
                sleep(10)
        logger.info("Judge model server '%s' is reachable.", judge_name)

    async def _run_llm_judge(
        self,
        *,
        task_text: str,
        reference_section: str,
        predicted_code: str,
        predicted_signature: dict[str, Any],
    ) -> tuple[Optional[bool], NotebookJudgeEvaluation]:
        cfg = self.config
        max_c = cfg.judge_max_output_chars
        pred_redacted = redact_signature_for_judge(predicted_signature, max_c)
        pred_json = json.dumps(pred_redacted, indent=2, default=str)
        pred_code_trunc = truncate_for_judge(predicted_code, max_c * 2)

        user_prompt = fill_notebook_judge_prompt(
            self._judge_prompt_template,
            task_text=task_text,
            reference_section=reference_section,
            predicted_code=pred_code_trunc,
            predicted_signature_json=pred_json,
        )

        responses_create_params = cfg.judge_responses_create_params.model_copy(deep=True)
        responses_create_params.input = [
            NeMoGymEasyInputMessage(role="user", content=user_prompt),
        ]

        async with self._judge_semaphore:
            try:
                http_response = await self.server_client.post(
                    server_name=cfg.judge_model_server.name,
                    url_path="/v1/responses",
                    json=responses_create_params,
                )
                judge_response = NeMoGymResponse.model_validate(await get_response_json(http_response))
            except Exception as e:
                logger.error("Judge HTTP POST error: %s %s", type(e).__name__, e)
                return None, NotebookJudgeEvaluation(
                    responses_create_params=responses_create_params,
                    verdict_label="judge_error",
                )

        evaluation = NotebookJudgeEvaluation(
            responses_create_params=responses_create_params,
            response=judge_response,
        )
        text = last_assistant_output_text(judge_response)
        if text is None:
            evaluation.verdict_label = "judge_parsing_error"
            return None, evaluation

        passed, vlabel, reason = parse_notebook_judge_verdict(text)
        evaluation.verdict_label = vlabel
        evaluation.reason = reason
        if passed is None:
            return None, evaluation
        return passed, evaluation

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

        skip_exec = (
            meta.skip_notebook_execution
            if meta.skip_notebook_execution is not None
            else self.config.skip_notebook_execution
        )

        timeout = self.config.execute_timeout_secs
        margin = self.config.wall_clock_margin_secs
        image_mode = self.config.image_compare_mode
        max_c = self.config.judge_max_output_chars

        if skip_exec:
            ref_log_path, pred_log_path, log_task = _setup_execution_logging(
                meta,
                body.responses_create_params,
                self.config,
                task_excerpt_if_pred_log=True,
            )
            _write_skip_mode_execution_logs(
                ref_log_path,
                pred_log_path,
                log_task,
                ref_sources,
                pred_sources,
                image_mode,
            )
            task_text, predicted_code_block = _judge_inputs_from_verify(body.responses_create_params, pred_sources)
            joined_ref = "\n\n---\n\n".join(ref_sources)
            reference_section = (
                "(Ground-truth reference code only; notebooks were not executed. "
                "No runtime outputs. The predicted execution block below is empty. "
                "Decide PASS/FAIL from the task and code alone.)\n\n" + truncate_for_judge(joined_ref, max_c * 2)
            )
            sig_pred = merged_output_signature(_notebook_from_code_sources(pred_sources), image_mode)
            ok, evaluation = await self._run_llm_judge(
                task_text=task_text,
                reference_section=reference_section,
                predicted_code=predicted_code_block,
                predicted_signature=sig_pred,
            )
            return _finalize_judge_verification(
                body,
                ok,
                evaluation,
                num_reference_code_cells=len(ref_sources),
                num_predicted_code_cells=len(pred_sources),
            )

        ref_log_path, pred_log_path, log_task = _setup_execution_logging(
            meta,
            body.responses_create_params,
            self.config,
            task_excerpt_if_pred_log=False,
        )

        def _execute_staged(sources: list[str], log_p: Optional[Path]) -> tuple[Optional[NotebookNode], Optional[str]]:
            return _run_staging_and_execute(
                sources,
                meta.data_paths,
                timeout,
                margin,
                log_path=log_p,
                log_task_excerpt=log_task,
            )

        loop = get_running_loop()
        async with self._semaphore:
            ref_nb, ref_err = await loop.run_in_executor(None, partial(_execute_staged, ref_sources, ref_log_path))
            pred_nb, pred_err = await loop.run_in_executor(None, partial(_execute_staged, pred_sources, pred_log_path))

        sig_ref: Optional[dict[str, Any]] = None
        if not ref_err:
            assert ref_nb is not None
            sig_ref = merged_output_signature(ref_nb, image_mode)

        if pred_err:
            pred_exc = _truncate_exec_error(pred_err)
            return _verify_response(
                body,
                reward=0.0,
                comparison_detail=f"predicted_execution_failed: {pred_exc}",
                predicted_execution_error=pred_exc,
                reference_merged_output=sig_ref,
                num_reference_code_cells=len(ref_sources),
                num_predicted_code_cells=len(pred_sources),
                reference_execution_error=_truncate_exec_error(ref_err),
            )

        assert pred_nb is not None
        sig_pred = merged_output_signature(pred_nb, image_mode)

        task_text, predicted_code_block = _judge_inputs_from_verify(body.responses_create_params, pred_sources)

        if ref_err:
            reference_section = (
                "(Reference notebook did not execute successfully; no reference output to compare.)\n"
                f"Error:\n{_truncate_exec_error(ref_err)}"
            )
        else:
            assert sig_ref is not None
            reference_section = json.dumps(redact_signature_for_judge(sig_ref, max_c), indent=2, default=str)

        ok, evaluation = await self._run_llm_judge(
            task_text=task_text,
            reference_section=reference_section,
            predicted_code=predicted_code_block,
            predicted_signature=sig_pred,
        )

        return _finalize_judge_verification(
            body,
            ok,
            evaluation,
            reference_merged_output=sig_ref,
            reference_execution_error=_truncate_exec_error(ref_err),
            num_reference_code_cells=len(ref_sources),
            num_predicted_code_cells=len(pred_sources),
        )


if __name__ == "__main__":
    DataAnalysisNotebookResourcesServer.run_webserver()
