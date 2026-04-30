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
import queue
import re
import shutil
import uuid
from asyncio import Semaphore, get_running_loop
from functools import partial
from pathlib import Path
from time import sleep
from typing import Any, Literal, Optional, TextIO

import nbformat
import requests
from jupyter_client.utils import ensure_async
from jupyter_core.utils import run_sync
from nbclient import NotebookClient
from nbclient.exceptions import CellTimeoutError
from nbformat.notebooknode import NotebookNode
from pydantic import BaseModel, ConfigDict, Field, ValidationError

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
    image_compare_mode: Literal["exact", "none"] = "exact"
    # When set, each successful /verify with logging metadata writes {stem}_reference.log and
    # {stem}_predicted.log under this directory (server machine). VerifierMetadata may override.
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
    # Basename for `{stem}_reference.log` and `{stem}_predicted.log`; if unset, a unique id is used.
    execution_log_stem: Optional[str] = None


class DataAnalysisNotebookVerifyRequest(BaseVerifyRequest):
    verifier_metadata: dict[str, Any]


class NotebookJudgeEvaluation(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    response: Optional[NeMoGymResponse] = None
    verdict_label: Optional[str] = None
    reason: Optional[str] = None


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


_JUDGE_VERDICT_PASS_RE = re.compile(r"^\s*VERDICT:\s*PASS\s*$", re.MULTILINE | re.IGNORECASE)
_JUDGE_VERDICT_FAIL_RE = re.compile(r"^\s*VERDICT:\s*FAIL\s*$", re.MULTILINE | re.IGNORECASE)
_JUDGE_REASON_RE = re.compile(r"^\s*REASON:\s*(.+?)(?:\n|$)", re.MULTILINE | re.DOTALL | re.IGNORECASE)


def _truncate_for_judge(s: str, max_chars: int) -> str:
    s = s or ""
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 20] + "\n...[truncated]...\n"


def redact_signature_for_judge(sig: dict[str, Any], max_chars_per_field: int) -> dict[str, Any]:
    """Drop PNG payloads; truncate text fields for the judge prompt."""
    pngs = sig.get("pngs") or []
    png_note = f"{len(pngs)} PNG figure(s); pixel data omitted."
    errors = sig.get("errors") or []
    err_out: list[dict[str, Any]] = []
    for e in errors[:50]:
        tb = e.get("traceback") or []
        if isinstance(tb, list) and len(tb) > 30:
            tb = tb[-30:]
        err_out.append(
            {
                "ename": e.get("ename", ""),
                "evalue": _truncate_for_judge(str(e.get("evalue", "")), max_chars_per_field),
                "traceback": tb,
            }
        )
    return {
        "text_merged": _truncate_for_judge(str(sig.get("text_merged") or ""), max_chars_per_field),
        "stderr_merged": _truncate_for_judge(str(sig.get("stderr_merged") or ""), max_chars_per_field),
        "png_summary": png_note,
        "errors": err_out,
    }


def parse_notebook_judge_verdict(text: str) -> tuple[Optional[bool], Optional[str], Optional[str]]:
    """Parse VERDICT line and optional REASON. Returns (passed, verdict_label, reason)."""
    has_pass = bool(_JUDGE_VERDICT_PASS_RE.search(text))
    has_fail = bool(_JUDGE_VERDICT_FAIL_RE.search(text))
    reason_m = _JUDGE_REASON_RE.search(text)
    reason = reason_m.group(1).strip() if reason_m else None
    if has_pass == has_fail:
        return None, "judge_parsing_error", reason
    if has_pass:
        return True, "pass", reason
    return False, "fail", reason


def _inputs_to_task_text(params: NeMoGymResponseCreateParamsNonStreaming) -> str:
    """Serialize conversation input into one string for the judge."""
    inp = params.input
    if isinstance(inp, str):
        return inp.strip()
    lines: list[str] = []
    for m in inp:
        if getattr(m, "type", None) != "message":
            continue
        role = getattr(m, "role", None)
        raw = _message_content_to_text(getattr(m, "content", None))
        if raw:
            lines.append(f"{role}:\n{raw}")
    return "\n\n".join(lines).strip() or "(empty task)"


def _resolve_judge_prompt_path(config_path: str) -> Path:
    p = Path(config_path)
    if p.is_absolute():
        return p
    return Path(__file__).resolve().parent / p


def _last_assistant_output_text(resp: NeMoGymResponse) -> Optional[str]:
    try:
        last_output = resp.output[-1]
        if getattr(last_output, "type", None) != "message":
            return None
        last_content = last_output.content[-1]
        return getattr(last_content, "text", "") or ""
    except Exception:
        return None


def _fill_notebook_judge_prompt(
    template: str,
    *,
    task_text: str,
    reference_section: str,
    predicted_code: str,
    predicted_signature_json: str,
) -> str:
    return (
        template.replace("{task_text}", task_text)
        .replace("{reference_section}", reference_section)
        .replace("{predicted_code}", predicted_code)
        .replace("{predicted_signature_json}", predicted_signature_json)
    )


def _comparison_detail_for_judge(ok: Optional[bool], evaluation: NotebookJudgeEvaluation) -> str:
    parts: list[str] = []
    if evaluation.reason:
        parts.append(evaluation.reason)
    if evaluation.verdict_label and evaluation.verdict_label != "pass":
        parts.append(f"verdict_label={evaluation.verdict_label}")
    if parts:
        return "; ".join(parts)
    if ok is True:
        return "pass"
    if ok is None:
        return f"judge_did_not_resolve verdict_label={evaluation.verdict_label or 'unknown'}"
    return f"fail verdict_label={evaluation.verdict_label or 'unknown'}"


# Lines of prior stdout/plain/stderr to include when a code cell times out.
_PREVIOUS_OUTPUT_PREVIEW_LINES = 5
_REFERENCE_EXEC_ERR_MAX_CHARS = 5000
_MAX_CELL_SOURCE_LOG_CHARS = 8_000
_MAX_LOG_BLOCK_CHARS = 120_000


def _write_execution_log_session_header(log_fh: TextIO, log_task_excerpt: Optional[str], cwd: Path) -> None:
    if log_task_excerpt:
        log_fh.write("# Task question\n")
        log_fh.write(log_task_excerpt.strip() + "\n\n")
    log_fh.write(f"Execution log (cwd={cwd})\n")
    log_fh.flush()


def _append_completed_cell_to_execution_log(log_fh: TextIO, code_cell_index: int, nb: NotebookNode) -> None:
    """Append this code cell’s source and output only to the log (one shot per cell, flushed)."""
    if code_cell_index < 0 or code_cell_index >= len(nb.cells):
        return
    cell = nb.cells[code_cell_index]
    src = cell.get("source", "")
    if isinstance(src, list):
        src = "".join(src)
    src = str(src)
    if len(src) > _MAX_CELL_SOURCE_LOG_CHARS:
        src = src[:_MAX_CELL_SOURCE_LOG_CHARS] + "\n... [truncated] ...\n"
    one_cell_nb = nbformat.v4.new_notebook()
    one_cell_nb.metadata = {}
    one_cell_nb.cells = [cell]
    sig = merged_output_signature(one_cell_nb, "none")
    try:
        sig_json = json.dumps(sig, indent=2, default=str)
    except (TypeError, ValueError) as e:  # pragma: no cover - defensive
        sig_json = f"<could not json-encode signature: {e!s}>"
    if len(sig_json) > _MAX_LOG_BLOCK_CHARS:
        sig_json = sig_json[:_MAX_LOG_BLOCK_CHARS] + "\n... [truncated] ...\n"
    log_fh.write(f"---\n# Code cell {code_cell_index} (0-based) completed\n")
    log_fh.write("## Source\n")
    log_fh.write(src)
    log_fh.write("\n## Output (this cell only, PNG omitted from JSON)\n")
    log_fh.write(sig_json)
    log_fh.write("\n")
    log_fh.flush()


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
    import os

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
    log_path: Optional[Path] = None,
    log_task_excerpt: Optional[str] = None,
) -> tuple[Optional[NotebookNode], Optional[str]]:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        cwd = Path(tmp)
        err = _stage_data_paths(cwd, data_paths)
        if err:
            return None, err
        return execute_notebook_cells_process(
            cwd, sources, timeout, wall_margin, log_path=log_path, log_task_excerpt=log_task_excerpt
        )


def _execution_log_basename(execution_log_stem: Optional[str]) -> str:
    """Return a single path segment for `{stem}_reference.log` / `{stem}_predicted.log`."""
    raw = (execution_log_stem or "").strip()
    if not raw or "/" in raw or "\\" in raw or ".." in raw or raw in (".", ".."):
        return uuid.uuid4().hex
    return raw


def _prepare_execution_log_paths(
    log_dir: Optional[str],
    execution_log_stem: Optional[str],
) -> tuple[Optional[Path], Optional[Path]]:
    """If ``log_dir`` is set, ensure it exists and return (reference, predicted) log file paths."""
    if not (log_dir or "").strip():
        return None, None
    base = Path(log_dir).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    stem = _execution_log_basename(execution_log_stem)
    return base / f"{stem}_reference.log", base / f"{stem}_predicted.log"


class DataAnalysisNotebookResourcesServer(SimpleResourcesServer):
    config: DataAnalysisNotebookResourcesServerConfig

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._semaphore: Semaphore = Semaphore(value=self.config.max_concurrent_executions)
        self._judge_semaphore = asyncio.Semaphore(value=self.config.judge_endpoint_max_concurrency)
        tpl_path = _resolve_judge_prompt_path(self.config.judge_prompt_template_fpath)
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
        pred_code_trunc = _truncate_for_judge(predicted_code, max_c * 2)

        user_prompt = _fill_notebook_judge_prompt(
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
        text = _last_assistant_output_text(judge_response)
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

        loop = get_running_loop()
        timeout = self.config.execute_timeout_secs
        margin = self.config.wall_clock_margin_secs
        image_mode = self.config.image_compare_mode
        max_c = self.config.judge_max_output_chars

        log_dir = meta.execution_log_directory or self.config.execution_log_directory
        if (log_dir or "").strip() == "":
            log_dir = None
        ref_log_path, pred_log_path = _prepare_execution_log_paths(log_dir, meta.execution_log_stem)
        log_task = _task_question_excerpt_for_log(body.responses_create_params) if ref_log_path else None

        def _execute_staged(sources: list[str], log_p: Optional[Path]) -> tuple[Optional[NotebookNode], Optional[str]]:
            return _run_staging_and_execute(
                sources,
                meta.data_paths,
                timeout,
                margin,
                log_path=log_p,
                log_task_excerpt=log_task,
            )

        async with self._semaphore:
            ref_nb, ref_err = await loop.run_in_executor(None, partial(_execute_staged, ref_sources, ref_log_path))
            pred_nb, pred_err = await loop.run_in_executor(None, partial(_execute_staged, pred_sources, pred_log_path))

        sig_ref: Optional[dict[str, Any]] = None
        if not ref_err:
            assert ref_nb is not None
            sig_ref = merged_output_signature(ref_nb, image_mode)

        if pred_err:
            pred_exc = pred_err[:_REFERENCE_EXEC_ERR_MAX_CHARS]
            return _verify_response(
                body,
                reward=0.0,
                comparison_detail=f"predicted_execution_failed: {pred_exc}",
                predicted_execution_error=pred_exc,
                reference_merged_output=sig_ref,
                num_reference_code_cells=len(ref_sources),
                num_predicted_code_cells=len(pred_sources),
                reference_execution_error=ref_err[:_REFERENCE_EXEC_ERR_MAX_CHARS] if ref_err else None,
            )

        assert pred_nb is not None
        sig_pred = merged_output_signature(pred_nb, image_mode)

        task_text = _inputs_to_task_text(body.responses_create_params)
        predicted_code_block = "\n\n---\n\n".join(pred_sources)

        if ref_err:
            reference_section = (
                "(Reference notebook did not execute successfully; no reference output to compare.)\n"
                f"Error:\n{ref_err[:_REFERENCE_EXEC_ERR_MAX_CHARS]}"
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

        reward = 1.0 if ok is True else 0.0
        match = ok is True
        comparison_detail = _comparison_detail_for_judge(ok, evaluation)
        if ok is None and evaluation.verdict_label == "judge_parsing_error":
            logger.error(
                "Notebook judge did not return a parseable verdict (verdict_label=%s). reward=0.0.",
                evaluation.verdict_label,
            )

        return _verify_response(
            body,
            reward=reward,
            match=match,
            comparison_detail=comparison_detail,
            reference_merged_output=sig_ref,
            reference_execution_error=ref_err[:_REFERENCE_EXEC_ERR_MAX_CHARS] if ref_err else None,
            predicted_execution_error=None,
            num_reference_code_cells=len(ref_sources),
            num_predicted_code_cells=len(pred_sources),
            judge_evaluations=[evaluation],
        )


if __name__ == "__main__":
    DataAnalysisNotebookResourcesServer.run_webserver()
