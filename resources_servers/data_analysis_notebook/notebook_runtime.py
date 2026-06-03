# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import json
import multiprocessing
import os
import queue
import re
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional, TextIO

import nbformat
from judge import message_content_to_text
from jupyter_client.utils import ensure_async
from jupyter_core.utils import run_sync
from nbclient import NotebookClient
from nbclient.exceptions import CellTimeoutError
from nbformat.notebooknode import NotebookNode

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming


_MAX_TASK_QUESTION_LOG_EXCERPT_CHARS = 500
_MAX_EXECUTION_LOG_STEM_CHARS = 80
_MAX_EXECUTION_LOG_STEM_TASK_FRAGMENT_CHARS = 56
_MAX_EXECUTION_LOG_STEM_DATA_FRAGMENT_CHARS = 28
_MAX_EXECUTION_LOG_STEM_NOTEBOOK_FRAGMENT_CHARS = 40
_PREVIOUS_OUTPUT_PREVIEW_LINES = 5
_EXEC_ERR_MAX_CHARS = 5000
_MAX_CELL_SOURCE_LOG_CHARS = 8_000
_MAX_LOG_BLOCK_CHARS = 120_000

_UNSAFE_LOG_STEM_CHARS_RE = re.compile(r'[/\\:*?"<>|\x00-\x1f]')
_LOG_STEM_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class ReferenceCell:
    cell_type: Literal["code", "markdown"]
    source: str


def first_line_excerpt_for_log(s: str) -> str:
    s = s.strip()
    if not s:
        return ""
    first = s.splitlines()[0].strip()
    if len(first) <= _MAX_TASK_QUESTION_LOG_EXCERPT_CHARS:
        return first
    return first[: _MAX_TASK_QUESTION_LOG_EXCERPT_CHARS - 3] + "..."


def task_question_excerpt_for_log(params: NeMoGymResponseCreateParamsNonStreaming) -> str:
    inp = params.input
    if isinstance(inp, str):
        return first_line_excerpt_for_log(inp)
    for m in reversed(inp):
        if getattr(m, "type", None) != "message" or getattr(m, "role", None) != "user":
            continue
        ex = first_line_excerpt_for_log(message_content_to_text(getattr(m, "content", None)))
        if ex:
            return ex
    return ""


def extract_reference_cells(reference_notebook: dict[str, Any]) -> list[ReferenceCell]:
    nb = nbformat.from_dict(reference_notebook)
    cells: list[ReferenceCell] = []
    for cell in nb.cells:
        ctype = cell.get("cell_type")
        if ctype not in ("code", "markdown"):
            continue
        src = _normalize_cell_source(cell.get("source", ""))
        cells.append(ReferenceCell(cell_type=ctype, source=src))
    return cells


def extract_reference_code_sources(reference_notebook: dict[str, Any]) -> list[str]:
    return [c.source for c in extract_reference_cells(reference_notebook) if c.cell_type == "code" and c.source]


def build_notebook_execution_view(
    cells: list[ReferenceCell],
    executed_nb: Optional[NotebookNode],
    image_mode: Literal["exact", "none"],
) -> dict[str, Any]:
    cells_out: list[dict[str, Any]] = []
    executed_cells = list(executed_nb.cells) if executed_nb is not None else []
    exec_cursor = 0
    for ref_cell in cells:
        if ref_cell.cell_type == "markdown":
            cells_out.append({"cell_type": "markdown", "source": ref_cell.source})
            continue
        outputs: list[dict[str, Any]] = []
        if exec_cursor < len(executed_cells):
            outputs = _filter_png_outputs(_canonicalize_cell_outputs(executed_cells[exec_cursor]), image_mode)
            exec_cursor += 1
        cells_out.append({"cell_type": "code", "source": ref_cell.source, "outputs": outputs})
    return {"cells": cells_out}


def build_predicted_execution_view(
    sources: list[str],
    executed_nb: Optional[NotebookNode],
    image_mode: Literal["exact", "none"],
) -> dict[str, Any]:
    cells = [ReferenceCell(cell_type="code", source=s) for s in sources]
    return build_notebook_execution_view(cells, executed_nb, image_mode)


def truncate_exec_error(err: Optional[str]) -> Optional[str]:
    return err[:_EXEC_ERR_MAX_CHARS] if err else None


def resolve_execution_log_stem(
    explicit: Optional[str],
    params: NeMoGymResponseCreateParamsNonStreaming,
    data_paths: Optional[list[dict[str, str]]],
    reference_notebook: dict[str, Any],
) -> str:
    raw_explicit = (explicit or "").strip()
    if (
        raw_explicit
        and "/" not in raw_explicit
        and "\\" not in raw_explicit
        and ".." not in raw_explicit
        and raw_explicit not in (".", "..")
    ):
        stem = _sanitize_log_stem_fragment(raw_explicit, _MAX_EXECUTION_LOG_STEM_CHARS)
        if stem:
            return stem

    parts: list[str] = []
    task_ex = task_question_excerpt_for_log(params)
    if task_ex:
        frag = _sanitize_log_stem_fragment(task_ex, _MAX_EXECUTION_LOG_STEM_TASK_FRAGMENT_CHARS)
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
    stem = _sanitize_log_stem_fragment("_".join(parts), _MAX_EXECUTION_LOG_STEM_CHARS)
    return stem if stem else uuid.uuid4().hex


def setup_execution_logging(
    log_dir: Optional[str],
    log_stem: str,
    params: NeMoGymResponseCreateParamsNonStreaming,
    *,
    include_pred_log_excerpt: bool,
) -> tuple[Optional[Path], Optional[Path], Optional[str]]:
    ref_log_path, pred_log_path = _prepare_execution_log_paths(log_dir, log_stem)
    need_excerpt = ref_log_path or (pred_log_path and include_pred_log_excerpt)
    log_task = task_question_excerpt_for_log(params) if need_excerpt else None
    return ref_log_path, pred_log_path, log_task


def write_skip_mode_execution_logs(
    ref_log_path: Optional[Path],
    pred_log_path: Optional[Path],
    log_task: Optional[str],
    ref_cells: list[ReferenceCell],
    pred_sources: list[str],
    image_mode: Literal["exact", "none"],
) -> None:
    skip_hdr = " — not executed (verify skipped notebook execution)"
    _write_skip_log(ref_log_path, log_task, ref_cells, image_mode, skip_hdr)
    pred_cells = [ReferenceCell("code", s) for s in pred_sources]
    _write_skip_log(pred_log_path, log_task, pred_cells, image_mode, skip_hdr)


def run_staging_and_execute(
    sources: list[str],
    data_paths: Optional[list[dict[str, str]]],
    timeout: int,
    wall_margin: int,
    log_path: Optional[Path] = None,
    log_task_excerpt: Optional[str] = None,
    log_cell_specs: Optional[list[ReferenceCell]] = None,
) -> tuple[Optional[NotebookNode], Optional[str]]:
    with tempfile.TemporaryDirectory() as tmp:
        cwd = Path(tmp)
        err = _stage_data_paths(cwd, data_paths)
        if err:
            return None, err
        return execute_notebook_cells_process(
            cwd,
            sources,
            timeout,
            wall_margin,
            log_path=log_path,
            log_task_excerpt=log_task_excerpt,
            log_cell_specs=log_cell_specs,
        )


def previous_output_excerpt(nb: NotebookNode, failing_code_cell_index: int, max_lines: int) -> str:
    if failing_code_cell_index <= 0 or not nb.cells:
        return "(no output from previous code cells)"
    try:
        parts: list[str] = []
        for cell in nb.cells[:failing_code_cell_index]:
            for rec in _canonicalize_cell_outputs(cell):
                if rec["kind"] in ("stream", "plain"):
                    text = (rec.get("text") or "").strip()
                    if text:
                        parts.append(text)
    except Exception:  # pragma: no cover
        return "(could not collect output from previous code cells)"
    blob = "\n".join(parts) if parts else "(empty output from previous code cells)"
    lines = blob.split("\n")
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "...\n" + "\n".join(lines[-max_lines:])


def execute_notebook_cells_process(
    cwd: Path,
    sources: list[str],
    timeout: int,
    wall_margin: int,
    log_path: Optional[Path] = None,
    log_task_excerpt: Optional[str] = None,
    log_cell_specs: Optional[list[ReferenceCell]] = None,
) -> tuple[Optional[NotebookNode], Optional[str]]:
    ctx = multiprocessing.get_context("spawn")
    q: multiprocessing.Queue = ctx.Queue(maxsize=1)
    lps: Optional[str] = str(log_path) if log_path is not None else None
    serialized = [(s.cell_type, s.source) for s in log_cell_specs] if log_cell_specs else None
    proc = ctx.Process(
        target=_notebook_worker,
        args=(str(cwd), sources, timeout, q, lps, log_task_excerpt, serialized),
    )
    proc.start()
    limit = timeout * max(len(sources), 1) + wall_margin
    try:
        status, payload = q.get(timeout=limit)
    except queue.Empty:
        if proc.is_alive():
            _terminate_process(proc)
            return None, f"Notebook execution exceeded wall-clock limit ({limit}s)"
        proc.join(timeout=5)
        return None, "Worker produced no result"

    _terminate_process(proc, join_timeout=5)
    if status == "ok":
        return nbformat.reads(payload, as_version=4), None
    return None, str(payload)


def _normalize_cell_source(src: Any) -> str:
    if isinstance(src, list):
        src = "".join(src)
    return str(src).strip()


def _normalize_text(s: str) -> str:
    return "\n".join(line.rstrip() for line in s.replace("\r\n", "\n").strip().split("\n")).strip()


def _filter_png_outputs(outputs: list[dict[str, Any]], image_mode: Literal["exact", "none"]) -> list[dict[str, Any]]:
    if image_mode == "exact":
        return outputs
    return [rec for rec in outputs if rec.get("kind") != "png"]


def _canonicalize_cell_outputs(cell: NotebookNode) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for out in cell.get("outputs", []):
        otype = out.get("output_type")
        if otype == "stream":
            text = out.get("text", "")
            if isinstance(text, list):
                text = "".join(text)
            records.append({"kind": "stream", "name": out.get("name", "stdout"), "text": _normalize_text(text)})
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
                records.append({"kind": "plain", "text": _normalize_text(str(plain))})
            png = data.get("image/png")
            if png is not None:
                png_b64 = png if isinstance(png, str) else base64.b64encode(png).decode("ascii")
                records.append({"kind": "png", "data": png_b64})
    return records


def _stage_data_paths(target: Path, data_paths: Optional[list[dict[str, str]]]) -> Optional[str]:
    if not data_paths:
        return None
    try:
        for item in data_paths:
            src = Path(item["source"]).expanduser().resolve()
            dest = target / item["path"]
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


def _sanitize_log_stem_fragment(s: str, max_len: int) -> str:
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
        for candidate in ((entry.get("path") or "").strip(), (entry.get("source") or "").strip()):
            if not candidate:
                continue
            frag = _sanitize_log_stem_fragment(Path(candidate).name, max_len)
            if frag:
                return frag
    return ""


def _reference_notebook_stem_hint(nb: dict[str, Any], max_len: int) -> str:
    md = nb.get("metadata")
    if not isinstance(md, dict):
        return ""
    title = md.get("title")
    if isinstance(title, str) and title.strip():
        return _sanitize_log_stem_fragment(title.strip(), max_len)
    return ""


def _prepare_execution_log_paths(log_dir: Optional[str], stem: str) -> tuple[Optional[Path], Optional[Path]]:
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


def _write_skip_log(
    path: Optional[Path],
    log_task: Optional[str],
    cells: list[ReferenceCell],
    image_mode: Literal["exact", "none"],
    skip_suffix: str,
) -> None:
    if path is None:
        return
    with open(path, "a", encoding="utf-8", buffering=1) as log_fh:
        _write_log_header(log_fh, log_task, None, execution_skipped=True)
        for nb_idx, spec in enumerate(cells):
            if spec.cell_type == "markdown":
                _write_markdown_log_block(
                    log_fh,
                    spec.source,
                    f"# Markdown cell {nb_idx} (0-based){skip_suffix}",
                )
            else:
                _write_code_log_block(
                    log_fh,
                    spec.source,
                    f"# Code cell {nb_idx} (0-based){skip_suffix}",
                    image_mode,
                )


def _write_log_header(
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
    elif cwd is not None:
        log_fh.write(f"Execution log (cwd={cwd})\n")
    log_fh.flush()


def _truncate_log_text(s: str, max_chars: int = _MAX_CELL_SOURCE_LOG_CHARS) -> str:
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "\n... [truncated] ...\n"


def _write_markdown_log_block(log_fh: TextIO, src: str, header_line: str) -> None:
    log_fh.write(f"---\n{header_line}\n## Source\n{_truncate_log_text(src)}\n")
    log_fh.flush()


def _write_code_log_block(
    log_fh: TextIO,
    src: str,
    header_line: str,
    image_mode: Literal["exact", "none"],
    *,
    outputs: Optional[list[dict[str, Any]]] = None,
    executed_cell: Optional[NotebookNode] = None,
) -> None:
    if outputs is None:
        outputs = (
            _filter_png_outputs(_canonicalize_cell_outputs(executed_cell), image_mode)
            if executed_cell is not None
            else []
        )
    try:
        out_json = json.dumps(outputs, indent=2, default=str)
    except (TypeError, ValueError) as e:  # pragma: no cover
        out_json = f"<could not json-encode outputs: {e!s}>"
    if len(out_json) > _MAX_LOG_BLOCK_CHARS:
        out_json = out_json[:_MAX_LOG_BLOCK_CHARS] + "\n... [truncated] ...\n"
    log_fh.write(
        f"---\n{header_line}\n## Source\n{_truncate_log_text(src)}"
        f"\n## Outputs (this cell only, PNG omitted when configured)\n{out_json}\n"
    )
    log_fh.flush()


def _format_cell_timeout_error(code_cell_index: int, nb: NotebookNode, original_exc: CellTimeoutError) -> str:
    excerpt = previous_output_excerpt(nb, code_cell_index, _PREVIOUS_OUTPUT_PREVIEW_LINES)
    return (
        f"Code cell index {code_cell_index} (0-based) timed out after execute_timeout_secs "
        f"({_PREVIOUS_OUTPUT_PREVIEW_LINES} lines of prior output at most, stdout+stderr):\n"
        f"{excerpt}\n---\n{original_exc!s}"
    )


async def _async_execute_code_cells(
    cwd: Path,
    sources: list[str],
    timeout: int,
    log_path: Optional[Path] = None,
    log_task_excerpt: Optional[str] = None,
    log_cell_specs: Optional[list[ReferenceCell]] = None,
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
                _write_log_header(log_fh, log_task_excerpt, cwd)
            msg_id = await ensure_async(client.kc.kernel_info())
            info_msg = await client.async_wait_for_reply(msg_id)
            if info_msg is None or "language_info" not in info_msg["content"]:
                raise RuntimeError(
                    'Kernel info received message content has no "language_info" key. '
                    "Content is:\n" + str(info_msg["content"] if info_msg else None)
                )
            client.nb.metadata["language_info"] = info_msg["content"]["language_info"]

            code_exec_index = 0
            if log_cell_specs:
                for nb_idx, spec in enumerate(log_cell_specs):
                    if spec.cell_type == "markdown":
                        if log_fh is not None:
                            _write_markdown_log_block(log_fh, spec.source, f"# Markdown cell {nb_idx} (0-based)")
                        continue
                    if code_exec_index >= len(client.nb.cells):
                        break
                    err = await _execute_one_code_cell(client, code_exec_index, log_fh, nb_idx)
                    if err is not None:
                        return None, err
                    code_exec_index += 1
            else:
                for index in range(len(client.nb.cells)):
                    err = await _execute_one_code_cell(client, index, log_fh, index)
                    if err is not None:
                        return None, err
            client.set_widgets_metadata()
        return client.nb, None
    finally:
        if log_fh is not None:
            log_fh.close()


async def _execute_one_code_cell(
    client: NotebookClient,
    code_index: int,
    log_fh: Optional[TextIO],
    log_label_index: int,
) -> Optional[str]:
    if code_index >= len(client.nb.cells):
        return None
    cell = client.nb.cells[code_index]
    try:
        await client.async_execute_cell(cell, code_index, execution_count=client.code_cells_executed + 1)
    except CellTimeoutError as e:
        err = _format_cell_timeout_error(code_index, client.nb, e)
        if log_fh is not None:
            log_fh.write(f"---\n# Code cell {log_label_index} (0-based) CellTimeoutError\n{err}\n")
            log_fh.flush()
        return err
    if log_fh is not None:
        src = _normalize_cell_source(cell.get("source", ""))
        _write_code_log_block(
            log_fh,
            src,
            f"# Code cell {log_label_index} (0-based) completed",
            "none",
            executed_cell=cell,
        )
    return None


@run_sync
async def _execute_notebook_with_timeout_error_detail(
    cwd: Path,
    sources: list[str],
    timeout: int,
    log_path: Optional[Path] = None,
    log_task_excerpt: Optional[str] = None,
    log_cell_specs: Optional[list[ReferenceCell]] = None,
) -> tuple[Optional[NotebookNode], Optional[str]]:
    return await _async_execute_code_cells(cwd, sources, timeout, log_path, log_task_excerpt, log_cell_specs)


def _notebook_worker(
    cwd_str: str,
    sources: list[str],
    timeout: int,
    result_queue: multiprocessing.Queue,
    log_path_str: Optional[str],
    log_task_excerpt: Optional[str],
    log_cell_specs_serialized: Optional[list[tuple[str, str]]],
) -> None:
    try:
        log_specs = (
            [ReferenceCell(cell_type=ctype, source=src) for ctype, src in log_cell_specs_serialized]
            if log_cell_specs_serialized
            else None
        )
        nb, err = _execute_notebook_with_timeout_error_detail(
            Path(cwd_str),
            sources,
            timeout,
            log_path=Path(log_path_str) if log_path_str else None,
            log_task_excerpt=log_task_excerpt,
            log_cell_specs=log_specs,
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
