# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import multiprocessing
import os
import queue
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional

import nbformat
from jupyter_client.utils import ensure_async
from jupyter_core.utils import run_sync
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, CellTimeoutError
from nbformat.notebooknode import NotebookNode

def run_staging_and_execute(
    nb: NotebookNode,
    data_paths: Optional[list[dict[str, str]]],
    timeout: int,
    wall_margin: int,
    write_nb: bool = False,
    notebook_output_fpath: Optional[Path] = None,
) -> tuple[Optional[NotebookNode], Optional[str]]:
    with tempfile.TemporaryDirectory() as tmp:
        cwd = Path(tmp)
        _stage_data_paths(cwd, data_paths)
        return execute_notebook_cells_process(
            cwd, nb, timeout, wall_margin, write_nb, notebook_output_fpath
        )

def execute_notebook_cells_process(
    cwd: Path,
    nb: NotebookNode,
    timeout: int,
    wall_margin: int,
    write_nb: bool = False,
    notebook_output_fpath: Optional[Path] = None,
) -> tuple[Optional[NotebookNode], Optional[str]]:
    ctx = multiprocessing.get_context("spawn")
    q: multiprocessing.Queue = ctx.Queue(maxsize=1)
    output_fpath = str(notebook_output_fpath) if notebook_output_fpath is not None else None
    proc = ctx.Process(
        target=_notebook_worker,
        args=(str(cwd), nb, timeout, q, write_nb, output_fpath),
    )
    proc.start()
    limit = timeout * max(len(nb.cells), 1) + wall_margin
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

_EXEC_ERR_MAX_CHARS = 5000


def truncate_exec_error(err: Optional[str]) -> Optional[str]:
    return err[:_EXEC_ERR_MAX_CHARS] if err else None


def _stage_data_paths(target: Path, data_paths: Optional[list[dict[str, str]]]) -> None:
    if not data_paths:
        return
    for item in data_paths:
        src = Path(item["source"]).expanduser().resolve()
        dest = target / item["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not src.exists():
            raise RuntimeError(f"data_paths: source does not exist: {src}")
        if src.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)
        else:
            shutil.copy2(src, dest)


@run_sync
async def _execute_code_cells(
    cwd: Path,
    nb: NotebookNode,
    timeout: int,
    write_nb: bool = False,
    notebook_output_fpath: Optional[Path] = None,
) -> tuple[Optional[NotebookNode], Optional[str]]:
    os.environ["MPLBACKEND"] = "Agg"
    os.chdir(cwd)
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(cwd)}},
    )
    client.reset_execution_trackers()

    executed_notebook_path: Optional[Path] = None
    if write_nb and notebook_output_fpath is not None:
        executed_notebook_path = notebook_output_fpath
        executed_notebook_path.parent.mkdir(parents=True, exist_ok=True)
        executed_notebook_path.write_text("", encoding="utf-8")

    async with client.async_setup_kernel():
        assert client.kc is not None
        msg_id = await ensure_async(client.kc.kernel_info())
        info_msg = await client.async_wait_for_reply(msg_id)
        if info_msg is None or "language_info" not in info_msg["content"]:
            raise RuntimeError(
                'Kernel info received message content has no "language_info" key. '
                "Content is:\n" + str(info_msg["content"] if info_msg else None)
            )
        client.nb.metadata["language_info"] = info_msg["content"]["language_info"]

        for index in range(len(client.nb.cells)):
            cell = client.nb.cells[index]
            try:
                await client.async_execute_cell(cell, index, execution_count=client.code_cells_executed + 1)

            except CellTimeoutError as e:
                # Output logging with added timeout error
                if executed_notebook_path is not None:
                    err_output = nbformat.v4.new_output("error", ename="CellTimeoutError", evalue=str(e))
                    cell.outputs.append(err_output)
                    executed_notebook_path.write_text(nbformat.writes(client.nb), encoding="utf-8")
                        
                return None, (
                    f"Code cell index {index} (0-based) timed out after execute_timeout_secs"
                )
            except CellExecutionError as e:
                if executed_notebook_path is not None:
                    if not any(o.get("output_type") == "error" for o in cell.get("outputs", [])):
                        tb = e.traceback.splitlines(keepends=True) if e.traceback else []
                        cell.outputs.append(
                            nbformat.v4.new_output("error", ename=e.ename, evalue=e.evalue, traceback=tb)
                        )
                    executed_notebook_path.write_text(nbformat.writes(client.nb), encoding="utf-8")
                return None, str(e)
        client.set_widgets_metadata()

    # Output logging
    if executed_notebook_path is not None:
        executed_notebook_path.write_text(nbformat.writes(client.nb), encoding="utf-8")

    return client.nb, None


def _notebook_worker(
    cwd_str: str,
    nb: NotebookNode,
    timeout: int,
    result_queue: multiprocessing.Queue,
    write_nb: bool = False,
    notebook_output_fpath: Optional[str] = None,
) -> None:
    try:
        output_path = Path(notebook_output_fpath) if notebook_output_fpath else None
        nb_executed, err = _execute_code_cells(
            Path(cwd_str), nb, timeout, write_nb, output_path
        )
        if err is not None:
            result_queue.put(("err", err))
            return
        assert nb_executed is not None
        result_queue.put(("ok", nbformat.writes(nb_executed)))
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
