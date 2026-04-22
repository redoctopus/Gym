#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# -----------------------------------------------------------------------------
# Kaggle-style .txt → NeMo-Gym JSONL for ``data_analysis_notebook``.
#
# Run (from anywhere):
#   python convert_txt_input_to_jsonl.py <path/to/tasks.txt>
#
# ``script_dir`` (for resolving ``data_paths``) defaults to the directory containing
# this script. Output: ``<path/to/tasks.jsonl>`` next to the input (same stem, ``.jsonl``).
#
# Input (.txt, UTF-8): repeated *blocks*. Each block = one problem line, then a JSON array of
# notebook-like cells (``cell_type`` ``code``|``markdown``, ``source`` string or list), then optional
# ``data_paths:`` line until a blank line. In problem and cell text, ``/kaggle/input/<rest>`` is
# replaced with ``<rest>``. Exception: if the path is just ``/kaggle/input/``, it is replaced with ``input/``.
# Optional ``data_paths`` is a JSON array of paths relative to ``script_dir``
# (files or directories on disk); each must exist before conversion succeeds.
#
# Output (.jsonl): one JSON object per line — ``id``, ``responses_create_params`` (system + user),
# ``verifier_metadata`` with ``reference_notebook`` and optional ``data_paths`` (``source`` + ``path`` pairs).
# -----------------------------------------------------------------------------

"""Convert Kaggle-style .txt task blocks into NeMo-Gym JSONL for data_analysis_notebook."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


_SYSTEM_PROMPT = (
    "You solve data analysis tasks with Python only. Output one or more ```python code blocks in order. "
    "Use print() for text results and call data frames directly to display them; matplotlib may be used (Agg backend). "
    "Data files listed in the task are in the current working directory."
)

def _remap_kaggle_input_paths(text: str) -> str:
    """Replace ``/kaggle/input/<path>`` with ``<path>`` everywhere in ``text``."""
    return text.replace('/kaggle/input/', "").replace('/kaggle/input', "input")


def _parse_optional_line(line: str) -> tuple[str, Any]:
    """Parse optional metadata line after the cell JSON: ``data_paths`` only."""
    stripped = line.strip()
    if stripped.startswith("data_paths:"):
        rest = stripped.split(":", 1)[1].strip()
        return ("data_paths", json.loads(rest))
    raise ValueError(f"Unrecognized optional line: {stripped[:120]!r}")


def _iter_txt_blocks(txt_path: Path) -> list[dict[str, Any]]:
    """Scan the full .txt into a list of per-task dicts: ``problem``, ``cells_raw``, optional ``data_paths``."""
    blocks = []
    decoder = json.JSONDecoder()

    with open(txt_path, "r", encoding="utf-8") as f_in:
        while True:
            # Retrieve problem statement
            problem = f_in.readline()
            if not problem:
                break
            problem = problem.strip()

            # Retrieve JSON array of reference notebook cells
            json_lines = ""
            nextline = f_in.readline()
            while nextline != "]\n":
                json_lines += nextline
                nextline = f_in.readline()
            json_lines += nextline

            # Parse JSON array of reference notebook cells
            try:
                cells_raw, _ = decoder.raw_decode(json_lines)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON cell list after problem {problem[:60]!r}: {e}") from e

            if not isinstance(cells_raw, list):
                raise ValueError("Cell payload must be a JSON array")

            # Retrieve optional metadata lines (data paths or data files)
            opt: dict[str, Any] = {}
            nextline = f_in.readline()
            while nextline and nextline != "\n":
                key, val = _parse_optional_line(nextline)
                if key == "data_paths":
                    opt["data_paths"] = val
                nextline = f_in.readline()

            blocks.append({"problem": problem, "cells_raw": cells_raw, **opt})

    return blocks


def _cells_to_reference_notebook(cells_raw: list[Any]) -> dict[str, Any]:
    """Build a minimal nbformat-4 notebook dict (cells only; used as verifier ground truth)."""
    cells_out: list[dict[str, Any]] = []
    for c in cells_raw:
        if not isinstance(c, dict):
            raise ValueError("Each cell must be a JSON object")
        ctype = c.get("cell_type")
        if ctype not in ("code", "markdown"):
            raise ValueError(f"Unsupported cell_type: {ctype!r}")
        src = c.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        cells_out.append({"cell_type": ctype, "metadata": {}, "outputs": [], "source": str(src)})
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": cells_out,
    }


def _normalize_input_rel(rel: str) -> str:
    """Strip leading ``./`` segments for stable prefix checks."""
    p = rel.replace("\\", "/").strip()
    while p.startswith("./"):
        p = p[2:]
    return p


def _staging_path_from_input_rel(input_rel: str) -> str:
    """
    Verifier ``path`` is CWD-relative without the Kaggle input prefix, matching prior JSONL output.
    Input entries are relative to ``script_dir``; if under ``kaggle/input/``, strip that prefix.
    """
    n = _normalize_input_rel(input_rel)
    kaggle_lead = "kaggle/input/"
    if n == kaggle_lead:
        return "input/"
    if n.startswith(kaggle_lead):
        n = n[len(kaggle_lead) :]
    if not n or n == ".":
        return "."
    return n


def _build_verifier_data_paths(script_dir: Path, input_rels: list[Any]) -> list[dict[str, str]]:
    """Resolve each input path under ``script_dir``, validate existence, emit ``source`` / ``path`` pairs."""
    if not isinstance(input_rels, list):
        raise TypeError("data_paths must be a JSON array of path strings")
    entries: list[dict[str, str]] = []
    for raw in input_rels:
        if not isinstance(raw, str):
            raise TypeError("Each data_paths entry must be a string path")
        if Path(raw).is_absolute():
            raise ValueError(f"data_paths entry must be relative to the script directory: {raw!r}")
        abs_src = (script_dir / raw).resolve()
        if not abs_src.exists():
            raise FileNotFoundError(f"data_paths entry does not exist: {abs_src} (from {raw!r})")
        staging = _staging_path_from_input_rel(raw)
        if abs_src.is_dir() and not staging.endswith("/") and staging != ".":
            staging = f"{staging}/"
        entries.append({"source": str(abs_src), "path": staging})
    return entries


def _record_to_jsonl(
    task_id: int,
    problem: str,
    reference_notebook: dict[str, Any],
    data_paths: list[dict[str, str]] | None,
) -> dict[str, Any]:
    """Assemble one Gym JSONL row: chat input + verifier metadata for the resources server."""
    row: dict[str, Any] = {
        "id": task_id,
        "responses_create_params": {
            "input": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": problem},
            ],
            "parallel_tool_calls": False,
        },
        "verifier_metadata": {"reference_notebook": reference_notebook},
    }
    if data_paths is not None:
        row["verifier_metadata"]["data_paths"] = data_paths
    return row


def convert_txt_to_jsonl(
    txt_path: Path,
    out_path: Path,
    *,
    script_dir: Path,
) -> None:
    """Parse ``txt_path``, remap ``/kaggle/input/...``, attach ``data_paths``, write ``out_path``."""
    blocks = _iter_txt_blocks(txt_path)
    lines_out: list[str] = []

    for idx, block in enumerate(blocks):
        problem = _remap_kaggle_input_paths(block["problem"])
        cells_raw = block["cells_raw"]
        data_paths_input = block.get("data_paths")

        remapped_cells: list[Any] = []
        for c in cells_raw:
            nc = dict(c)
            src = nc.get("source", "")
            if isinstance(src, list):
                src = "".join(src)
            nc["source"] = _remap_kaggle_input_paths(str(src))
            remapped_cells.append(nc)

        ref_nb = _cells_to_reference_notebook(remapped_cells)

        data_paths: list[dict[str, str]] | None = None
        if data_paths_input is not None:
            data_paths = _build_verifier_data_paths(script_dir, data_paths_input)

        row = _record_to_jsonl(idx, problem, ref_nb, data_paths)
        lines_out.append(json.dumps(row, ensure_ascii=False))

    out_path.write_text("\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")


def main() -> None:
    """CLI: parse args, default script dir to this file's parent, run conversion and print output path."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "input_txt",
        type=Path,
        help="Path to {dir}/{name}.txt",
    )
    args = p.parse_args()
    inp = args.input_txt.resolve()
    if not inp.is_file():
        sys.exit(f"Input not found: {inp}")
    out = inp.with_suffix(".jsonl")
    script_dir = Path(__file__).resolve().parent
    try:
        convert_txt_to_jsonl(inp, out, script_dir=script_dir)
    except (FileNotFoundError, ValueError, TypeError, json.JSONDecodeError) as e:
        sys.exit(f"Conversion failed: {e}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
