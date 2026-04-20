#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# -----------------------------------------------------------------------------
# Kaggle-style .txt → NeMo-Gym JSONL for ``data_analysis_notebook``.
#
# Run (from anywhere):
#   python convert_txt_input_to_jsonl.py <path/to/tasks.txt> [--kaggle-input-root <dir>]
#
# Defaults: ``--kaggle-input-root`` is ``<this script>/kaggle/input`` if omitted.
# Output: ``<path/to/tasks.jsonl>`` next to the input (same stem, ``.jsonl``).
#
# Input (.txt, UTF-8): repeated *blocks*. Each block = one problem line, then a JSON array of
# notebook-like cells (``cell_type`` ``code``|``markdown``, ``source`` string or list), then optional
# ``data_path:`` / ``data_paths:`` / ``data_files:`` lines until a blank line. Paths may use
# ``/kaggle/input/...``; those are remapped using the Kaggle input root.
#
# Output (.jsonl): one JSON object per line — ``id``, ``responses_create_params`` (system + user),
# ``verifier_metadata`` with ``reference_notebook`` and optional ``data_paths`` / ``data_files``.
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


def _parse_quoted_or_bare_path(raw: str) -> str:
    """Normalize a ``data_path:`` value: strip quotes and fix escaped spaces/quotes inside quoted paths."""
    raw = raw.strip()
    if raw.startswith('"') and raw.endswith('"'):
        inner = raw[1:-1].replace("\\ ", " ").replace('\\"', '"')
        return inner
    return raw


def _parse_optional_line(line: str) -> tuple[str, Any]:
    """Parse a single optional metadata line after the cell JSON: ``data_path``, ``data_paths``, or ``data_files``."""
    stripped = line.strip()
    if stripped.startswith("data_path:"):
        val = stripped.split(":", 1)[1].strip()
        return ("data_path", _parse_quoted_or_bare_path(val))
    if stripped.startswith("data_paths:"):
        rest = stripped.split(":", 1)[1].strip()
        return ("data_paths", json.loads(rest))
    if stripped.startswith("data_files:"):
        rest = stripped.split(":", 1)[1].strip()
        return ("data_files", json.loads(rest))
    raise ValueError(f"Unrecognized optional line: {stripped[:120]!r}")


def _iter_txt_blocks(txt_path: Path) -> list[dict[str, Any]]:
    """Scan the full .txt into a list of per-task dicts: ``problem``, ``cells_raw``, optional path fields."""
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
                if key == "data_path":
                    opt["data_path"] = val
                elif key == "data_paths":
                    opt["data_paths_override"] = val
                elif key == "data_files":
                    opt["data_files"] = val
                nextline = f_in.readline()

            blocks.append({"problem": problem, "cells_raw": cells_raw, **opt})

    return blocks


def _kaggle_suffix_from_abs_path(abs_path: str) -> str | None:
    """If ``abs_path`` is under ``/kaggle/input/``, return the relative suffix; else ``None``."""
    prefix = "/kaggle/input/"
    if not abs_path.startswith(prefix):
        return None
    return abs_path[len(prefix) :].lstrip("/")


def _discover_suffixes_from_tree(combined_text: str, kaggle_input_root: Path) -> list[str]:
    """Find files under ``kaggle_input_root`` whose ``/kaggle/input/<relpath>`` string appears in ``combined_text``."""
    if not kaggle_input_root.is_dir():
        return []
    found: list[str] = []
    for p in sorted(kaggle_input_root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(kaggle_input_root).as_posix()
        needle = "/kaggle/input/" + rel
        if needle in combined_text:
            found.append(rel)
    return found


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


def _remap_kaggle_paths(text: str, suffixes: list[str]) -> str:
    """Replace ``/kaggle/input/<suffix>`` with relative ``suffix`` so paths match CWD layout in the env."""
    for suf in sorted(set(suffixes), key=len, reverse=True):
        old = "/kaggle/input/" + suf
        text = text.replace(old, suf)
    text = text.replace("'/kaggle/input'", "'.'")
    text = text.replace('"/kaggle/input"', '"."')
    return text


def _build_data_paths(suffixes: list[str], kaggle_input_root: Path) -> list[dict[str, str]] | None:
    """Map discovered relative suffixes to absolute ``source`` paths under ``kaggle_input_root`` for verifier staging."""
    rel_set: set[str] = set(suffixes)

    entries: list[dict[str, str]] = []
    for rel in sorted(rel_set):
        src = (kaggle_input_root / rel).resolve()
        if not src.exists():
            print(f"Warning: missing kaggle input file (skipping data_paths entry): {src}", file=sys.stderr)
            continue
        entries.append({"source": str(src), "path": rel})

    return entries or None


def _record_to_jsonl(
    task_id: int,
    problem: str,
    reference_notebook: dict[str, Any],
    data_paths: list[dict[str, str]] | None,
    data_files: dict[str, str] | None,
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
    if data_files is not None:
        row["verifier_metadata"]["data_files"] = data_files
    if data_paths is not None:
        row["verifier_metadata"]["data_paths"] = data_paths
    return row


def convert_txt_to_jsonl(
    txt_path: Path,
    out_path: Path,
    *,
    kaggle_input_root: Path,
) -> None:
    """Parse ``txt_path``, remap Kaggle paths, attach data file metadata, write ``out_path`` (one JSON per line)."""
    blocks = _iter_txt_blocks(txt_path)
    lines_out: list[str] = []

    for idx, block in enumerate(blocks):
        problem = block["problem"]
        cells_raw = block["cells_raw"]
        data_files = block.get("data_files")
        override_paths = block.get("data_paths_override")

        # Match files on disk whose /kaggle/input/<relpath> substring appears in the problem or cells.
        combined = problem + "\n" + json.dumps(cells_raw, ensure_ascii=False)
        discovered = _discover_suffixes_from_tree(combined, kaggle_input_root)

        explicit = block.get("data_path")
        if explicit:
            esuf = _kaggle_suffix_from_abs_path(explicit)
            if esuf:
                discovered = list({*discovered, esuf})

        remapped_problem = _remap_kaggle_paths(problem, discovered)
        remapped_cells: list[Any] = []
        for c in cells_raw:
            nc = dict(c)
            src = nc.get("source", "")
            if isinstance(src, list):
                src = "".join(src)
            nc["source"] = _remap_kaggle_paths(str(src), discovered)
            remapped_cells.append(nc)

        ref_nb = _cells_to_reference_notebook(remapped_cells)

        if override_paths is not None:
            data_paths = override_paths
        elif data_files is not None:
            data_paths = None
        else:
            data_paths = _build_data_paths(discovered, kaggle_input_root)

        row = _record_to_jsonl(idx, remapped_problem, ref_nb, data_paths, data_files)
        lines_out.append(json.dumps(row, ensure_ascii=False))

    out_path.write_text("\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")


def main() -> None:
    """CLI: parse args, default Kaggle root next to this file, run conversion and print output path."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "input_txt",
        type=Path,
        help="Path to {dir}/{name}.txt",
    )
    p.add_argument(
        "--kaggle-input-root",
        type=Path,
        default=None,
        help="Directory mirroring /kaggle/input (default: <this_dir>/kaggle/input)",
    )
    args = p.parse_args()
    inp = args.input_txt.resolve()
    if not inp.is_file():
        sys.exit(f"Input not found: {inp}")
    out = inp.with_suffix(".jsonl")
    root = args.kaggle_input_root or (Path(__file__).resolve().parent / "kaggle" / "input")
    convert_txt_to_jsonl(inp, out, kaggle_input_root=root.resolve())
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
