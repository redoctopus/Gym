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

import re
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)


_JUDGE_VERDICT_PASS_RE = re.compile(r"^\s*VERDICT:\s*PASS\s*$", re.MULTILINE | re.IGNORECASE)
_JUDGE_VERDICT_FAIL_RE = re.compile(r"^\s*VERDICT:\s*FAIL\s*$", re.MULTILINE | re.IGNORECASE)
_JUDGE_REASON_RE = re.compile(r"^\s*REASON:\s*(.+?)(?:\n|$)", re.MULTILINE | re.DOTALL | re.IGNORECASE)


class NotebookJudgeEvaluation(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    response: Optional[NeMoGymResponse] = None
    verdict_label: Optional[str] = None
    reason: Optional[str] = None


def message_content_to_text(content: Any) -> str:
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


def truncate_for_judge(s: str, max_chars: int) -> str:
    s = s or ""
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 20] + "\n...[truncated]...\n"


def _redact_output_record(rec: dict[str, Any], max_chars_per_field: int) -> dict[str, Any]:
    kind = rec.get("kind")
    if kind == "stream":
        return {
            "kind": "stream",
            "name": rec.get("name", "stdout"),
            "text": truncate_for_judge(str(rec.get("text") or ""), max_chars_per_field),
        }
    if kind == "plain":
        return {"kind": "plain", "text": truncate_for_judge(str(rec.get("text") or ""), max_chars_per_field)}
    if kind == "png":
        return {"kind": "png", "omitted": True}
    if kind == "error":
        tb = rec.get("traceback") or []
        if isinstance(tb, list) and len(tb) > 30:
            tb = tb[-30:]
        return {
            "kind": "error",
            "ename": rec.get("ename", ""),
            "evalue": truncate_for_judge(str(rec.get("evalue", "")), max_chars_per_field),
            "traceback": tb,
        }
    return dict(rec)


def redact_execution_for_judge(view: dict[str, Any], max_chars_per_field: int) -> dict[str, Any]:
    """Drop PNG payloads; truncate text fields for the judge prompt."""
    cells_out: list[dict[str, Any]] = []
    total_pngs = 0
    for cell in view.get("cells") or []:
        ctype = cell.get("cell_type")
        if ctype == "markdown":
            cells_out.append(
                {
                    "cell_type": "markdown",
                    "source": truncate_for_judge(str(cell.get("source") or ""), max_chars_per_field),
                }
            )
            continue
        if ctype != "code":
            continue
        outputs_out = [_redact_output_record(rec, max_chars_per_field) for rec in (cell.get("outputs") or [])]
        cell_pngs = sum(1 for rec in outputs_out if rec.get("kind") == "png")
        total_pngs += cell_pngs
        code_cell: dict[str, Any] = {
            "cell_type": "code",
            "source": truncate_for_judge(str(cell.get("source") or ""), max_chars_per_field),
            "outputs": outputs_out,
        }
        if cell_pngs:
            code_cell["png_count"] = cell_pngs
        cells_out.append(code_cell)
    result: dict[str, Any] = {"cells": cells_out}
    if total_pngs:
        result["png_summary"] = f"{total_pngs} PNG figure(s) across cells; pixel data omitted."
    return result


def parse_notebook_judge_verdict(text: str) -> tuple[Optional[bool], Optional[str], Optional[str]]:
    """Parse VERDICT line and optional REASON. Returns (passed, verdict_label, reason)."""
    has_pass = bool(_JUDGE_VERDICT_PASS_RE.search(text))
    has_fail = bool(_JUDGE_VERDICT_FAIL_RE.search(text))
    reason_m = _JUDGE_REASON_RE.search(text)
    reason = reason_m.group(1).strip() if reason_m else None
    if has_pass == has_fail:
        return None, "judge_parsing_error", reason
    return (True, "pass", reason) if has_pass else (False, "fail", reason)


def inputs_to_task_text(params: NeMoGymResponseCreateParamsNonStreaming) -> str:
    """Serialize conversation input into one string for the judge."""
    inp = params.input
    if isinstance(inp, str):
        return inp.strip()
    lines: list[str] = []
    for m in inp:
        if getattr(m, "type", None) != "message":
            continue
        role = getattr(m, "role", None)
        raw = message_content_to_text(getattr(m, "content", None))
        if raw:
            lines.append(f"{role}:\n{raw}")
    return "\n\n".join(lines).strip() or "(empty task)"


def last_assistant_output_text(resp: NeMoGymResponse) -> Optional[str]:
    try:
        last_output = resp.output[-1]
        if getattr(last_output, "type", None) != "message":
            return None
        last_content = last_output.content[-1]
        return getattr(last_content, "text", "") or ""
    except Exception:
        return None


def fill_notebook_judge_prompt(
    template: str,
    *,
    task_text: str,
    reference_section: str,
    predicted_code: str,
    predicted_execution_json: str,
) -> str:
    for key, value in (
        ("{task_text}", task_text),
        ("{reference_section}", reference_section),
        ("{predicted_code}", predicted_code),
        ("{predicted_execution_json}", predicted_execution_json),
    ):
        template = template.replace(key, value)
    return template


def comparison_detail_for_judge(ok: Optional[bool], evaluation: NotebookJudgeEvaluation) -> str:
    parts: list[str] = []
    if evaluation.reason:
        parts.append(evaluation.reason)
    if evaluation.verdict_label and evaluation.verdict_label != "pass":
        parts.append(f"verdict_label={evaluation.verdict_label}")
    if parts:
        return "; ".join(parts)
    if ok is True:
        return "pass"
    label = evaluation.verdict_label or "unknown"
    prefix = "judge_did_not_resolve" if ok is None else "fail"
    return f"{prefix} verdict_label={label}"
