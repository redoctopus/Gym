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


def parse_notebook_judge_verdict(text: str) -> tuple[Optional[bool], Optional[str], Optional[str]]:
    """Parse VERDICT line and optional REASON. Returns (passed, verdict_label, reason)."""
    has_pass = bool(_JUDGE_VERDICT_PASS_RE.search(text))
    has_fail = bool(_JUDGE_VERDICT_FAIL_RE.search(text))
    reason_m = _JUDGE_REASON_RE.search(text)
    reason = reason_m.group(1).strip() if reason_m else None
    if has_pass == has_fail:
        return None, "judge_parsing_error", text
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
