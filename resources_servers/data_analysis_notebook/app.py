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
import json
import logging
import re
from asyncio import Semaphore, get_running_loop
from functools import partial
from pathlib import Path
from time import sleep
from typing import Any, Literal, Optional

import requests
from judge import (
    NotebookJudgeEvaluation,
    comparison_detail_for_judge,
    fill_notebook_judge_prompt,
    inputs_to_task_text,
    last_assistant_output_text,
    parse_notebook_judge_verdict,
    redact_execution_for_judge,
    truncate_for_judge,
)
from nbformat.notebooknode import NotebookNode
from notebook_runtime import (
    ReferenceCell,
    build_notebook_execution_view,
    build_predicted_execution_view,
    extract_reference_cells,
    extract_reference_code_sources,
    first_line_excerpt_for_log,
    previous_output_excerpt,
    resolve_execution_log_stem,
    run_staging_and_execute,
    setup_execution_logging,
    task_question_excerpt_for_log,
    truncate_exec_error,
    write_skip_mode_execution_logs,
)
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

# Re-export for tests and callers that import from app.

_first_line_excerpt_for_log = first_line_excerpt_for_log
_previous_output_excerpt = previous_output_excerpt
_resolve_execution_log_stem = resolve_execution_log_stem
_task_question_excerpt_for_log = task_question_excerpt_for_log


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
    reference_execution_output: Optional[dict[str, Any]] = Field(
        default=None,
        description=(
            "Per-cell reference notebook view after execution: ordered cells (markdown and code) with "
            "per-output records (stream, plain, png, error) attached to each code cell."
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
    reference_execution_output: Optional[dict[str, Any]] = None,
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
        reference_execution_output=reference_execution_output,
        num_reference_code_cells=num_reference_code_cells,
        num_predicted_code_cells=num_predicted_code_cells,
        judge_evaluations=judge_evaluations,
    )


def _assistant_output_text(response: NeMoGymResponse) -> str:
    parts: list[str] = []
    for output in response.output:
        if output.type != "message" or output.role != "assistant":
            continue
        for content in output.content:
            if content.type == "output_text":
                parts.append(content.text)
    text = _THINKING_SPLIT_RE.split("".join(parts))[-1]
    return _XML_THINKING_RE.sub("", text).strip()


def extract_predicted_code_cells(text: str) -> list[str]:
    """Ordered Python cells from ```python ... ``` fences."""
    if not text or not text.strip():
        return []
    return [m.group(1).strip() for m in _CODE_FENCE_RE.finditer(text) if m.group(1).strip()]


def _reference_section_for_judge(
    *,
    skip_exec: bool,
    ref_err: Optional[str],
    ref_view: dict[str, Any],
    max_chars: int,
) -> str:
    if skip_exec:
        prefix = (
            "(Ground-truth reference only; notebooks were not executed. "
            "No runtime outputs. The predicted execution block below is empty. "
            "Decide PASS/FAIL from the task and cells alone.)\n\n"
        )
        return prefix + json.dumps(redact_execution_for_judge(ref_view, max_chars), indent=2, default=str)
    if ref_err:
        return (
            "(Reference notebook did not execute successfully; no reference output to compare.)\n"
            f"Error:\n{truncate_exec_error(ref_err)}"
        )
    return json.dumps(redact_execution_for_judge(ref_view, max_chars), indent=2, default=str)


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
        predicted_execution: dict[str, Any],
    ) -> tuple[Optional[bool], NotebookJudgeEvaluation]:
        cfg = self.config
        max_c = cfg.judge_max_output_chars
        user_prompt = fill_notebook_judge_prompt(
            self._judge_prompt_template,
            task_text=task_text,
            reference_section=reference_section,
            predicted_code=truncate_for_judge(predicted_code, max_c * 2),
            predicted_execution_json=json.dumps(
                redact_execution_for_judge(predicted_execution, max_c), indent=2, default=str
            ),
        )

        responses_create_params = cfg.judge_responses_create_params.model_copy(deep=True)
        responses_create_params.input = [NeMoGymEasyInputMessage(role="user", content=user_prompt)]

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

    def _execution_log_paths(
        self,
        meta: VerifierMetadata,
        params: NeMoGymResponseCreateParamsNonStreaming,
        *,
        include_pred_log_excerpt: bool,
    ) -> tuple[Optional[Path], Optional[Path], Optional[str]]:
        log_dir = (meta.execution_log_directory or self.config.execution_log_directory or "").strip() or None
        stem = resolve_execution_log_stem(meta.execution_log_stem, params, meta.data_paths, meta.reference_notebook)
        return setup_execution_logging(log_dir, stem, params, include_pred_log_excerpt=include_pred_log_excerpt)

    async def verify(self, body: DataAnalysisNotebookVerifyRequest) -> DataAnalysisNotebookVerifyResponse:
        if not body.verifier_metadata:
            return _verify_response(body, reward=0.0, comparison_detail="verifier_metadata is missing")
        try:
            meta = VerifierMetadata.model_validate(body.verifier_metadata)
        except ValidationError as e:
            return _verify_response(body, reward=0.0, comparison_detail=f"invalid verifier_metadata: {e}"[:2000])

        ref_cells = extract_reference_cells(meta.reference_notebook)
        ref_sources = [c.source for c in ref_cells if c.cell_type == "code" and c.source]
        pred_sources = extract_predicted_code_cells(_assistant_output_text(body.response))
        n_ref, n_pred = len(ref_sources), len(pred_sources)

        if not ref_sources:
            return _verify_response(
                body,
                reward=0.0,
                comparison_detail="reference notebook has no code cells",
                num_predicted_code_cells=n_pred,
            )
        if not pred_sources:
            return _verify_response(
                body,
                reward=0.0,
                comparison_detail="model output has no python code fences",
                num_reference_code_cells=n_ref,
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

        ref_log, pred_log, log_task = self._execution_log_paths(
            meta, body.responses_create_params, include_pred_log_excerpt=skip_exec
        )

        ref_nb: Optional[NotebookNode] = None
        pred_nb: Optional[NotebookNode] = None
        ref_err: Optional[str] = None
        pred_err: Optional[str] = None

        if skip_exec:
            write_skip_mode_execution_logs(ref_log, pred_log, log_task, ref_cells, pred_sources, image_mode)
        else:

            def _execute(
                sources: list[str],
                log_p: Optional[Path],
                cell_specs: Optional[list[ReferenceCell]] = None,
            ) -> tuple[Optional[NotebookNode], Optional[str]]:
                return run_staging_and_execute(
                    sources,
                    meta.data_paths,
                    timeout,
                    margin,
                    log_path=log_p,
                    log_task_excerpt=log_task,
                    log_cell_specs=cell_specs,
                )

            loop = get_running_loop()
            async with self._semaphore:
                ref_nb, ref_err = await loop.run_in_executor(None, partial(_execute, ref_sources, ref_log, ref_cells))
                pred_nb, pred_err = await loop.run_in_executor(None, partial(_execute, pred_sources, pred_log, None))

            if pred_err:
                ref_view = build_notebook_execution_view(ref_cells, ref_nb, image_mode) if not ref_err else None
                return _verify_response(
                    body,
                    reward=0.0,
                    comparison_detail=f"predicted_execution_failed: {truncate_exec_error(pred_err)}",
                    predicted_execution_error=truncate_exec_error(pred_err),
                    reference_execution_output=ref_view,
                    num_reference_code_cells=n_ref,
                    num_predicted_code_cells=n_pred,
                    reference_execution_error=truncate_exec_error(ref_err),
                )

        ref_view = build_notebook_execution_view(ref_cells, ref_nb, image_mode)
        pred_view = build_predicted_execution_view(pred_sources, pred_nb, image_mode)
        task_text = inputs_to_task_text(body.responses_create_params)
        predicted_code = "\n\n---\n\n".join(pred_sources)

        ok, evaluation = await self._run_llm_judge(
            task_text=task_text,
            reference_section=_reference_section_for_judge(
                skip_exec=skip_exec, ref_err=ref_err, ref_view=ref_view, max_chars=max_c
            ),
            predicted_code=predicted_code,
            predicted_execution=pred_view,
        )

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
            reference_execution_output=None if skip_exec or ref_err else ref_view,
            reference_execution_error=truncate_exec_error(ref_err),
            num_reference_code_cells=n_ref,
            num_predicted_code_cells=n_pred,
            judge_evaluations=[evaluation],
        )


if __name__ == "__main__":
    DataAnalysisNotebookResourcesServer.run_webserver()
