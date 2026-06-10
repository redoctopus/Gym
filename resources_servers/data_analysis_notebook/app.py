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
import nbformat
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
    inputs_to_task_text,
    last_assistant_output_text,
    parse_notebook_judge_verdict,
    redact_execution_for_judge,
    truncate_for_judge,
)
from nbformat.notebooknode import NotebookNode
from notebook_runtime import (
    build_notebook_execution_view,
    run_staging_and_execute,
    truncate_exec_error,
)
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

_XML_THINKING_RE = re.compile(r"<thinking>.*?</thinking>", re.DOTALL | re.IGNORECASE)
_PREDICTED_CELL_FENCE_RE = re.compile(
    r"```(python|py|markdown)?\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


class DataAnalysisNotebookResourcesServerConfig(BaseResourcesServerConfig):
    max_concurrent_executions: int = 2
    execute_timeout_secs: int = 120
    wall_clock_margin_secs: int = 90
    # When true, skip staging and subprocess execution for reference and predicted code; the judge
    # sees task text, reference source, predicted code, and empty execution signatures only.
    skip_notebook_execution: bool = False
    image_compare_mode: Literal["exact", "none"] = "exact"
    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    judge_endpoint_max_concurrency: int = 32
    judge_prompt_template_fpath: str = "prompt_templates/notebook_task_judge.txt"
    judge_max_output_chars: int = 12000
    judge_probe_on_startup: bool = True
    executed_notebooks_directory: str = "executed_notebooks"


class VerifierMetadata(BaseModel):
    """Fields expected under verifier_metadata for this server."""

    reference_notebook: dict[str, Any]
    data_paths: Optional[list[dict[str, str]]] = None
    # If set, overrides `skip_notebook_execution` from resources server config for this request.
    skip_notebook_execution: Optional[bool] = None


class DataAnalysisNotebookVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")

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
        num_reference_code_cells=num_reference_code_cells,
        num_predicted_code_cells=num_predicted_code_cells,
        judge_evaluations=judge_evaluations,
    )

def _strip_thinking_blocks(text: str) -> str:
    """Drop reasoning preambles before parsing fenced notebook cells."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    elif "<think>" in text:
        return ""
    text = _XML_THINKING_RE.sub("", text)
    return text.strip()


def extract_predicted_notebook(text: str) -> NotebookNode:
    """Ordered code and markdown cells from ```python```/```py``` and ```markdown``` fences."""
    nb = nbformat.v4.new_notebook()
    if not text or not text.strip():
        return nb
    text = _strip_thinking_blocks(text)
    if not text:
        return nb
    for match in _PREDICTED_CELL_FENCE_RE.finditer(text):
        lang = (match.group(1) or "python").lower()
        source = match.group(2).strip()
        if not source:  # Skip empty cells
            continue
        if lang == "markdown":
            nb.cells.append(nbformat.v4.new_markdown_cell(source=source))
        else:
            nb.cells.append(nbformat.v4.new_code_cell(source=source))
    return nb


def _executed_notebook_stem(body: DataAnalysisNotebookVerifyRequest) -> str:
    dump = body.model_dump()
    if (task_index := dump.get("_ng_task_index")) is not None:
        rollout_index = dump.get("_ng_rollout_index", 0)
        return f"task_{task_index}_rollout_{rollout_index}"
    if (task_id := dump.get("id")) is not None:
        return f"task_{task_id}"
    return "unknown_task"


class DataAnalysisNotebookResourcesServer(SimpleResourcesServer):
    config: DataAnalysisNotebookResourcesServerConfig

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        # Semaphore for notebook execution
        self._semaphore: Semaphore = Semaphore(value=self.config.max_concurrent_executions)
        # Semaphore for judge endpoint
        self._judge_semaphore = asyncio.Semaphore(value=self.config.judge_endpoint_max_concurrency)

        tpl_path = Path(self.config.judge_prompt_template_fpath)
        if not tpl_path.is_absolute():
            tpl_path = Path(__file__).resolve().parent / tpl_path
        with open(tpl_path, encoding="utf-8") as f:
            self._judge_prompt_template = f.read().strip()

        executed_dir = Path(self.config.executed_notebooks_directory)
        if not executed_dir.is_absolute():
            executed_dir = Path(__file__).resolve().parent / executed_dir
        executed_dir.mkdir(parents=True, exist_ok=True)
        self._executed_notebooks_dir = executed_dir

    def setup_webserver(self):
        if self.config.judge_probe_on_startup:
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
                    logger.warning(
                        "Judge backend not yet reachable through '%s', retrying in 10s...", judge_name
                    )
                    sleep(10)
            logger.info("Judge model server '%s' is reachable.", judge_name)
        return super().setup_webserver()

    async def _run_llm_judge(
        self,
        *,
        task_text: str,
        reference_section: str,
        predicted_code: str,
        predicted_execution: dict[str, Any],
    ) -> tuple[Optional[bool], NotebookJudgeEvaluation]:
        # Set up judge prompt from template
        max_c = self.config.judge_max_output_chars
        for key, value in (
            ("{task_text}", task_text),
            ("{reference_section}", reference_section),
            ("{predicted_code}", truncate_for_judge(predicted_code, max_c)),
            (
                "{predicted_execution_json}",
                json.dumps(redact_execution_for_judge(predicted_execution, max_c), indent=2, default=str),
            ),
        ):
            user_prompt = self._judge_prompt_template.replace(key, value)

        responses_create_params = self.config.judge_responses_create_params.model_copy(deep=True)
        responses_create_params.input = [NeMoGymEasyInputMessage(role="user", content=user_prompt)]

        # Run judge model
        async with self._judge_semaphore:
            try:
                http_response = await self.server_client.post(
                    server_name=self.config.judge_model_server.name,
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

        # Parse judge response
        evaluation = NotebookJudgeEvaluation(
            responses_create_params=responses_create_params,
            response=judge_response,
        )
        text = last_assistant_output_text(judge_response)
        if text is None:
            evaluation.verdict_label = "judge_parsing_error"
            return None, evaluation

        passed, verdict_label, reason = parse_notebook_judge_verdict(text)
        evaluation.verdict_label = verdict_label
        evaluation.reason = reason

        # If passed is None, this means the judge output was not parseable.
        if passed is None:
            logger.warning("Judge output was not parseable: %s", reason)
            return None, evaluation
        return passed, evaluation

    async def verify(self, body: DataAnalysisNotebookVerifyRequest) -> DataAnalysisNotebookVerifyResponse:
        # Metadata includes ref notebook and data path locations and mounts
        meta = VerifierMetadata.model_validate(body.verifier_metadata)

        ref_nb = nbformat.from_dict(meta.reference_notebook)
        ref_code_cells = [c.source for c in ref_nb.cells if c.cell_type == "code" and c.source]
        pred_nb = extract_predicted_notebook(body.response.output_text)
        pred_code_cells = [c.source for c in pred_nb.cells if c.cell_type == "code"]
        n_ref, n_pred = len(ref_code_cells), len(pred_code_cells)

        # Still perform evaluation if reference code cells are empty
        if not ref_code_cells:
            logger.warning("reference notebook has no code cells, performing verification with no reference")

        if not pred_code_cells:
            return _verify_response(
                body,
                reward=0.0,
                comparison_detail="model output has no python code cells",
                num_reference_code_cells=n_ref,
            )

        # Execution setup (if not skipped)!
        skip_exec = (
            meta.skip_notebook_execution
            if meta.skip_notebook_execution is not None
            else self.config.skip_notebook_execution
        )
        timeout = self.config.execute_timeout_secs
        margin = self.config.wall_clock_margin_secs
        image_mode = self.config.image_compare_mode
        max_c = self.config.judge_max_output_chars

        ref_nb_executed: Optional[NotebookNode] = None
        pred_nb_executed: Optional[NotebookNode] = None
        ref_err: Optional[str] = None
        pred_err: Optional[str] = None

        if not skip_exec:
            execute = partial[tuple[Any | None, str | None]](
                run_staging_and_execute,
                data_paths=meta.data_paths,
                timeout=timeout,
                wall_margin=margin,
            )
            loop = get_running_loop()
            stem = _executed_notebook_stem(body)
            ref_output_fpath = self._executed_notebooks_dir / f"{stem}_reference.ipynb"
            pred_output_fpath = self._executed_notebooks_dir / f"{stem}_predicted.ipynb"
            async with self._semaphore:
                ref_nb_executed, ref_err = await loop.run_in_executor(
                    None,
                    partial(
                        execute,
                        ref_nb,
                        write_nb=True,
                        notebook_output_fpath=ref_output_fpath,
                    ),
                )
                pred_nb_executed, pred_err = await loop.run_in_executor(
                    None,
                    partial(
                        execute,
                        pred_nb,
                        write_nb=True,
                        notebook_output_fpath=pred_output_fpath,
                    ),
                )

            if pred_err:
                return _verify_response(
                    body,
                    reward=0.0,
                    comparison_detail=f"predicted_execution_failed: {truncate_exec_error(pred_err)}",
                    predicted_execution_error=truncate_exec_error(pred_err),
                    num_reference_code_cells=n_ref,
                    num_predicted_code_cells=n_pred,
                    reference_execution_error=truncate_exec_error(ref_err),
                )

        # Build notebook views and GT reference section for judge
        ref_view = build_notebook_execution_view(ref_nb, ref_nb_executed, image_mode)
        pred_view = build_notebook_execution_view(pred_nb, pred_nb_executed, image_mode)

        if skip_exec:
            reference_section = (
                "(Ground-truth reference only; notebooks were not executed. "
                "No runtime outputs. The predicted execution block below is empty. "
                "Decide PASS/FAIL from the task and cells alone.)\n\n"
                + json.dumps(redact_execution_for_judge(ref_view, max_c), indent=2, default=str)
            )
        elif ref_err:
            reference_section = (
                "(Reference notebook did not execute successfully; no reference output to compare.)\n"
                f"Error:\n{truncate_exec_error(ref_err)}"
            )
        elif not ref_code_cells:
            reference_section = (
                "(Reference notebook has no code cells, performing verification with no reference.)"            )
        else:
            reference_section = json.dumps(redact_execution_for_judge(ref_view, max_c), indent=2, default=str)

        # Run LLM judge
        ok, evaluation = await self._run_llm_judge(
            task_text=inputs_to_task_text(body.responses_create_params),
            reference_section=reference_section,
            predicted_code="\n\n---\n\n".join(c.source for c in pred_nb.cells),
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
            reference_execution_error=truncate_exec_error(ref_err),
            num_reference_code_cells=n_ref,
            num_predicted_code_cells=n_pred,
            judge_evaluations=[evaluation],
        )


if __name__ == "__main__":
    DataAnalysisNotebookResourcesServer.run_webserver()
