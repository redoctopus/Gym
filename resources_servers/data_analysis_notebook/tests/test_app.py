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

import tempfile
from pathlib import Path
from typing import Any, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import nbformat
import pytest
from app import (
    DataAnalysisNotebookResourcesServer,
    DataAnalysisNotebookResourcesServerConfig,
    DataAnalysisNotebookVerifyRequest,
    DataAnalysisNotebookVerifyResponse,
    ReferenceCell,
    _first_line_excerpt_for_log,
    _previous_output_excerpt,
    _resolve_execution_log_stem,
    _task_question_excerpt_for_log,
    build_notebook_execution_view,
    build_predicted_execution_view,
    extract_predicted_code_cells,
    extract_reference_cells,
    extract_reference_code_sources,
)
from fastapi.testclient import TestClient
from judge import parse_notebook_judge_verdict, redact_execution_for_judge
from nbformat.v4 import new_output
from pydantic import ValidationError

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient


_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompt_templates" / "notebook_task_judge.txt"


def _default_dan_config(**kwargs: Any) -> DataAnalysisNotebookResourcesServerConfig:
    base: dict[str, Any] = {
        "host": "0.0.0.0",
        "port": 8080,
        "entrypoint": "",
        "name": "",
        "max_concurrent_executions": 1,
        "execute_timeout_secs": 90,
        "wall_clock_margin_secs": 60,
        "judge_model_server": ModelServerRef(type="responses_api_models", name="judge"),
        "judge_responses_create_params": NeMoGymResponseCreateParamsNonStreaming(input=[]),
        "judge_prompt_template_fpath": str(_PROMPT_PATH),
        "judge_probe_on_startup": False,
    }
    base.update(kwargs)
    return DataAnalysisNotebookResourcesServerConfig(**base)


def _stub_server_client() -> MagicMock:
    mock = MagicMock(spec=ServerClient)
    mock.poll_for_status = MagicMock(return_value="success")
    mock._build_server_base_url = MagicMock(return_value="http://127.0.0.1:9")
    mock.global_config_dict = MagicMock()
    return mock


def _mock_judge_response(server_mock: MagicMock, judge_text: str) -> None:
    judge_response = NeMoGymResponse(
        id="judge_resp",
        created_at=0.0,
        model="judge",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id="msg_judge",
                content=[NeMoGymResponseOutputText(annotations=[], text=judge_text, type="output_text")],
                role="assistant",
                status="completed",
                type="message",
            )
        ],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )
    post_mock = MagicMock()
    post_mock.read = AsyncMock(return_value=judge_response.model_dump_json().encode())
    server_mock.post = AsyncMock(return_value=post_mock)


def _make_test_client(judge_output: str = "VERDICT: PASS\nREASON: ok") -> TestClient:
    mock = _stub_server_client()
    _mock_judge_response(mock, judge_output)
    server = DataAnalysisNotebookResourcesServer(config=_default_dan_config(), server_client=mock)
    return TestClient(server.setup_webserver())


def _ref_nb(*sources: str) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [{"cell_type": "code", "metadata": {}, "outputs": [], "source": s} for s in sources],
    }


def _code_output_text(view: dict[str, Any]) -> str:
    parts: list[str] = []
    for cell in view.get("cells") or []:
        if cell.get("cell_type") != "code":
            continue
        for rec in cell.get("outputs") or []:
            if rec.get("kind") in ("stream", "plain"):
                parts.append(str(rec.get("text") or ""))
    return "\n".join(parts)


def _assistant_response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="r1",
        created_at=0.0,
        model="dummy",
        object="response",
        output=[
            {
                "id": "m1",
                "content": [{"annotations": [], "text": text, "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )


class TestTaskQuestionExcerpt:
    def test_uses_last_user_message(self) -> None:
        p = NeMoGymResponseCreateParamsNonStreaming(
            input=[
                {"type": "message", "role": "user", "content": "early line"},
                {"type": "message", "role": "user", "content": "what is 2+2?"},
            ],
            parallel_tool_calls=False,
        )
        assert _task_question_excerpt_for_log(p) == "what is 2+2?"

    def test_str_input_first_line_only(self) -> None:
        p = NeMoGymResponseCreateParamsNonStreaming(
            input="line a\nline b",
            parallel_tool_calls=False,
        )
        assert _task_question_excerpt_for_log(p) == "line a"

    def test_first_line_truncation(self) -> None:
        long = "x" * 600
        assert _first_line_excerpt_for_log(long).endswith("...")
        assert len(_first_line_excerpt_for_log(long)) == 500


class TestExtract:
    def test_predicted_empty(self) -> None:
        assert extract_predicted_code_cells("") == []
        assert extract_predicted_code_cells("no fences here") == []

    def test_predicted_multiple(self) -> None:
        src = "```python\nprint(1)\n```\n\n```py\nprint(2)\n```"
        cells = extract_predicted_code_cells(src)
        assert cells == ["print(1)", "print(2)"]

    def test_reference_cells_include_markdown(self) -> None:
        nb = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {},
            "cells": [
                {"cell_type": "markdown", "metadata": {}, "source": "# Title"},
                {"cell_type": "code", "metadata": {}, "outputs": [], "source": "print(99)"},
            ],
        }
        cells = extract_reference_cells(nb)
        assert len(cells) == 2
        assert cells[0].cell_type == "markdown"
        assert cells[1].cell_type == "code"
        assert extract_reference_code_sources(nb) == ["print(99)"]


class TestNotebookExecutionView:
    def test_two_code_cells_stay_separate(self) -> None:
        nb_split = nbformat.v4.new_notebook()
        c1 = nbformat.v4.new_code_cell("x")
        c1.outputs = [new_output("stream", name="stdout", text="1\n")]
        c2 = nbformat.v4.new_code_cell("y")
        c2.outputs = [new_output("stream", name="stdout", text="2\n")]
        nb_split.cells = [c1, c2]

        cells = [ReferenceCell("code", "x"), ReferenceCell("code", "y")]
        view = build_notebook_execution_view(cells, nb_split, "exact")
        assert len(view["cells"]) == 2
        assert view["cells"][0]["outputs"][0]["text"] == "1"
        assert view["cells"][1]["outputs"][0]["text"] == "2"

    def test_markdown_cells_preserved_in_view(self) -> None:
        cells = [
            ReferenceCell("markdown", "# Title"),
            ReferenceCell("code", "print(1)"),
        ]
        nb = nbformat.v4.new_notebook()
        c = nbformat.v4.new_code_cell("print(1)")
        c.outputs = [new_output("stream", name="stdout", text="1\n")]
        nb.cells = [c]
        view = build_notebook_execution_view(cells, nb, "exact")
        assert view["cells"][0]["cell_type"] == "markdown"
        assert view["cells"][1]["outputs"][0]["text"] == "1"

    def test_stderr_and_stdout_are_separate_output_records(self) -> None:
        nb = nbformat.v4.new_notebook()
        c = nbformat.v4.new_code_cell("x")
        c.outputs = [
            new_output("stream", name="stdout", text="a\n"),
            new_output("stream", name="stderr", text="b\n"),
        ]
        nb.cells = [c]
        view = build_predicted_execution_view(["x"], nb, "exact")
        outputs = view["cells"][0]["outputs"]
        assert outputs[0]["name"] == "stdout"
        assert outputs[0]["text"] == "a"
        assert outputs[1]["name"] == "stderr"
        assert outputs[1]["text"] == "b"

    def test_stdout_and_plain_stay_separate_output_records(self) -> None:
        nb = nbformat.v4.new_notebook()
        c = nbformat.v4.new_code_cell("x")
        c.outputs = [
            new_output("stream", name="stdout", text="printed\n"),
            {
                "output_type": "execute_result",
                "data": {"text/plain": "repr_value"},
                "metadata": {},
                "execution_count": 1,
            },
        ]
        nb.cells = [c]
        view = build_predicted_execution_view(["x"], nb, "exact")
        outputs = view["cells"][0]["outputs"]
        assert outputs[0]["kind"] == "stream"
        assert outputs[1]["kind"] == "plain"
        assert outputs[1]["text"] == "repr_value"

    def test_skip_mode_empty_outputs(self) -> None:
        view = build_predicted_execution_view(["print(1)", "x = 2"], None, "exact")
        assert all(cell["outputs"] == [] for cell in view["cells"])

    def test_previous_output_excerpt_trailing_lines(self) -> None:
        nb = nbformat.v4.new_notebook()
        c0 = nbformat.v4.new_code_cell("x")
        c0.outputs = [new_output("stream", name="stdout", text="line0\n" * 20)]
        c1 = nbformat.v4.new_code_cell("y")
        c1.outputs = [new_output("stream", name="stdout", text="last\n")]
        nb.cells = [c0, c1]
        out = _previous_output_excerpt(nb, failing_code_cell_index=1, max_lines=3)
        assert "line0" in out
        assert out.count("\n") <= 4  # at most 3 data lines + leading "..." line

    def test_previous_output_excerpt_no_prior_cells(self) -> None:
        nb = nbformat.v4.new_notebook()
        nb.cells = [nbformat.v4.new_code_cell("x")]
        assert "no output" in _previous_output_excerpt(nb, 0, 5).lower()


@pytest.fixture
def client_judge_pass() -> Generator[TestClient, None, None]:
    with _make_test_client("VERDICT: PASS\nREASON: ok") as client:
        yield client


@pytest.fixture
def client_judge_fail() -> Generator[TestClient, None, None]:
    with _make_test_client("VERDICT: FAIL\nREASON: outputs disagree") as client:
        yield client


class TestParseJudgeVerdict:
    def test_pass(self) -> None:
        ok, label, reason = parse_notebook_judge_verdict("Analysis.\nVERDICT: PASS\nREASON: looks good")
        assert ok is True and label == "pass" and reason == "looks good"

    def test_fail(self) -> None:
        ok, label, reason = parse_notebook_judge_verdict("VERDICT: FAIL\nREASON: missing plot")
        assert ok is False and label == "fail"

    def test_ambiguous(self) -> None:
        ok, label, _ = parse_notebook_judge_verdict("VERDICT: PASS\nVERDICT: FAIL")
        assert ok is None and label == "judge_parsing_error"

    def test_empty(self) -> None:
        ok, label, _ = parse_notebook_judge_verdict("")
        assert ok is None and label == "judge_parsing_error"


class TestRedactExecutionForJudge:
    def test_pngs_become_summary(self) -> None:
        view = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": "x",
                    "outputs": [
                        {"kind": "plain", "text": "a"},
                        {"kind": "png", "data": "aaa"},
                        {"kind": "png", "data": "bbb"},
                    ],
                }
            ]
        }
        out = redact_execution_for_judge(view, 1000)
        assert "2 PNG figure" in out["png_summary"]
        outputs = out["cells"][0]["outputs"]
        assert all(rec.get("kind") != "png" or rec.get("omitted") for rec in outputs)


class TestExecutionLogStem:
    def test_explicit_stem_sanitized(self) -> None:
        params = NeMoGymResponseCreateParamsNonStreaming.model_validate(
            {"input": [{"type": "message", "role": "user", "content": "ignored"}], "parallel_tool_calls": False}
        )
        stem = _resolve_execution_log_stem("  My Run #1  ", params, None, {})
        assert stem == "My_Run_1"

    def test_derived_from_task_and_data_path(self) -> None:
        params = NeMoGymResponseCreateParamsNonStreaming.model_validate(
            {
                "input": [{"type": "message", "role": "user", "content": "Summarize the dataset."}],
                "parallel_tool_calls": False,
            }
        )
        stem = _resolve_execution_log_stem(
            None,
            params,
            [{"source": "/mnt/d.csv", "path": "custom/subset.csv"}],
            {},
        )
        assert stem.startswith("Summarize_the_dataset")
        assert "subset.csv" in stem

    def test_invalid_explicit_falls_through_to_task(self) -> None:
        params = NeMoGymResponseCreateParamsNonStreaming.model_validate(
            {
                "input": [{"type": "message", "role": "user", "content": "Valid line for stem."}],
                "parallel_tool_calls": False,
            }
        )
        stem = _resolve_execution_log_stem("../evil", params, None, {})
        assert "Valid_line_for_stem" in stem

    def test_notebook_title_when_no_user_text(self) -> None:
        params = NeMoGymResponseCreateParamsNonStreaming.model_validate(
            {"input": [{"type": "message", "role": "user", "content": "  "}], "parallel_tool_calls": False}
        )
        nb = {
            "metadata": {"title": "Kaggle Starter Notebook"},
            "nbformat": 4,
            "nbformat_minor": 5,
            "cells": [],
        }
        stem = _resolve_execution_log_stem(None, params, None, nb)
        assert "Kaggle_Starter_Notebook" in stem


class TestVerifyIntegration:
    async def test_verify_pass_matching_code(self, client_judge_pass: TestClient) -> None:
        body = DataAnalysisNotebookVerifyRequest(
            responses_create_params={
                "input": [{"role": "user", "content": "Print hello."}],
                "parallel_tool_calls": False,
            },
            response=_assistant_response('```python\nprint("hello")\n```'),
            verifier_metadata={"reference_notebook": _ref_nb('print("hello")')},
        )
        r = client_judge_pass.post("/verify", json=body.model_dump())
        assert r.status_code == 200
        res = DataAnalysisNotebookVerifyResponse.model_validate(r.json())
        assert res.reward == 1.0
        assert res.match is True
        assert res.reference_execution_output is not None
        assert "hello" in _code_output_text(res.reference_execution_output)
        assert res.judge_evaluations is not None
        assert res.judge_evaluations[0].verdict_label == "pass"

    async def test_verify_semantic_pass_when_outputs_differ(self, client_judge_pass: TestClient) -> None:
        """Judge PASS yields reward 1.0 even when merged stdout differs from reference."""
        body = DataAnalysisNotebookVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "x"}], "parallel_tool_calls": False},
            response=_assistant_response('```python\nprint("candidate")\n```'),
            verifier_metadata={"reference_notebook": _ref_nb('print("reference")')},
        )
        r = client_judge_pass.post("/verify", json=body.model_dump())
        res = DataAnalysisNotebookVerifyResponse.model_validate(r.json())
        assert res.reward == 1.0
        assert res.match is True
        assert "reference" in _code_output_text(res.reference_execution_output)

    async def test_verify_fail_different_output(self, client_judge_fail: TestClient) -> None:
        body = DataAnalysisNotebookVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "x"}], "parallel_tool_calls": False},
            response=_assistant_response('```python\nprint("wrong")\n```'),
            verifier_metadata={"reference_notebook": _ref_nb('print("right")')},
        )
        r = client_judge_fail.post("/verify", json=body.model_dump())
        res = DataAnalysisNotebookVerifyResponse.model_validate(r.json())
        assert res.reward == 0.0
        assert res.match is False
        assert res.reference_execution_output is not None
        assert "right" in _code_output_text(res.reference_execution_output)

    async def test_verify_fail_no_python_fence(self, client_judge_pass: TestClient) -> None:
        body = DataAnalysisNotebookVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "x"}], "parallel_tool_calls": False},
            response=_assistant_response("just prose, no code"),
            verifier_metadata={"reference_notebook": _ref_nb("print(1)")},
        )
        r = client_judge_pass.post("/verify", json=body.model_dump())
        res = DataAnalysisNotebookVerifyResponse.model_validate(r.json())
        assert res.reward == 0.0
        assert "no python code fences" in res.comparison_detail

    async def test_verify_predicted_syntax_error(self, client_judge_pass: TestClient) -> None:
        body = DataAnalysisNotebookVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "x"}], "parallel_tool_calls": False},
            response=_assistant_response("```python\n*** not valid\n```"),
            verifier_metadata={"reference_notebook": _ref_nb("print(1)")},
        )
        r = client_judge_pass.post("/verify", json=body.model_dump())
        res = DataAnalysisNotebookVerifyResponse.model_validate(r.json())
        assert res.reward == 0.0
        assert res.predicted_execution_error
        assert res.reference_execution_output is not None
        assert _code_output_text(res.reference_execution_output) == "1"

    async def test_verify_with_data_paths(self, client_judge_pass: TestClient) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write("k,v\n9,9\n")
            src_path = f.name
        try:
            ref = "import pathlib\nprint(pathlib.Path('staged.csv').read_text().strip())"
            pred = "```python\nimport pathlib\nprint(pathlib.Path('staged.csv').read_text().strip())\n```"
            body = DataAnalysisNotebookVerifyRequest(
                responses_create_params={
                    "input": [{"role": "user", "content": "Read staged.csv"}],
                    "parallel_tool_calls": False,
                },
                response=_assistant_response(pred),
                verifier_metadata={
                    "reference_notebook": _ref_nb(ref),
                    "data_paths": [{"source": src_path, "path": "staged.csv"}],
                },
            )
            r = client_judge_pass.post("/verify", json=body.model_dump())
            res = DataAnalysisNotebookVerifyResponse.model_validate(r.json())
            assert res.reward == 1.0
        finally:
            Path(src_path).unlink(missing_ok=True)

    async def test_strip_thinking_before_fences(self, client_judge_pass: TestClient) -> None:
        body = DataAnalysisNotebookVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "x"}], "parallel_tool_calls": False},
            response=_assistant_response(
                "<think>\ninternal\n</think>\n```python\nprint(7)\n```",
            ),
            verifier_metadata={"reference_notebook": _ref_nb("print(7)")},
        )
        r = client_judge_pass.post("/verify", json=body.model_dump())
        res = DataAnalysisNotebookVerifyResponse.model_validate(r.json())
        assert res.reward == 1.0

    async def test_verify_writes_per_execution_log_files(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "execution_logs"
        log_dir.mkdir()
        mock = _stub_server_client()
        _mock_judge_response(mock, "VERDICT: PASS\nREASON: ok")
        server = DataAnalysisNotebookResourcesServer(
            config=_default_dan_config(execution_log_directory=str(log_dir)),
            server_client=mock,
        )
        app = server.setup_webserver()
        with TestClient(app) as client:
            body = DataAnalysisNotebookVerifyRequest(
                responses_create_params={
                    "input": [{"role": "user", "content": "Print hello."}],
                    "parallel_tool_calls": False,
                },
                response=_assistant_response('```python\nprint("smoke_log_marker")\n```'),
                verifier_metadata={
                    "reference_notebook": _ref_nb('print("smoke_log_marker")'),
                    "execution_log_stem": "task_smoke",
                },
            )
            r = client.post("/verify", json=body.model_dump())
            res = DataAnalysisNotebookVerifyResponse.model_validate(r.json())
        assert r.status_code == 200
        assert res.reward == 1.0
        ref_logs = sorted(log_dir.glob("*_task_smoke_reference.log"))
        pred_logs = sorted(log_dir.glob("*_task_smoke_predicted.log"))
        assert len(ref_logs) == 1 and len(pred_logs) == 1
        ref_log, pred_log = ref_logs[0], pred_logs[0]
        assert ref_log.is_file() and pred_log.is_file()
        out_ref = ref_log.read_text(encoding="utf-8")
        out_pred = pred_log.read_text(encoding="utf-8")
        assert "smoke_log_marker" in out_ref and "smoke_log_marker" in out_pred
        assert "Outputs (this cell only" in out_ref
        assert "# Task question" in out_ref
        assert "Print hello" in out_ref
        assert "# Task question" in out_pred

    async def test_verify_skip_execution_does_not_call_runner(self) -> None:
        def _fail_run(*_a: Any, **_k: Any) -> None:
            raise AssertionError("run_staging_and_execute must not run when skip_notebook_execution")

        with patch("app.run_staging_and_execute", side_effect=_fail_run):
            with _make_test_client("VERDICT: PASS\nREASON: ok") as client:
                body = DataAnalysisNotebookVerifyRequest(
                    responses_create_params={
                        "input": [{"role": "user", "content": "Do the task."}],
                        "parallel_tool_calls": False,
                    },
                    response=_assistant_response("```python\nprint(1)\n```"),
                    verifier_metadata={
                        "reference_notebook": _ref_nb("print(2)"),
                        "skip_notebook_execution": True,
                    },
                )
                r = client.post("/verify", json=body.model_dump())
        assert r.status_code == 200
        res = DataAnalysisNotebookVerifyResponse.model_validate(r.json())
        assert res.reward == 1.0
        assert res.match is True
        assert res.reference_execution_output is None
        assert res.reference_execution_error is None
        assert res.predicted_execution_error is None

    async def test_verify_skip_execution_writes_log_files_when_log_dir_set(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "skip_logs"
        log_dir.mkdir()

        def _fail_run(*_a: Any, **_k: Any) -> None:
            raise AssertionError("run_staging_and_execute must not run when skip_notebook_execution")

        with patch("app.run_staging_and_execute", side_effect=_fail_run):
            mock = _stub_server_client()
            _mock_judge_response(mock, "VERDICT: PASS\nREASON: ok")
            server = DataAnalysisNotebookResourcesServer(
                config=_default_dan_config(
                    skip_notebook_execution=True,
                    execution_log_directory=str(log_dir),
                ),
                server_client=mock,
            )
            app = server.setup_webserver()
            with TestClient(app) as client:
                body = DataAnalysisNotebookVerifyRequest(
                    responses_create_params={
                        "input": [{"role": "user", "content": "Skip log stem task."}],
                        "parallel_tool_calls": False,
                    },
                    response=_assistant_response("```python\nprint('pred_skip_marker')\n```"),
                    verifier_metadata={
                        "reference_notebook": _ref_nb("print('ref_skip_marker')"),
                        "execution_log_stem": "skip_log_stem",
                    },
                )
                r = client.post("/verify", json=body.model_dump())
        assert r.status_code == 200
        res = DataAnalysisNotebookVerifyResponse.model_validate(r.json())
        assert res.reward == 1.0
        ref_logs = sorted(log_dir.glob("*_skip_log_stem_reference.log"))
        pred_logs = sorted(log_dir.glob("*_skip_log_stem_predicted.log"))
        assert len(ref_logs) == 1 and len(pred_logs) == 1
        out_ref = ref_logs[0].read_text(encoding="utf-8")
        out_pred = pred_logs[0].read_text(encoding="utf-8")
        assert "skip_notebook_execution" in out_ref and "skip_notebook_execution" in out_pred
        assert "ref_skip_marker" in out_ref
        assert "pred_skip_marker" in out_pred
        assert "not executed (verify skipped notebook execution)" in out_ref
        assert "# Task question" in out_ref
        assert "Skip log stem task" in out_ref

    async def test_verify_skip_invalid_python_still_calls_judge(self) -> None:
        """Skip mode never executes; invalid syntax is not surfaced as predicted_execution_error."""

        def _fail_run(*_a: Any, **_k: Any) -> None:
            raise AssertionError("no execution")

        with patch("app.run_staging_and_execute", side_effect=_fail_run):
            with _make_test_client("VERDICT: PASS\nREASON: ok") as client:
                body = DataAnalysisNotebookVerifyRequest(
                    responses_create_params={
                        "input": [{"role": "user", "content": "x"}],
                        "parallel_tool_calls": False,
                    },
                    response=_assistant_response("```python\n*** not valid\n```"),
                    verifier_metadata={
                        "reference_notebook": _ref_nb("print(1)"),
                        "skip_notebook_execution": True,
                    },
                )
                r = client.post("/verify", json=body.model_dump())
        res = DataAnalysisNotebookVerifyResponse.model_validate(r.json())
        assert res.reward == 1.0
        assert res.predicted_execution_error is None

    def test_skip_metadata_false_overrides_config_true(self) -> None:
        calls = 0

        def fake_run(
            sources: list[str],
            data_paths: Any,
            timeout: int,
            margin: int,
            log_path: Any = None,
            log_task_excerpt: Any = None,
            log_cell_specs: Any = None,
        ) -> tuple[Any, None]:
            nonlocal calls
            calls += 1
            nb = nbformat.v4.new_notebook()
            c = nbformat.v4.new_code_cell("x")
            c.outputs = [new_output("stream", name="stdout", text="ok\n")]
            nb.cells = [c]
            return nb, None

        mock = _stub_server_client()
        _mock_judge_response(mock, "VERDICT: PASS\nREASON: ok")
        server = DataAnalysisNotebookResourcesServer(
            config=_default_dan_config(skip_notebook_execution=True),
            server_client=mock,
        )
        with patch("app.run_staging_and_execute", side_effect=fake_run):
            with TestClient(server.setup_webserver()) as client:
                body = DataAnalysisNotebookVerifyRequest(
                    responses_create_params={
                        "input": [{"role": "user", "content": "x"}],
                        "parallel_tool_calls": False,
                    },
                    response=_assistant_response("```python\nprint(1)\n```"),
                    verifier_metadata={
                        "reference_notebook": _ref_nb("print(1)"),
                        "skip_notebook_execution": False,
                    },
                )
                r = client.post("/verify", json=body.model_dump())
        assert r.status_code == 200
        assert calls == 2


class TestJudgeFailure:
    def test_judge_unreachable_returns_zero(self) -> None:
        mock = _stub_server_client()
        mock.post = AsyncMock(side_effect=ConnectionError("refused"))
        server = DataAnalysisNotebookResourcesServer(config=_default_dan_config(), server_client=mock)
        with TestClient(server.setup_webserver()) as client:
            body = DataAnalysisNotebookVerifyRequest(
                responses_create_params={"input": [{"role": "user", "content": "x"}], "parallel_tool_calls": False},
                response=_assistant_response("```python\nprint(1)\n```"),
                verifier_metadata={"reference_notebook": _ref_nb("print(1)")},
            )
            r = client.post("/verify", json=body.model_dump())
        res = DataAnalysisNotebookVerifyResponse.model_validate(r.json())
        assert res.reward == 0.0
        assert res.judge_evaluations is not None
        assert res.judge_evaluations[0].verdict_label == "judge_error"

    def test_judge_garbage_output_returns_parsing_error(self) -> None:
        mock = _stub_server_client()
        _mock_judge_response(mock, "I cannot decide.")
        server = DataAnalysisNotebookResourcesServer(config=_default_dan_config(), server_client=mock)
        with TestClient(server.setup_webserver()) as client:
            body = DataAnalysisNotebookVerifyRequest(
                responses_create_params={"input": [{"role": "user", "content": "x"}], "parallel_tool_calls": False},
                response=_assistant_response("```python\nprint(1)\n```"),
                verifier_metadata={"reference_notebook": _ref_nb("print(1)")},
            )
            r = client.post("/verify", json=body.model_dump())
        res = DataAnalysisNotebookVerifyResponse.model_validate(r.json())
        assert res.reward == 0.0
        assert res.judge_evaluations is not None
        assert res.judge_evaluations[0].verdict_label == "judge_parsing_error"


def test_verify_request_requires_response() -> None:
    with pytest.raises(ValidationError):
        DataAnalysisNotebookVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "x"}], "parallel_tool_calls": False},
            verifier_metadata={"reference_notebook": _ref_nb("print(1)")},
        )
