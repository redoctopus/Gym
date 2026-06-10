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
    extract_predicted_notebook,
)
from fastapi.testclient import TestClient
from judge import parse_notebook_judge_verdict, redact_execution_for_judge
from nbformat.v4 import new_output
from notebook_runtime import build_notebook_execution_view
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


def _code_notebook(*sources: str) -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_code_cell(source) for source in sources]
    return nb


def _predicted_code_sources(text: str) -> list[str]:
    nb = extract_predicted_notebook(text)
    return [cell.source for cell in nb.cells if cell.cell_type == "code"]


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


class TestExtractPredictedNotebook:
    def test_empty_text_returns_empty_notebook(self) -> None:
        nb = extract_predicted_notebook("")
        assert nb.cells == []
        nb = extract_predicted_notebook("no fences here")
        assert nb.cells == []

    def test_multiple_python_fences(self) -> None:
        src = "```python\nprint(1)\n```\n\n```py\nprint(2)\n```"
        assert _predicted_code_sources(src) == ["print(1)", "print(2)"]

    def test_markdown_and_code_in_order(self) -> None:
        src = (
            "```markdown\n# Title\n```\n\n"
            "```python\nprint(1)\n```\n\n"
            "```markdown\n## Section\n```\n\n"
            "```python\nprint(2)\n```"
        )
        nb = extract_predicted_notebook(src)
        assert [cell.cell_type for cell in nb.cells] == ["markdown", "code", "markdown", "code"]
        assert nb.cells[0].source == "# Title"
        assert nb.cells[1].source == "print(1)"
        assert nb.cells[3].source == "print(2)"

    def test_strip_thinking_before_fences(self) -> None:
        src = "<think>\ninternal\n</think>\n```python\nprint(7)\n```"
        assert _predicted_code_sources(src) == ["print(7)"]

    def test_unclosed_redacted_thinking_yields_no_cells(self) -> None:
        src = "<think>\nstill reasoning\n```python\nprint(1)\n```"
        assert extract_predicted_notebook(src).cells == []


class TestNotebookExecutionView:
    def test_two_code_cells_stay_separate(self) -> None:
        source_nb = _code_notebook("x", "y")
        executed_nb = nbformat.v4.new_notebook()
        c1 = nbformat.v4.new_code_cell("x")
        c1.outputs = [new_output("stream", name="stdout", text="1\n")]
        c2 = nbformat.v4.new_code_cell("y")
        c2.outputs = [new_output("stream", name="stdout", text="2\n")]
        executed_nb.cells = [c1, c2]

        view = build_notebook_execution_view(source_nb, executed_nb, "exact")
        assert len(view["cells"]) == 2
        assert view["cells"][0]["outputs"][0]["text"] == "1"
        assert view["cells"][1]["outputs"][0]["text"] == "2"

    def test_markdown_cells_preserved_in_view(self) -> None:
        source_nb = nbformat.v4.new_notebook()
        source_nb.cells = [nbformat.v4.new_markdown_cell("# Title"), nbformat.v4.new_code_cell("print(1)")]
        executed_nb = nbformat.v4.new_notebook()
        executed_code = nbformat.v4.new_code_cell("print(1)")
        executed_code.outputs = [new_output("stream", name="stdout", text="1\n")]
        executed_nb.cells = [nbformat.v4.new_markdown_cell("# Title"), executed_code]

        view = build_notebook_execution_view(source_nb, executed_nb, "exact")
        assert view["cells"][0]["cell_type"] == "markdown"
        assert view["cells"][1]["outputs"][0]["text"] == "1"

    def test_stderr_and_stdout_are_separate_output_records(self) -> None:
        source_nb = _code_notebook("x")
        executed_nb = nbformat.v4.new_notebook()
        c = nbformat.v4.new_code_cell("x")
        c.outputs = [
            new_output("stream", name="stdout", text="a\n"),
            new_output("stream", name="stderr", text="b\n"),
        ]
        executed_nb.cells = [c]
        view = build_notebook_execution_view(source_nb, executed_nb, "exact")
        outputs = view["cells"][0]["outputs"]
        assert outputs[0]["name"] == "stdout"
        assert outputs[0]["text"] == "a"
        assert outputs[1]["name"] == "stderr"
        assert outputs[1]["text"] == "b"

    def test_stdout_and_plain_stay_separate_output_records(self) -> None:
        source_nb = _code_notebook("x")
        executed_nb = nbformat.v4.new_notebook()
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
        executed_nb.cells = [c]
        view = build_notebook_execution_view(source_nb, executed_nb, "exact")
        outputs = view["cells"][0]["outputs"]
        assert outputs[0]["kind"] == "stream"
        assert outputs[1]["kind"] == "plain"
        assert outputs[1]["text"] == "repr_value"

    def test_skip_mode_empty_outputs(self) -> None:
        view = build_notebook_execution_view(_code_notebook("print(1)", "x = 2"), None, "exact")
        assert all(cell["outputs"] == [] for cell in view["cells"])


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

    async def test_verify_fail_no_python_fence(self, client_judge_pass: TestClient) -> None:
        body = DataAnalysisNotebookVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "x"}], "parallel_tool_calls": False},
            response=_assistant_response("just prose, no code"),
            verifier_metadata={"reference_notebook": _ref_nb("print(1)")},
        )
        r = client_judge_pass.post("/verify", json=body.model_dump())
        res = DataAnalysisNotebookVerifyResponse.model_validate(r.json())
        assert res.reward == 0.0
        assert "no python code cells" in res.comparison_detail

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
        assert res.reference_execution_error is None

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

    async def test_strip_thinking_before_fences_integration(self, client_judge_pass: TestClient) -> None:
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
        assert res.reference_execution_error is None
        assert res.predicted_execution_error is None

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
            nb: nbformat.NotebookNode,
            data_paths: Any,
            timeout: int,
            wall_margin: int,
            write_nb: bool = False,
            notebook_output_fpath: Any = None,
        ) -> tuple[Any, None]:
            nonlocal calls
            calls += 1
            executed = nbformat.v4.new_notebook()
            c = nbformat.v4.new_code_cell("x")
            c.outputs = [new_output("stream", name="stdout", text="ok\n")]
            executed.cells = [c]
            return executed, None

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
