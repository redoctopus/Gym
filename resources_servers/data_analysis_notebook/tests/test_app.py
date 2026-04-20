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
from typing import Generator
from unittest.mock import MagicMock

import nbformat
import pytest
from app import (
    DataAnalysisNotebookResourcesServer,
    DataAnalysisNotebookResourcesServerConfig,
    DataAnalysisNotebookVerifyRequest,
    DataAnalysisNotebookVerifyResponse,
    extract_predicted_code_cells,
    extract_reference_code_sources,
    merged_output_signature,
    merged_signatures_match,
)
from fastapi.testclient import TestClient
from nbformat.v4 import new_output
from pydantic import ValidationError

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient


def _ref_nb(*sources: str) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [{"cell_type": "code", "metadata": {}, "outputs": [], "source": s} for s in sources],
    }


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


class TestExtract:
    def test_predicted_empty(self) -> None:
        assert extract_predicted_code_cells("") == []
        assert extract_predicted_code_cells("no fences here") == []

    def test_predicted_multiple(self) -> None:
        src = "```python\nprint(1)\n```\n\n```py\nprint(2)\n```"
        cells = extract_predicted_code_cells(src)
        assert cells == ["print(1)", "print(2)"]

    def test_reference_skips_markdown(self) -> None:
        nb = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {},
            "cells": [
                {"cell_type": "markdown", "metadata": {}, "source": "# Title"},
                {"cell_type": "code", "metadata": {}, "outputs": [], "source": "print(99)"},
            ],
        }
        assert extract_reference_code_sources(nb) == ["print(99)"]


class TestMergedSignature:
    def test_two_cells_same_stdout_as_one_cell(self) -> None:
        nb_split = nbformat.v4.new_notebook()
        c1 = nbformat.v4.new_code_cell("x")
        c1.outputs = [new_output("stream", name="stdout", text="1\n")]
        c2 = nbformat.v4.new_code_cell("y")
        c2.outputs = [new_output("stream", name="stdout", text="2\n")]
        nb_split.cells = [c1, c2]

        nb_one = nbformat.v4.new_notebook()
        c0 = nbformat.v4.new_code_cell("z")
        c0.outputs = [new_output("stream", name="stdout", text="1\n2\n")]
        nb_one.cells = [c0]

        a = merged_output_signature(nb_split, "exact")
        b = merged_output_signature(nb_one, "exact")
        assert merged_signatures_match(a, b)

    def test_stderr_separate_from_stdout(self) -> None:
        nb = nbformat.v4.new_notebook()
        c = nbformat.v4.new_code_cell("x")
        c.outputs = [
            new_output("stream", name="stdout", text="a\n"),
            new_output("stream", name="stderr", text="b\n"),
        ]
        nb.cells = [c]
        sig = merged_output_signature(nb, "exact")
        assert sig["text_merged"] == "a"
        assert sig["stderr_merged"] == "b"

    def test_stdout_and_plain_interleaved_in_text_merged(self) -> None:
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
        sig = merged_output_signature(nb, "exact")
        assert sig["text_merged"] == "printed\nrepr_value"


@pytest.fixture(scope="module")
def dan_verify_client() -> Generator[TestClient, None, None]:
    server = DataAnalysisNotebookResourcesServer(
        config=DataAnalysisNotebookResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            max_concurrent_executions=1,
            execute_timeout_secs=90,
            wall_clock_margin_secs=60,
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    app = server.setup_webserver()
    with TestClient(app) as client:
        yield client


class TestVerifyIntegration:
    async def test_verify_pass_matching_code(self, dan_verify_client: TestClient) -> None:
        body = DataAnalysisNotebookVerifyRequest(
            responses_create_params={
                "input": [{"role": "user", "content": "Print hello."}],
                "parallel_tool_calls": False,
            },
            response=_assistant_response('```python\nprint("hello")\n```'),
            verifier_metadata={"reference_notebook": _ref_nb('print("hello")')},
        )
        r = dan_verify_client.post("/verify", json=body.model_dump())
        assert r.status_code == 200
        res = DataAnalysisNotebookVerifyResponse.model_validate(r.json())
        assert res.reward == 1.0
        assert res.match is True
        assert res.reference_merged_output is not None
        assert "hello" in res.reference_merged_output["text_merged"]

    async def test_verify_fail_different_output(self, dan_verify_client: TestClient) -> None:
        body = DataAnalysisNotebookVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "x"}], "parallel_tool_calls": False},
            response=_assistant_response('```python\nprint("wrong")\n```'),
            verifier_metadata={"reference_notebook": _ref_nb('print("right")')},
        )
        r = dan_verify_client.post("/verify", json=body.model_dump())
        res = DataAnalysisNotebookVerifyResponse.model_validate(r.json())
        assert res.reward == 0.0
        assert res.match is False
        assert res.reference_merged_output is not None
        assert "right" in res.reference_merged_output["text_merged"]

    async def test_verify_fail_no_python_fence(self, dan_verify_client: TestClient) -> None:
        body = DataAnalysisNotebookVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "x"}], "parallel_tool_calls": False},
            response=_assistant_response("just prose, no code"),
            verifier_metadata={"reference_notebook": _ref_nb("print(1)")},
        )
        r = dan_verify_client.post("/verify", json=body.model_dump())
        res = DataAnalysisNotebookVerifyResponse.model_validate(r.json())
        assert res.reward == 0.0
        assert "no python code fences" in (res.comparison_detail or "")

    async def test_verify_predicted_syntax_error(self, dan_verify_client: TestClient) -> None:
        body = DataAnalysisNotebookVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "x"}], "parallel_tool_calls": False},
            response=_assistant_response("```python\n*** not valid\n```"),
            verifier_metadata={"reference_notebook": _ref_nb("print(1)")},
        )
        r = dan_verify_client.post("/verify", json=body.model_dump())
        res = DataAnalysisNotebookVerifyResponse.model_validate(r.json())
        assert res.reward == 0.0
        assert res.predicted_execution_error
        assert res.reference_merged_output is not None
        assert res.reference_merged_output["text_merged"] == "1"

    async def test_verify_with_data_files(self, dan_verify_client: TestClient) -> None:
        ref = "import pathlib\nprint(pathlib.Path('toy.csv').read_text().strip())"
        pred = "```python\nimport pathlib\nprint(pathlib.Path('toy.csv').read_text().strip())\n```"
        body = DataAnalysisNotebookVerifyRequest(
            responses_create_params={
                "input": [{"role": "user", "content": "Read toy.csv"}],
                "parallel_tool_calls": False,
            },
            response=_assistant_response(pred),
            verifier_metadata={
                "reference_notebook": _ref_nb(ref),
                "data_files": {"toy.csv": "x,y\n1,2"},
            },
        )
        r = dan_verify_client.post("/verify", json=body.model_dump())
        res = DataAnalysisNotebookVerifyResponse.model_validate(r.json())
        assert res.reward == 1.0

    async def test_verify_with_data_paths(self, dan_verify_client: TestClient) -> None:
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
            r = dan_verify_client.post("/verify", json=body.model_dump())
            res = DataAnalysisNotebookVerifyResponse.model_validate(r.json())
            assert res.reward == 1.0
        finally:
            Path(src_path).unlink(missing_ok=True)

    async def test_strip_thinking_before_fences(self, dan_verify_client: TestClient) -> None:
        body = DataAnalysisNotebookVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "x"}], "parallel_tool_calls": False},
            response=_assistant_response(
                "<think>\ninternal\n</think>\n```python\nprint(7)\n```",
            ),
            verifier_metadata={"reference_notebook": _ref_nb("print(7)")},
        )
        r = dan_verify_client.post("/verify", json=body.model_dump())
        res = DataAnalysisNotebookVerifyResponse.model_validate(r.json())
        assert res.reward == 1.0


def test_verify_request_requires_response() -> None:
    with pytest.raises(ValidationError):
        DataAnalysisNotebookVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "x"}], "parallel_tool_calls": False},
            verifier_metadata={"reference_notebook": _ref_nb("print(1)")},
        )
