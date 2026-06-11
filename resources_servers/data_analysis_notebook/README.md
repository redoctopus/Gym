# Data analysis notebook (resources server)

Verifies single-turn model answers by running **reference** code cells (from a ground-truth notebook) and **predicted** cells (parsed from the model reply) in **separate** temporary working directories and fresh IPython kernels—unless **`skip_notebook_execution`** is enabled (server config or per-request `verifier_metadata`). In skip mode, nothing is executed: the **LLM judge** receives the task, reference cells (code and markdown), predicted code, and unexecuted notebook JSON, and decides whether the prediction fulfills the task from content alone.

By default, executed notebooks are serialized to **nbformat v4 JSON** (`json.loads(nbformat.writes(...))`) and passed to the judge together with the task text and predicted code. The judge decides whether the prediction **correctly and completely** satisfies the task, using reference execution output as a guide when it is available (semantic match; not byte-identical output).

If the reference notebook fails to execute, verification still runs for the predicted notebook only: the judge does not receive reference output (only the task and predicted code/output).

**Charts / images:** Image outputs (`image/png`, `image/jpeg`, `image/svg+xml`, etc.) are written to `executed_notebooks/` as `{stem}_reference|predicted_image_{n}.{ext}` and replaced in the judge JSON with `{mime}_path` pointers. This is done to cut down on judge input length. Text and numeric outputs remain inline.

Reference notebooks preserve **markdown** and **code** cells in order. Only code cells are executed; markdown cells appear in judge input.

## Security

`/verify` normally executes **untrusted model code** with the same OS privileges as the server. With `skip_notebook_execution` enabled (config or metadata), code is not run for that request; the judge still sees full source strings. Run in an isolated environment when executing, keep `execute_timeout_secs` and `max_concurrent_executions` low, and do not expose this endpoint to untrusted callers without additional sandboxing.

## JSONL schema

Each line includes:

- `responses_create_params` — usual Gym fields; put the task and data instructions in `input` (system + user messages).
- `verifier_metadata` — **required** for verification:
  - `reference_notebook` (object): full nbformat v4 notebook JSON. Code and markdown cells are preserved in order; only code cells are run (or only their sources are sent to the judge when execution is skipped).
  - `skip_notebook_execution` (optional): if `true` or `false`, overrides the resources server config `skip_notebook_execution` for that `/verify` call.
  - `data_paths` (optional): list of `{"source": "/path/on/server/or/relative", "path": "relative/path/in/sandbox"}` entries; each `source` file or directory is copied into the sandbox at `path` before execution (same layout for reference and predicted runs). Paths are resolved on the machine running the resources server.

Top-level fields are forwarded to `/verify` by `simple_agent` (`extra="allow"`).

## Model output format

The assistant must emit one or more Python cells as fenced blocks:

```text
```python
print("example")
```
```

The number of fences **may** differ from the number of reference code cells; the judge sees predicted code plus the full executed notebook JSON for the predicted run.

Thinking wrappers (`</think>`, `<thinking>...</thinking>`) are removed before parsing fences.

## Configuration

See [`configs/data_analysis_notebook.yaml`](configs/data_analysis_notebook.yaml): `skip_notebook_execution` (default `false`), execution limits (`max_concurrent_executions`, `execute_timeout_secs`, `wall_clock_margin_secs`), and **judge** settings:

- `judge_model_server` — second `responses_api_models` instance (default YAML uses OpenAI-compatible `gpt-5-mini` via `notebook_judge_model`).
- `judge_responses_create_params` — `max_output_tokens` large enough for `VERDICT:` + optional `REASON:`.
- `judge_prompt_template_fpath` — rubric template under `prompt_templates/`.
- `judge_endpoint_max_concurrency` — max concurrent judge HTTP calls.
- `judge_probe_on_startup` — wait until the judge `/v1/responses` endpoint is reachable at startup (set `false` in tests).
- `executed_notebooks_directory` — where executed reference/predicted `.ipynb` files are written when execution is enabled.

Composite runs must include **both** the resources server config and the judge model server config, for example:

```bash
ng_run "+config_paths=[resources_servers/data_analysis_notebook/configs/data_analysis_notebook.yaml,responses_api_models/openai_model/configs/openai_model.yaml]"
```

(`env.yaml` should define `policy_base_url`, `policy_api_key`, and optionally override model names.)

The verify response includes `judge_evaluations` with the judge call and parsed `verdict_label` (`pass`, `fail`, `judge_error`, `judge_parsing_error`).

## Dependencies

The Jupyter stack (`nbclient`, `ipykernel`, …) is pinned in this server’s [`requirements.txt`](requirements.txt). `wcwidth` is listed explicitly because some minimal installs omit it and the kernel subprocess then fails to start.

## Running tests

```bash
cd resources_servers/data_analysis_notebook
uv venv && uv pip install -r requirements.txt
pytest tests/ -v
```

Or from the repo root: `ng_test +entrypoint=resources_servers/data_analysis_notebook`.
