# Data analysis notebook (resources server)

Verifies single-turn model answers by running **reference** code cells (from a ground-truth notebook) and **predicted** cells (parsed from the model reply) in **separate** temporary working directories and fresh IPython kernels. **Merged** execution outputs (stdout, stderr, `text/plain`, optional PNG counts) are sent to an **LLM judge** together with the task text and predicted code. The judge decides whether the prediction **correctly and completely** satisfies the task, using reference execution output as a guide when it is available (semantic match; not byte-identical output).

If the reference notebook fails to execute, verification still runs for the predicted notebook only: the judge does not receive reference output (only the task and predicted code/output).

**Charts / images:** PNG pixel data is **not** sent to the judge (only how many figures were produced). Pixel-level chart equivalence would require a separate vision-capable judge.

Markdown cells in the reference notebook are ignored; only `cell_type == "code"` sources are executed, in order.

## Security

`/verify` executes **untrusted model code** with the same OS privileges as the server. Run in an isolated environment, keep `execute_timeout_secs` and `max_concurrent_executions` low, and do not expose this endpoint to untrusted callers without additional sandboxing.

## JSONL schema

Each line includes:

- `responses_create_params` — usual Gym fields; put the task and data instructions in `input` (system + user messages).
- `verifier_metadata` — **required** for verification:
  - `reference_notebook` (object): full nbformat v4 notebook JSON. Only code cells are run.
  - `data_paths` (optional): list of `{"source": "/path/on/server/or/relative", "path": "relative/path/in/sandbox"}` entries; each `source` file or directory is copied into the sandbox at `path` before execution (same layout for reference and predicted runs). Paths are resolved on the machine running the resources server.
  - `execution_log_directory` (optional, server path): if set, overrides the resources server config field of the same name. When a log directory is active (config and/or this field), each `/verify` writes two append-only files directly in that directory: `<YYYY-mm-dd_HH-MM-SS_ffffff>_{stem}_reference.log` and the same prefix with `_predicted.log` (local time, microsecond resolution in the prefix). Unless `execution_log_stem` is set to a valid single path component, `stem` is derived from the first line of the task (last user message), then the first `data_paths` filename, then the reference notebook’s `metadata.title` if needed; the stem is sanitized and length-capped. If nothing usable remains, a random id is used. On the rare case that those paths already exist, an extra short random segment is inserted after the timestamp. The top of each file includes a **Task question** line: the first line of the last `user` message in `responses_create_params.input` (or the first line of `input` if it is a string), truncated to a max character count. After each code cell, the log records that cell’s source and **only that cell’s** structured output (not a cumulative merge across prior cells), then flushes—so a wall-clock kill can still show completed cells without repeated earlier output. Treat paths like `data_paths`: the dataset/operator is trusted. Very large cell output can produce large log files; source and JSON are truncated in the log beyond fixed limits.
  - `execution_log_stem` (optional): if set to a single safe path component (no `/` or `\\`), used as the `{stem}` part of the filenames after sanitization and length capping; otherwise ignored and the automatic stem above applies.

Top-level fields are forwarded to `/verify` by `simple_agent` (`extra="allow"`).

## Model output format

The assistant must emit one or more Python cells as fenced blocks:

```text
```python
print("example")
```
```

The number of fences **may** differ from the number of reference code cells; the judge sees **merged** stdout/stderr and display outputs across all executed cells (after redaction/truncation for the prompt).

Thinking wrappers (`</think>`, `<thinking>...</thinking>`) are removed before parsing fences.

## Configuration

See [`configs/data_analysis_notebook.yaml`](configs/data_analysis_notebook.yaml): execution limits (`max_concurrent_executions`, `execute_timeout_secs`, `wall_clock_margin_secs`), `image_compare_mode` (`exact` | `none`) for building merged signatures (PNG lists), optional `execution_log_directory`, and **judge** settings:

- `judge_model_server` — second `responses_api_models` instance (default YAML uses OpenAI-compatible `gpt-4o-mini` via `notebook_judge_model`).
- `judge_responses_create_params` — `max_output_tokens` large enough for `VERDICT:` + optional `REASON:`.
- `judge_prompt_template_fpath` — rubric template under `prompt_templates/`.
- `judge_max_output_chars` — per-field truncation for reference/predicted text sent to the judge.
- `judge_probe_on_startup` — wait until the judge `/v1/responses` endpoint is reachable at startup (set `false` in tests).

Composite runs must include **both** the resources server config and the judge model server config, for example:

```bash
ng_run "+config_paths=[resources_servers/data_analysis_notebook/configs/data_analysis_notebook.yaml,responses_api_models/openai_model/configs/openai_model.yaml]"
```

(`env.yaml` should define `policy_base_url`, `policy_api_key`, and optionally override model names.)

The verify response may include `judge_evaluations` with the judge call and parsed `verdict_label` (`pass`, `fail`, `judge_error`, `judge_parsing_error`).

## Dependencies

The Jupyter stack (`nbclient`, `ipykernel`, …) is pinned in this server’s `[requirements.txt](requirements.txt)`. `wcwidth` is listed explicitly because some minimal installs omit it and the kernel subprocess then fails to start.

## Running tests

```bash
cd resources_servers/data_analysis_notebook
uv venv && uv pip install -r requirements.txt
pytest tests/ -v
```

Or from the repo root: `ng_test +entrypoint=resources_servers/data_analysis_notebook`.