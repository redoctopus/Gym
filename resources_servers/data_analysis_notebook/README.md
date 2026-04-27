# Data analysis notebook (resources server)

Verifies single-turn model answers by running **reference** code cells (from a ground-truth notebook) and **predicted** cells (parsed from the model reply) in **separate** temporary working directories and fresh IPython kernels, then comparing **merged** outputs (stdout, stderr, `text/plain` displays, and optionally PNG).

Markdown cells in the reference notebook are ignored; only `cell_type == "code"` sources are executed, in order.

## Security

`/verify` executes **untrusted model code** with the same OS privileges as the server. Run in an isolated environment, keep `execute_timeout_secs` and `max_concurrent_executions` low, and do not expose this endpoint to untrusted callers without additional sandboxing.

## JSONL schema

Each line includes:

- `responses_create_params` — usual Gym fields; put the task and data instructions in `input` (system + user messages).
- `verifier_metadata` — **required** for verification:
  - `reference_notebook` (object): full nbformat v4 notebook JSON. Only code cells are run.
  - `data_paths` (optional): list of `{"source": "/path/on/server/or/relative", "path": "relative/path/in/sandbox"}` entries; each `source` file or directory is copied into the sandbox at `path` before execution (same layout for reference and predicted runs). Paths are resolved on the machine running the resources server.
  - `execution_log_directory` (optional, server path): if set, overrides the resources server config field of the same name. When a log directory is active (config and/or this field), each `/verify` writes two append-only files, `{execution_log_stem}_reference.log` and `{execution_log_stem}_predicted.log` (or a random `stem` if `execution_log_stem` is omitted). The top of each file includes a **Task question** line: the first line of the last `user` message in `responses_create_params.input` (or the first line of `input` if it is a string), truncated to a max character count. After each code cell, the log records that cell’s source and **only that cell’s** structured output (not a cumulative merge across prior cells), then flushes—so a wall-clock kill can still show completed cells without repeated earlier output. Treat paths like `data_paths`: the dataset/operator is trusted. Very large cell output can produce large log files; source and JSON are truncated in the log beyond fixed limits.
  - `execution_log_stem` (optional): non-empty string used as the basename for the two log files; must be a single path component (no `/` or `\\`). If missing and logging is enabled, a unique id is used per request.

Top-level fields are forwarded to `/verify` by `simple_agent` (`extra="allow"`).

## Model output format

The assistant must emit one or more Python cells as fenced blocks:

```text
```python
print("example")
```
```

The number of fences **may** differ from the number of reference code cells; comparison uses **merged** stdout/stderr and display outputs across all executed cells.

Thinking wrappers (`</think>`, `<thinking>...</thinking>`) are removed before parsing fences.

## Configuration

See `[configs/data_analysis_notebook.yaml](configs/data_analysis_notebook.yaml)`: `max_concurrent_executions`, `execute_timeout_secs`, `wall_clock_margin_secs`, `image_compare_mode` (`exact` | `none`), and optional `execution_log_directory` (set to a writable path on the server to enable per-request logs unless overridden in `verifier_metadata`).

## Dependencies

The Jupyter stack (`nbclient`, `ipykernel`, …) is pinned in this server’s `[requirements.txt](requirements.txt)`. `wcwidth` is listed explicitly because some minimal installs omit it and the kernel subprocess then fails to start.

## Running tests

```bash
cd resources_servers/data_analysis_notebook
uv venv && uv pip install -r requirements.txt
pytest tests/ -v
```

Or from the repo root: `ng_test +entrypoint=resources_servers/data_analysis_notebook`.