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
  - `data_files` (optional): map of relative path → file contents. Use UTF-8 text for CSV etc. For binary, use the string `base64:<standard_base64>`.
  - `data_paths` (optional): list of `{"source": "/path/on/server/or/relative", "path": "relative/path/in/sandbox"}` entries; each `source` file or directory is copied into the sandbox at `path` before execution (same layout for reference and predicted runs). Paths are resolved on the machine running the resources server.

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

See `[configs/data_analysis_notebook.yaml](configs/data_analysis_notebook.yaml)`: `max_concurrent_executions`, `execute_timeout_secs`, `wall_clock_margin_secs`, `image_compare_mode` (`exact` | `none`).

## Dependencies

The Jupyter stack (`nbclient`, `ipykernel`, …) is pinned in this server’s `[requirements.txt](requirements.txt)`. `wcwidth` is listed explicitly because some minimal installs omit it and the kernel subprocess then fails to start.

## Running tests

```bash
cd resources_servers/data_analysis_notebook
uv venv && uv pip install -r requirements.txt
pytest tests/ -v
```

Or from the repo root: `ng_test +entrypoint=resources_servers/data_analysis_notebook`.