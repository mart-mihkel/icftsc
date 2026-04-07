# In-Context Fine-Tuning for Sequence Classification

## Development

Use [uv](https://docs.astral.sh/uv/) for package management.

Setup a virtualenv with torch backend for cpu or cuda. When using cuda you
should also have cuda-toolkit available on the system to compile `flash-attn`.

```bash
MAX_JOBS=[n-jobs] uv sync --compile-bytecode --extra [cpu|cu128]
```

## Usage

The `cli` installed in the virtualenv contains scripts for fine-tuning,
prompt-tuning, few-shot learning and utilities.

```bash
cli --help
```
