# In-Context Fine-Tuning for Sequence Classification

## Development

Use [uv](https://docs.astral.sh/uv/) for package management.

Setup a virtualenv with your corresponding accelerator

```bash
uv sync --extra [cpu|cu128]
```

## Usage

The cli contains scripts for fine-tuning, prefix-tuning, few-shot learning and
utilities.

```bash
uv run cli --help
```
