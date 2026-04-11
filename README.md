# In-Context Fine-Tuning for Sequence Classification

## Development

Use [uv](https://docs.astral.sh/uv/) for package management.

Setup a virtualenv with torch backend for cpu or cuda. When using cuda you
should also have cuda-toolkit on the system to compile flash attention.

```bash
make install BACKEND=[cpu|cu128]
```

You can limit the number of compile workers by setting the `MAX_JOBS` variable.

## Usage

The `cli` installed in the virtualenv contains scripts for fine-tuning,
prompt-tuning, few-shot learning and utilities.

```bash
cli --help
```

Example experiments are in the [run](./run) directory.

## Tracking

Experiment are tracked to `mlflow` and can be seen by serving the ui

```bash
mlflow ui
```

For tracking on a remote server set the `MLFLOW_TRACKING_URI` environment
variable.
