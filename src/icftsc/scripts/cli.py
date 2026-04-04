from __future__ import annotations

from typing import Annotated, Any, Literal

from typer import Context, Option, Typer

from icftsc.types import DatasetName, PrefixInit, Task

app = Typer(no_args_is_help=True)


def save_params(args: dict[str, Any], run_name: str):
    import json
    import os

    from icftsc.logging import logger

    logdir = os.path.join("out", run_name)
    argpath = os.path.join(logdir, "cli_args.json")
    logger.debug("save cli args to '%s'", argpath)

    os.makedirs(logdir, exist_ok=True)
    with open(argpath, "w") as f:
        json.dump(args, f, indent=2)


@app.command(no_args_is_help=True)
def fine_tune(
    ctx: Context,
    model: Annotated[str, Option(help="HuggingFace model or path to checkpoint")],
    dataset: Annotated[DatasetName.__value__, Option(help="Dataset name")],
    task: Annotated[Task.__value__, Option(help="NLP task type")],
    head_only: Annotated[
        bool,
        Option(help="Freeze all parameters except for classifier head"),
    ],
    experiment: Annotated[str, Option(help="Experiment name for MLflow tracking")],
    run_name: Annotated[str, Option(help="Run name for MLflow tracking")],
    n_shot: Annotated[int, Option(help="Number of examples in system prompt")] = 3,
    epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
):
    from icftsc.logging import logger
    from icftsc.scripts.fine_tune import fine_tune

    logger.setLevel(log_level)
    save_params(ctx.params, run_name)
    fine_tune(
        model_path=model,
        dataset=dataset,
        task=task,
        head_only=head_only,
        n_shot=n_shot,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        experiment=experiment,
        run_name=run_name,
    )


@app.command(no_args_is_help=True)
def prompt_tune(
    ctx: Context,
    model: Annotated[str, Option(help="HuggingFace model or path to checkpoint")],
    dataset: Annotated[DatasetName.__value__, Option(help="Dataset name")],
    task: Annotated[Task.__value__, Option(help="NLP task type")],
    prefix_init: Annotated[
        PrefixInit.__value__,
        Option(help="Prefix initialization method"),
    ],
    experiment: Annotated[str, Option(help="Experiment name for MLflow tracking")],
    run_name: Annotated[str, Option(help="Run name for MLflow tracking")],
    n_shot: Annotated[int, Option(help="Number of examples in system prompt")] = 3,
    epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
):
    from icftsc.logging import logger
    from icftsc.scripts.prompt_tune import prompt_tune

    logger.setLevel(log_level)
    save_params(ctx.params, run_name)
    prompt_tune(
        model_path=model,
        dataset=dataset,
        task=task,
        prefix_init=prefix_init,
        n_shot=n_shot,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        experiment=experiment,
        run_name=run_name,
    )


@app.command(no_args_is_help=True)
def few_shot(
    ctx: Context,
    model: Annotated[str, Option(help="HuggingFace model or path to checkpoint")],
    dataset: Annotated[DatasetName.__value__, Option(help="Dataset name")],
    task: Annotated[Task.__value__, Option(help="NLP task type")],
    experiment: Annotated[str, Option(help="Experiment name for MLflow tracking")],
    run_name: Annotated[str, Option(help="Run name for MLflow tracking")],
    n_shot: Annotated[int, Option(help="Number of examples in system prompt")] = 3,
    batch_size: int = 8,
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
):
    from icftsc.logging import logger
    from icftsc.scripts.few_shot import few_shot

    logger.setLevel(log_level)
    save_params(ctx.params, run_name)
    few_shot(
        model_path=model,
        dataset=dataset,
        task=task,
        n_shot=n_shot,
        batch_size=batch_size,
        experiment=experiment,
        run_name=run_name,
    )


@app.command(no_args_is_help=True)
def predict_superglue(
    checkpoint: Annotated[str, Option(help="HuggingFace model or path to checkpoint")],
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
):
    from icftsc.logging import logger
    from icftsc.scripts.superglue import predict_boolq

    logger.setLevel(log_level)
    predict_boolq(checkpoint)


@app.command(no_args_is_help=True)
def collect_metrics(
    experiment: Annotated[str, Option(help="MLflow experiment name")],
    mlflow_tracking_uri: Annotated[
        str,
        Option(help="MLflow tracking server URI"),
    ] = "sqlite:///mlflow.db",
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
):
    from icftsc.logging import logger
    from icftsc.scripts.tracking import collect_metrics

    logger.setLevel(log_level)
    collect_metrics(mlflow_tracking_uri, experiment, write_csv=True)


if __name__ == "__main__":
    app()
