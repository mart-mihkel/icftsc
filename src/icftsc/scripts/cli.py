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


@app.command()
def fine_tune(
    ctx: Context,
    model: Annotated[str, Option()],
    dataset: Annotated[DatasetName.__value__, Option()],
    task: Annotated[Task.__value__, Option()],
    head_only: Annotated[bool, Option()],
    experiment: Annotated[str, Option()],
    run_name: Annotated[str, Option()],
    n_shot: int = 0,
    epochs: int = 1,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    grad_chkpts: bool = False,
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
        grad_chkpts=grad_chkpts,
        experiment=experiment,
        run_name=run_name,
    )


@app.command()
def prompt_tune(
    ctx: Context,
    model: Annotated[str, Option()],
    dataset: Annotated[DatasetName.__value__, Option()],
    task: Annotated[Task.__value__, Option()],
    prefix_init: Annotated[PrefixInit.__value__, Option()],
    experiment: Annotated[str, Option()],
    run_name: Annotated[str, Option()],
    n_shot: int = 5,
    epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    grad_chkpts: bool = False,
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
        grad_chkpts=grad_chkpts,
        experiment=experiment,
        run_name=run_name,
    )


@app.command()
def few_shot(
    ctx: Context,
    model: Annotated[str, Option()],
    dataset: Annotated[DatasetName.__value__, Option()],
    task: Annotated[Task.__value__, Option()],
    experiment: Annotated[str, Option()],
    run_name: Annotated[str, Option()],
    n_shot: int = 5,
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


@app.command()
def predict_superglue(
    checkpoint: Annotated[str, Option()],
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
):
    from icftsc.logging import logger
    from icftsc.scripts.superglue import predict_boolq

    logger.setLevel(log_level)
    predict_boolq(checkpoint)


@app.command()
def collect_metrics(
    mlflow_tracking_uri: Annotated[str, Option()],
    experiment: Annotated[str, Option()],
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
):
    from icftsc.logging import logger
    from icftsc.scripts.tracking import collect_metrics

    logger.setLevel(log_level)
    collect_metrics(mlflow_tracking_uri, experiment, write_csv=True)


if __name__ == "__main__":
    app()
