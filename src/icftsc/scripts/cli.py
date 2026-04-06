from __future__ import annotations

from typing import Annotated, Literal

from typer import Option, Typer

from icftsc.types import DatasetName, PrefixInit, Task

app = Typer(no_args_is_help=True)


@app.command(no_args_is_help=True)
def fine_tune(
    model: Annotated[str, Option(help="HuggingFace model or path to checkpoint")],
    dataset: Annotated[DatasetName.__value__, Option(help="Dataset name")],
    task: Annotated[Task.__value__, Option(help="NLP task type")],
    head_only: Annotated[
        bool,
        Option(help="Freeze all parameters except for classifier head"),
    ] = False,
    n_shot: Annotated[int, Option(help="Number of examples in system prompt")] = 3,
    n_train_samples: Annotated[
        int | None,
        Option(help="If present take a subset of tokenized train data"),
    ] = None,
    n_dev_samples: Annotated[
        int | None,
        Option(help="If present take a subset of tokenized dev data"),
    ] = None,
    do_eval: Annotated[bool, Option(help="Run evalutaion during training")] = False,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    experiment: Annotated[
        str | None,
        Option(help="Experiment for tracking, inferred from parameters by default"),
    ] = None,
    run_name: Annotated[
        str | None,
        Option(help="Run name for tracking, inferred from parameters by default"),
    ] = None,
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
):
    from icftsc.logging import logger
    from icftsc.scripts.fine_tune import fine_tune

    logger.setLevel(log_level)
    fine_tune(
        model_path=model,
        dataset=dataset,
        task=task,
        head_only=head_only,
        n_shot=n_shot,
        n_train_samples=n_train_samples,
        n_dev_samples=n_dev_samples,
        do_eval=do_eval,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        experiment=experiment,
        run_name=run_name,
    )


@app.command(no_args_is_help=True)
def prompt_tune(
    model: Annotated[str, Option(help="HuggingFace model or path to checkpoint")],
    dataset: Annotated[DatasetName.__value__, Option(help="Dataset name")],
    task: Annotated[Task.__value__, Option(help="NLP task type")],
    prefix_init: Annotated[
        PrefixInit.__value__,
        Option(help="Prefix initialization method"),
    ],
    n_shot: Annotated[int, Option(help="Number of examples in system prompt")] = 3,
    n_train_samples: Annotated[
        int | None,
        Option(help="If present take a subset of tokenized train data"),
    ] = None,
    n_dev_samples: Annotated[
        int | None,
        Option(help="If present take a subset of tokenized dev data"),
    ] = None,
    do_eval: Annotated[bool, Option(help="Run evalutaion during training")] = False,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    experiment: Annotated[
        str | None,
        Option(help="Experiment for tracking, inferred from parameters by default"),
    ] = None,
    run_name: Annotated[
        str | None,
        Option(help="Run name for tracking, inferred from parameters by default"),
    ] = None,
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
):
    from icftsc.logging import logger
    from icftsc.scripts.prompt_tune import prompt_tune

    logger.setLevel(log_level)
    prompt_tune(
        model_path=model,
        dataset=dataset,
        task=task,
        prefix_init=prefix_init,
        n_shot=n_shot,
        n_train_samples=n_train_samples,
        n_dev_samples=n_dev_samples,
        do_eval=do_eval,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        experiment=experiment,
        run_name=run_name,
    )


@app.command(no_args_is_help=True)
def few_shot(
    model: Annotated[str, Option(help="HuggingFace model or path to checkpoint")],
    dataset: Annotated[DatasetName.__value__, Option(help="Dataset name")],
    task: Annotated[Task.__value__, Option(help="NLP task type")],
    n_shot: Annotated[int, Option(help="Number of examples in system prompt")] = 3,
    batch_size: int = 8,
    experiment: Annotated[
        str | None,
        Option(help="Experiment for tracking, inferred from parameters by default"),
    ] = None,
    run_name: Annotated[
        str | None,
        Option(help="Run name for tracking, inferred from parameters by default"),
    ] = None,
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
):
    from icftsc.logging import logger
    from icftsc.scripts.few_shot import few_shot

    logger.setLevel(log_level)
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
