from collections.abc import Callable
from typing import Annotated, Literal

from typer import Context, Option, Typer

from icftsc.types import DatasetName, PrefixInit, PromptMode, Task

app = Typer(no_args_is_help=True)


def timed(func: Callable) -> Callable:
    import time
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        from icftsc.logging import logger

        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start

        hours, remainder = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)
        logger.info("time elapsed %02d:%02d:%02d", hours, minutes, seconds)

        return result

    return wrapper


@app.callback()
def callback(log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"):
    from icftsc.logging import logger

    logger.setLevel(log_level)


@app.command()
@timed
def fine_tune(
    ctx: Context,
    model: Annotated[str, Option()],
    run_name: Annotated[str, Option()],
    dataset: Annotated[DatasetName.__value__, Option()],
    task: Annotated[Task.__value__, Option()],
    head_only: Annotated[bool, Option()],
    workers: int = 0,
    epochs: int = 1,
    batch_size: int = 8,
    effective_batch_size: int = 16,
    learning_rate: float = 5e-5,
    grad_chkpts: bool = False,
    mlflow_tracking_uri: str | None = None,
):
    from icftsc.scripts.common import save_params
    from icftsc.scripts.fine_tune import fine_tune

    save_params(ctx.params, run_name)
    fine_tune(
        model_path=model,
        run_name=run_name,
        dataset=dataset,
        task=task,
        head_only=head_only,
        workers=workers,
        epochs=epochs,
        batch_size=batch_size,
        effective_batch_size=effective_batch_size,
        learning_rate=learning_rate,
        grad_chkpts=grad_chkpts,
        mlflow_tracking_uri=mlflow_tracking_uri,
    )


@app.command()
@timed
def prompt_tune(
    ctx: Context,
    model: Annotated[str, Option()],
    run_name: Annotated[str, Option()],
    dataset: Annotated[DatasetName.__value__, Option()],
    task: Annotated[Task.__value__, Option()],
    prefix_init: Annotated[PrefixInit.__value__, Option()],
    workers: int = 0,
    epochs: int = 1,
    batch_size: int = 8,
    effective_batch_size: int = 16,
    learning_rate: float = 5e-5,
    grad_chkpts: bool = False,
    mlflow_tracking_uri: str | None = None,
):
    from icftsc.scripts.common import save_params
    from icftsc.scripts.prompt_tune import prompt_tune

    save_params(ctx.params, run_name)
    prompt_tune(
        model_path=model,
        run_name=run_name,
        dataset=dataset,
        task=task,
        prefix_init=prefix_init,
        workers=workers,
        epochs=epochs,
        batch_size=batch_size,
        effective_batch_size=effective_batch_size,
        learning_rate=learning_rate,
        grad_chkpts=grad_chkpts,
        mlflow_tracking_uri=mlflow_tracking_uri,
    )


@app.command()
@timed
def few_shot(
    ctx: Context,
    model: Annotated[str, Option()],
    run_name: Annotated[str, Option()],
    dataset: Annotated[DatasetName.__value__, Option()],
    task: Annotated[Task.__value__, Option()],
    prompt_mode: Annotated[PromptMode.__value__, Option()],
    n_shot: Annotated[int, Option()],
    workers: int = 0,
    batch_size: int = 8,
    mlflow_tracking_uri: str | None = None,
):
    from icftsc.scripts.common import save_params
    from icftsc.scripts.few_shot import few_shot

    save_params(ctx.params, run_name)
    few_shot(
        model_path=model,
        run_name=run_name,
        dataset=dataset,
        task=task,
        prompt_mode=prompt_mode,
        n_shot=n_shot,
        workers=workers,
        batch_size=batch_size,
        mlflow_tracking_uri=mlflow_tracking_uri,
    )


@app.command()
@timed
def predict_superglue(checkpoint: Annotated[str, Option()]):
    from icftsc.scripts.superglue import predict_boolq

    predict_boolq(checkpoint=checkpoint)


@app.command()
@timed
def collect_metrics(mlflow_tracking_uri: Annotated[str, Option()]):
    from icftsc.scripts.tracking import collect_metrics

    collect_metrics(mlflow_tracking_uri=mlflow_tracking_uri)


if __name__ == "__main__":
    app()
