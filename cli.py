from typing import Annotated, Literal, Callable
from functools import wraps

import time
from typer import Option, Typer

app = Typer(no_args_is_help=True, add_completion=False)


def timed(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        from icft.logging import logger

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
    from icft.logging import logger

    logger.setLevel(log_level)


@app.command()
@timed
def fine_tune(
    task: Annotated[Literal["seq2seq", "seq-cls", "causal-lm"], Option()],
    dataset: Annotated[Literal["multinerd"], Option()],
    system_prompt: Annotated[Literal["ner", "random", "none"], Option()],
    head_only: Annotated[bool, Option()],
    model: Annotated[str, Option()],
    run_name: Annotated[str, Option()],
    grad_chkpts: bool = False,
    epochs: int = 1,
    lr: float = 5e-5,
    batch_size: int = 8,
    workers: int = 4,
):
    from icft.scripts.fine_tune import main

    main(
        task=task,
        dataset=dataset,
        system_prompt=system_prompt,
        head_only=head_only,
        model_path=model,
        run_name=run_name,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        workers=workers,
        grad_chkpts=grad_chkpts,
    )


@app.command()
@timed
def prompt_tune(
    task: Annotated[Literal["seq2seq", "seq-cls", "causal-lm"], Option()],
    dataset: Annotated[Literal["multinerd"], Option()],
    prefix_init: Annotated[Literal["pretrained", "labels", "random"], Option()],
    model: Annotated[str, Option()],
    run_name: Annotated[str, Option()],
    grad_chkpts: bool = False,
    epochs: int = 1,
    lr: float = 5e-5,
    batch_size: int = 8,
    workers: int = 4,
):
    from icft.scripts.prompt_tune import main

    main(
        task=task,
        dataset=dataset,
        prefix_init=prefix_init,
        model_path=model,
        run_name=run_name,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        workers=workers,
        grad_chkpts=grad_chkpts,
    )


if __name__ == "__main__":
    app()
