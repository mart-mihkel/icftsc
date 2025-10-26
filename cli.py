import logging
from typing import Annotated, Literal

from typer import Option, Typer

logger = logging.getLogger(__name__)

app = Typer(
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)


@app.callback()
def callback(
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
):
    level = getattr(logging, log_level)
    if log_level == "DEBUG":
        logging.basicConfig(level=level, format="%(levelname)s:%(name)s:%(message)s")
    else:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.basicConfig(level=level, format="%(message)s")


@app.command()
def fine_tune(
    task: Annotated[Literal["seq2seq", "seq-cls", "causal-lm"], Option()],
    dataset: Annotated[Literal["multinerd"], Option()],
    system_prompt: Annotated[Literal["ner", "random", "none"], Option()],
    head_only: Annotated[bool, Option()],
    model: Annotated[str, Option()],
    run_name: Annotated[str, Option()],
    epochs: int = 1,
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
        batch_size=batch_size,
        workers=workers,
    )


@app.command()
def prompt_tune(
    task: Annotated[Literal["seq2seq", "seq-cls", "causal-lm"], Option()],
    dataset: Annotated[Literal["multinerd"], Option()],
    prefix_init: Annotated[Literal["pretrained", "labels", "random"], Option()],
    model: Annotated[str, Option()],
    run_name: Annotated[str, Option()],
    epochs: int = 1,
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
        batch_size=batch_size,
        workers=workers,
    )


@app.command()
def collect_metrics():
    import os
    import json
    import polars as pl

    records = []
    for run in os.listdir("out"):
        path = f"out/{run}/test_results.json"
        if not os.path.exists(path):
            continue

        with open(path) as f:
            res = json.load(f)
            res["run"] = run
            records.append(res)

    out = "out/test_results.csv"
    df = pl.from_dicts(records)
    df.write_csv(out)

    logger.info(df)
    logger.info("saved to '%s'", out)


if __name__ == "__main__":
    app()
