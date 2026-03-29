import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    from pathlib import Path

    import plotnine as pn
    import polars as pl

    from icftsc.logging import logger
    from icftsc.modeling.util import get_arch
    from icftsc.scripts.tracking import collect_metrics

    return Path, collect_metrics, get_arch, logger, os, pl, pn


@app.cell
def _(Path, logger, os):
    logger.setLevel("INFO")
    figpath = Path("out/fig")
    os.makedirs(figpath, exist_ok=True)
    return (figpath,)


@app.cell(hide_code=True)
def _(get_arch, logger, pl):
    def add_metadata(df: pl.DataFrame) -> pl.DataFrame:
        """
        Manually add metadata to old experiments if missing.

        Add 'method', 'base_model' and 'architecture' columns
        for forwards compatibility.
        """
        cols = {"method", "base_model", "architecture"}
        if cols.issubset(df.columns):
            return df

        logger.info("add forwards compatible metadata to old experiment")
        return df.with_columns(
            pl.when(pl.col("run_name").str.contains("fine-tune"))
            .then(pl.lit("fine-tune"))
            .when(pl.col("run_name").str.contains("pretrained-prefix"))
            .then(pl.lit("prompt-tune-pretrained"))
            .when(pl.col("run_name").str.contains("random-prefix"))
            .then(pl.lit("prompt-tune-random"))
            .when(pl.col("run_name").str.contains("head"))
            .then(pl.lit("cls-head"))
            .otherwise(pl.lit("few-shot"))
            .alias("method"),
            pl.col("run_name")
            .str.replace("-(fine|pretrained|random|head|\\d-shot).*", "")
            .alias("base_model"),
            pl.col("model_type")
            .map_elements(get_arch, pl.String)
            .alias("architecture"),
        )

    return (add_metadata,)


@app.cell
def _(add_metadata, collect_metrics, os):
    _tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    _experiment = "icftsc-multinerd"

    df = collect_metrics(_tracking_uri, _experiment)
    df = add_metadata(df)
    return (df,)


@app.cell(hide_code=True)
def _(df, pl):
    _gpu_hr_cost = 0.5

    (
        df.select("train_runtime", "eval_runtime", "test_runtime")
        .sum()
        .unpivot(variable_name="task", value_name="seconds")
        .with_columns(
            [
                (pl.col("seconds") / 3600 * _gpu_hr_cost).round(2).alias("cost_eur"),
                (pl.col("seconds") / 3600).round(2).alias("gpu_hours"),
                pl.col("task").str.replace("_runtime", ""),
            ]
        )
        .select(["task", "gpu_hours", "cost_eur"])
        .pipe(
            lambda _d: pl.concat(
                [
                    _d,
                    pl.DataFrame(
                        {
                            "task": ["total"],
                            "gpu_hours": [round(_d["gpu_hours"].sum(), 2)],
                            "cost_eur": [round(_d["cost_eur"].sum(), 2)],
                        }
                    ),
                ]
            )
        )
    )
    return


@app.cell
def _(df, figpath, pl, pn):
    _df = df.with_columns(pl.col("total_parameters").mul(1e-6))
    _p = (
        pn.ggplot(_df)
        + pn.aes(
            x="total_parameters",
            y="test_f1",
            color="method",
            shape="architecture",
        )
        + pn.labs(
            x="Parameters (M)",
            y="F1",
            title="Performance Scaling",
        )
        + pn.scale_x_log10()
        + pn.ylim(0, 1)
        + pn.facet_wrap("model_type")
        + pn.geom_line()
        + pn.geom_point()
    )

    _p.save(figpath / "pref_scaling.png")
    _p
    return


@app.cell
def _(df, figpath, pl, pn):
    _df = df.with_columns(
        pl.col("train_runtime").mul(1 / 3600),
        pl.col("trainable_parameters").mul(1e-6),
    )

    _p = (
        pn.ggplot(_df)
        + pn.aes(
            x="trainable_parameters",
            y="train_runtime",
            color="method",
            shape="model_type",
        )
        + pn.labs(
            x="Trainable Parameters (M)",
            y="Train Runtime (h)",
            title="Compute Time Scaling",
        )
        + pn.scale_x_log10()
        + pn.scale_y_log10()
        + pn.geom_point()
    )

    _p.save(figpath / "compute_scaling.png")
    _p
    return


@app.cell
def _(df, figpath, pl, pn):
    _df = (
        df.filter(pl.col("method").is_in(("fine-tune", "prompt-tune-pretrained")))
        .sort("base_model", "method")
        .group_by("base_model")
        .agg(
            pl.col("test_f1").diff().last(),
            pl.col("architecture").last(),
        )
    )

    _p = (
        pn.ggplot(_df)
        + pn.aes(
            x="base_model",
            y="test_f1",
            fill="architecture",
        )
        + pn.labs(
            x="Model",
            y="Prompt-Tuning Absolute F1 Gain",
            title="Prompt-Tuning vs Fine-Tuning",
        )
        + pn.geom_bar(stat="identity")
        + pn.coord_flip()
    )

    _p.save(figpath / "perf_loss.png")
    _p
    return


if __name__ == "__main__":
    app.run()
