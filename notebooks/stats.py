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
    figpath = Path("out/fig/multinerd")
    os.makedirs(figpath, exist_ok=True)
    return (figpath,)


@app.cell(hide_code=True)
def _(get_arch, logger, pl):
    def wrangle_metadata(df: pl.DataFrame) -> pl.DataFrame:
        """
        Manually add or modify metadata of old experiments.

        Add 'method', 'base_model' and 'architecture' columns
        for forwards compatibility.
        Remove train and eval runtime from few-shot runs.
        """
        logger.info("add forwards compatible metadata to old experiment")
        _method = (
            pl.when(pl.col("run_name").str.contains("fine-tune"))
            .then(pl.lit("fine-tune"))
            .when(pl.col("run_name").str.contains("pretrained-prefix"))
            .then(pl.lit("prompt-tune-pretrained"))
            .when(pl.col("run_name").str.contains("random-prefix"))
            .then(pl.lit("prompt-tune-random"))
            .when(pl.col("run_name").str.contains("head"))
            .then(pl.lit("cls-head"))
            .otherwise(pl.lit("few-shot"))
            .alias("method")
        )

        _train_runtime = (
            pl.when(pl.col("run_name").str.contains(r"fine|prefix|head"))
            .then(pl.col("train_runtime"))
            .otherwise(None)
            .alias("train_runtime")
        )

        _eval_runtime = (
            pl.when(pl.col("run_name").str.contains(r"fine|prefix|head"))
            .then(pl.col("eval_runtime"))
            .otherwise(None)
            .alias("eval_runtime")
        )

        _base_model = (
            pl.col("run_name")
            .str.replace(r"-(fine|pretrained|random|cls|head|few|\d-shot).*", "")
            .alias("base_model")
        )

        _arch = (
            pl.col("model_type").map_elements(get_arch, pl.String).alias("architecture")
        )

        return df.with_columns(
            _method,
            _train_runtime,
            _eval_runtime,
            _base_model,
            _arch,
        )

    return (wrangle_metadata,)


@app.cell
def _(collect_metrics, os, wrangle_metadata):
    _tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    _experiment = "icftsc-multinerd"

    df = collect_metrics(_tracking_uri, _experiment)
    df = wrangle_metadata(df)
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
            fill="method",
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
        + pn.geom_line(pn.aes(color="method"))
        + pn.geom_point(stroke=0.25, size=2)
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
            fill="method",
            shape="model_type",
        )
        + pn.labs(
            x="Trainable Parameters (M)",
            y="Train Runtime (h)",
            title="Compute Time Scaling",
        )
        + pn.scale_x_log10()
        + pn.scale_y_log10()
        + pn.geom_point(stroke=0.25, size=2)
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
            pl.col("test_f1").diff().last().alias("f1_gain_abs"),
            pl.col("test_f1")
            .diff()
            .last()
            .mul(1 / pl.col("test_f1").first())
            .alias("f1_gain_rel"),
            pl.col("architecture").last(),
            pl.col("model_type").last(),
            pl.col("total_parameters").first(),
        )
    )

    _p = (
        pn.ggplot(_df)
        + pn.aes(
            x="f1_gain_abs",
            y="f1_gain_rel",
            fill="architecture",
            shape="model_type",
            size="total_parameters",
        )
        + pn.labs(
            x="PT Absolute F1 Increase",
            y="PT Relative F1 Increase",
            title="Prompt-Tuning vs Fine-Tuning",
        )
        + pn.geom_point(stroke=0.25)
    )

    _p.save(figpath / "pt_vs_ft.png")
    _p
    return


if __name__ == "__main__":
    app.run()
