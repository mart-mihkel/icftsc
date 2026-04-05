import marimo

__generated_with = "0.22.4"
app = marimo.App(width="medium")

with app.setup:
    import os
    from pathlib import Path

    import plotnine as pn
    import polars as pl

    from icftsc.logging import logger
    from icftsc.scripts.tracking import collect_metrics

    logger.setLevel("INFO")
    figpath = Path("out/fig/multinerd")
    os.makedirs(figpath, exist_ok=True)


@app.cell
def _():
    _tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    _experiment = "icftsc-multinerd"

    df = collect_metrics(_tracking_uri, _experiment, write_csv=False)
    return (df,)


@app.cell
def _(df):
    (
        df.with_columns(
            pl.col("end_time")
            .sub(pl.col("start_time"))
            .mul(1 / 1000)
            .alias("total_runtime")
        )
        .select("train_runtime", "test_runtime", "total_runtime")
        .sum()
        .unpivot(variable_name="task", value_name="time")
        .with_columns(
            pl.col("time").mul(0.5 / 3600).round(2).alias("cost_eur"),
            pl.col("time").mul(1 / 3600).round(2).alias("gpu_hours"),
            pl.col("task").str.replace("_runtime", ""),
        )
        .select("task", "gpu_hours", "cost_eur")
    )
    return


@app.cell
def _(df):
    _p = (
        pn.ggplot(df)
        + pn.aes(
            x="total_parameters",
            y="test_f1",
            fill="method",
            shape="architecture",
        )
        + pn.labs(
            x="Parameters",
            y="F1",
            title="Performance Scaling",
        )
        + pn.scale_x_log10()
        + pn.ylim(0, 1)
        + pn.facet_wrap("model_type")
        + pn.geom_line(pn.aes(color="method"))
        + pn.geom_point(stroke=0.25, size=2)
    )

    _p.save(figpath / "perf_scaling.png")
    _p
    return


@app.cell
def _(df):
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
def _(df):
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
