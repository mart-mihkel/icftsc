import os

from mlflow.tracking import MlflowClient
from polars import DataFrame

from icftsc.logging import logger


def collect_metrics(
    mlflow_tracking_uri: str,
    experiment: str,
    write_csv: bool,
) -> DataFrame:
    logger.info("connecting to '%s'", mlflow_tracking_uri)
    client = MlflowClient(tracking_uri=mlflow_tracking_uri)
    exp = client.get_experiment_by_name(experiment)

    if exp is None:
        raise RuntimeError(f"experiment '{experiment}' not found")

    runs = client.search_runs(exp.experiment_id, "")
    rows = []
    for run in runs:
        run_data = {
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
        }

        metrics = run.data.metrics
        for key, value in metrics.items():
            run_data[key] = value

        params = run.data.params
        for key, value in params.items():
            run_data[key] = value

        rows.append(run_data)

    logdir = os.path.join("log", "metrics")
    path = os.path.join(logdir, f"{experiment}.csv")
    os.makedirs(logdir, exist_ok=True)

    df = DataFrame(rows)
    logger.info("found %d runs with %d params", df.shape[0], df.shape[1])
    if write_csv:
        df.write_csv(path)
        logger.info("saved metrics to '%s'", path)

    return df
