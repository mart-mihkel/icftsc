import polars as pl
from mlflow.tracking import MlflowClient

from icftsc.logging import logger

path = "out/metrics.csv"


def collect_metrics(mlflow_tracking_uri: str):
    client = MlflowClient(tracking_uri=mlflow_tracking_uri)
    experiment = client.get_experiment_by_name("icftsc")

    if experiment is None:
        raise ValueError("Experiment 'icftsc' not found")

    runs = client.search_runs(experiment.experiment_id, "")
    logger.info("found %d runs", len(runs))

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

    df = pl.DataFrame(rows)
    df.write_csv(path)

    logger.info("collected %d records and %d parameters", df.shape[0], df.shape[1])
    logger.info("save metrics to %s", path)
