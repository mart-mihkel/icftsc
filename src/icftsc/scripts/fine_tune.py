import mlflow
from transformers import AutoConfig

from icftsc.datasets.util import get_collator, load_data, load_tokenizer
from icftsc.logging import logger
from icftsc.metrics import get_metrics_fn
from icftsc.modeling.util import get_model, train
from icftsc.types import DatasetName, Task


def fine_tune(
    model_path: str,
    run_name: str,
    dataset: DatasetName,
    task: Task,
    head_only: bool,
    n_shot: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    grad_chkpts: bool,
    mlflow_tracking_uri: str,
    experiment: str,
):
    logger.info("load model config")
    config = AutoConfig.from_pretrained(model_path)

    logger.info("load pretrained tokenizer")
    tokenizer = load_tokenizer(model_path)
    collate_fn = get_collator(tokenizer, task)
    metrics_fn = get_metrics_fn(tokenizer, task)

    logger.info("load '%s' dataset", dataset)
    model_type = config.model_type
    data, info = load_data(tokenizer, dataset, model_type, task, n_shot)

    if dataset == "boolq" or dataset == "wic":
        logger.warning("drop superglue test data, labels are private")
        data.pop("test")

    logger.info("load pretrained '%s' for '%s'", model_path, task)
    model = get_model(tokenizer, model_path, info, task, head_only)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info("total parameters %d", total)
    logger.info("trainable parameters %d", trainable)
    logger.info(
        "tracking '%s' of experiment '%s' at '%s'",
        run_name,
        experiment,
        mlflow_tracking_uri,
    )

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment)
    mlflow.start_run(run_name=run_name)
    mlflow.log_params(
        {
            "task": task,
            "dataset": dataset,
            "head_only": head_only,
            "system_prompt": info["system_prompt"],
        }
    )

    mlflow.log_metrics(
        {
            "n_shot": n_shot,
            "total_parameters": total,
            "trainable_parameters": trainable,
        }
    )

    train(
        model=model,
        data=data,
        collate_fn=collate_fn,
        metrics_fn=metrics_fn,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        grad_chkpts=grad_chkpts,
        run_name=run_name,
        report_to="mlflow",
    )

    mlflow.end_run()
