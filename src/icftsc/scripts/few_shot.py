from typing import cast

import mlflow
from torch.utils.data import Dataset
from transformers import AutoConfig

from icftsc.datasets.util import get_collator, load_data, load_tokenizer
from icftsc.logging import logger
from icftsc.metrics import get_metrics_fn
from icftsc.modeling import get_arch, get_model, get_trainer
from icftsc.types import DatasetName, Task


def few_shot(
    model_path: str,
    run_name: str,
    dataset: DatasetName,
    task: Task,
    n_shot: int,
    batch_size: int,
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
        logger.warning("using superglue dev data, test labels are private")
        data["test"] = data["dev"]

    logger.info("load pretrained '%s' for '%s'", model_path, task)
    model = get_model(tokenizer, model_path, info, task, head_only=False)

    total = sum(p.numel() for p in model.parameters())

    logger.info("tracking '%s' of experiment '%s'", run_name, experiment)

    mlflow.set_experiment(experiment)
    mlflow.start_run(run_name=run_name)
    mlflow.log_param("task", task)
    mlflow.log_param("n_shot", n_shot)
    mlflow.log_param("dataset", dataset)
    mlflow.log_param("base_model", model_path)
    mlflow.log_param("method", f"{n_shot}-shot")
    mlflow.log_param("architecture", get_arch(model_type))
    mlflow.log_param("system_prompt", info["system_prompt"])
    mlflow.log_metric("train_samples", len(data["train"]))
    mlflow.log_metric("dev_samples", len(data["dev"]))
    mlflow.log_metric("test_samples", len(data["test"]))
    mlflow.log_metric("total_parameters", total)
    mlflow.log_metric("trainable_parameters", 0)

    trainer = get_trainer(
        model=model,
        data=data,
        collate_fn=collate_fn,
        metrics_fn=metrics_fn,
        batch_size=batch_size,
        run_name=run_name,
        report_to="mlflow",
    )

    logger.info("start test evaluation")
    test = cast(Dataset, data["test"])
    metrics = trainer.evaluate(test, metric_key_prefix="test")
    logger.info(metrics)

    mlflow.end_run()
