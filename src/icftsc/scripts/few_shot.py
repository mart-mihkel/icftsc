from typing import cast

import mlflow
from datasets.splits import Split
from torch.utils.data import Dataset
from transformers import AutoConfig

from icftsc.datasets.util import get_collator, load_data, load_tokenizer
from icftsc.logging import logger
from icftsc.metrics import get_metrics_fn
from icftsc.modeling import get_arch, get_model, get_trainer
from icftsc.types import DatasetName, Task


def few_shot(
    model_path: str,
    dataset: DatasetName,
    task: Task,
    n_shot: int,
    batch_size: int,
    experiment: str | None,
    run_name: str | None,
):
    logger.info("load config for '%s'", model_path)
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = load_tokenizer(model_path)
    collate_fn = get_collator(tokenizer, task)
    metrics_fn = get_metrics_fn(tokenizer, task)

    model_type = config.model_type
    split = cast(Split, {"test": "test"})
    data, info = load_data(tokenizer, dataset, model_type, task, n_shot, split=split)

    if dataset == "boolq" or dataset == "wic":
        logger.warning("using superglue dev data, test labels are private")
        data["test"] = data["dev"]

    logger.info("load '%s'", model_path)
    model = get_model(tokenizer, model_path, info, task, head_only=False)

    total = sum(p.numel() for p in model.parameters())
    if experiment is None:
        experiment = f"icftsc-{dataset}"

    if run_name is None:
        run_name = f"{model_path}/few-shot/{task}"

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
    mlflow.log_metric("train_samples", 0)
    mlflow.log_metric("dev_samples", 0)
    mlflow.log_metric("test_samples", len(data["test"]))
    mlflow.log_metric("total_parameters", total)
    mlflow.log_metric("trainable_parameters", 0)

    trainer = get_trainer(
        model=model,
        data=data,
        collate_fn=collate_fn,
        metrics_fn=metrics_fn,
        do_eval=False,
        batch_size=batch_size,
        run_name=run_name,
        report_to="mlflow",
    )

    logger.debug("start test eval")
    test = cast(Dataset, data["test"])
    metrics = trainer.evaluate(test, metric_key_prefix="test")
    logger.debug(metrics)

    mlflow.end_run()
