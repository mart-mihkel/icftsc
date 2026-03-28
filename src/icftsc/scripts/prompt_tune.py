from typing import cast

import mlflow
from peft import PromptTuningConfig
from transformers import AutoConfig

from icftsc.datasets.util import get_collator, load_data, load_tokenizer
from icftsc.logging import logger
from icftsc.metrics import get_metrics_fn
from icftsc.modeling.util import get_pt_model, train
from icftsc.types import DatasetName, PrefixInit, Task


def prompt_tune(
    model_path: str,
    run_name: str,
    dataset: DatasetName,
    task: Task,
    prefix_init: PrefixInit,
    n_shot: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    grad_chkpts: bool,
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
        logger.warning("drop boolq test data, labels are private")
        data.pop("test")

    logger.info(
        "init pt-model from '%s' with %s prefix initialization for '%s'",
        model_path,
        prefix_init,
        task,
    )

    model = get_pt_model(prefix_init, tokenizer, model_path, task, info)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ptcfg = cast(PromptTuningConfig, model.peft_config["default"])

    logger.info("total parameters %d", total)
    logger.info("trainable parameters %d", trainable)
    logger.info("virtual tokens %d", ptcfg.num_virtual_tokens)
    logger.info("tracking '%s' of experiment '%s'", run_name, experiment)

    mlflow.set_experiment(experiment)
    mlflow.start_run(run_name=run_name)
    mlflow.log_params(
        {
            "task": task,
            "dataset": dataset,
            "prefix_init": prefix_init,
            "system_prompt": info["system_prompt"],
        }
    )

    mlflow.log_metrics(
        {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "num_virtual_tokens": ptcfg.num_virtual_tokens,
            "n_shot": n_shot,
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
