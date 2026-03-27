from transformers import AutoConfig, AutoModel

from icftsc.datasets.util import get_collator, load_data, load_tokenizer
from icftsc.logging import logger
from icftsc.metrics import get_metrics_fn
from icftsc.modeling.util import train
from icftsc.types import DatasetName, Task


def few_shot(
    model_path: str,
    run_name: str,
    dataset: DatasetName,
    task: Task,
    n_shot: int,
    batch_size: int,
    mlflow_tracking_uri: str | None,
):
    logger.info("load model config")
    config = AutoConfig.from_pretrained(model_path)

    logger.info("load pretrained tokenizer")
    tokenizer = load_tokenizer(model_path)
    collate_fn = get_collator(tokenizer, task)
    metrics_fn = get_metrics_fn(tokenizer, task)

    logger.info("load dataset '%s'", dataset)
    data, _ = load_data(
        model_type=config.model_type,
        tokenizer=tokenizer,
        dataset=dataset,
        n_shot=n_shot,
        task=task,
    )

    if dataset == "boolq" or dataset == "wic":
        logger.warning("using supergluq dev data, test labels are private")
        data["test"] = data["dev"]

    logger.info("load pretrained '%s' for '%s'", model_path, task)
    model = AutoModel.from_pretrained(model_path)

    train(
        model=model,
        data=data,
        collate_fn=collate_fn,
        metrics_fn=metrics_fn,
        run_name=run_name,
        epochs=0,
        learning_rate=5e-5,
        batch_size=batch_size,
        grad_chkpts=False,
        mlflow_tracking_uri=mlflow_tracking_uri,
    )
