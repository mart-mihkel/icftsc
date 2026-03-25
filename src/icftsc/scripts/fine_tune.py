from transformers import AutoConfig

from icftsc.datasets.common import get_collator, load_tokenizer
from icftsc.logging import logger
from icftsc.metrics import get_metrics_fn
from icftsc.scripts.common import get_model, load_data, train
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
    mlflow_tracking_uri: str | None,
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

    train(
        model=model,
        data=data,
        collate_fn=collate_fn,
        metrics_fn=metrics_fn,
        run_name=run_name,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        grad_chkpts=grad_chkpts,
        mlflow_tracking_uri=mlflow_tracking_uri,
    )
