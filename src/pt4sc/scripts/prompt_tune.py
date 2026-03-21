from transformers import AutoConfig, DataCollatorWithPadding

from pt4sc.logging import logger
from pt4sc.metrics import compute_metrics_seq_cls
from pt4sc.scripts.common import init_data, init_pt_model, init_tokenizer, train
from pt4sc.types import DatasetName, PrefixInit, Task


def prompt_tune(
    model_path: str,
    run_name: str,
    dataset: DatasetName,
    task: Task,
    prefix_init: PrefixInit,
    workers: int,
    epochs: int,
    batch_size: int,
    effective_batch_size: int,
    learning_rate: float,
    grad_chkpts: bool,
    mlflow_tracking_uri: str | None,
):
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = init_tokenizer(model_path=model_path)
    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    logger.info("init dataset '%s'", dataset)
    data, info = init_data(
        model_type=config.model_type,
        tokenizer=tokenizer,
        dataset=dataset,
        workers=workers,
        task=task,
    )

    logger.info("drop prompted test datasets for prompt tuning")
    data.pop("test-system")
    data.pop("test-random")

    if dataset == "superglue-boolq":
        logger.warning("drop superglue test data, labels are private")
        data.pop("test")

    logger.info(
        "init pt model for '%s' with %s prefix initialization",
        model_path,
        prefix_init,
    )

    model = init_pt_model(
        prefix_init=prefix_init,
        tokenizer=tokenizer,
        model_path=model_path,
        data_info=info,
        task=task,
    )

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    prefix = model.prefix.numel()

    logger.info("total parameters %d", total)
    logger.info("trainable parameters %d", trainable)
    logger.info("head parameters %d", trainable - prefix)
    logger.info("prefix parameters %d", prefix)
    logger.info("prefix with %d virtual tokens", model.prefix.shape[0])

    train(
        model=model,
        data=data,
        collate_fn=collate_fn,
        metrics_fn=compute_metrics_seq_cls,
        run_name=run_name,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        effective_batch_size=effective_batch_size,
        grad_chkpts=grad_chkpts,
        mlflow_tracking_uri=mlflow_tracking_uri,
    )
