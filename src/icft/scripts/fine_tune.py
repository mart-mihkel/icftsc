from transformers import AutoConfig, DataCollatorWithPadding

from icft.logging import logger
from icft.metrics import compute_metrics_seq_cls
from icft.scripts.common import (
    DatasetName,
    init_data,
    init_model,
    init_tokenizer,
    train,
)


def fine_tune(
    model_path: str,
    run_name: str,
    dataset: DatasetName,
    head_only: bool,
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
    )

    if dataset == "superglue-boolq":
        logger.warning("drop superglue test data, labels are private")
        data.pop("test")
        data.pop("test-system")
        data.pop("test-random")

    logger.info("init model '%s'", model_path)
    model, _ = init_model(
        head_only=head_only,
        tokenizer=tokenizer,
        model_path=model_path,
        data_info=info,
    )

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.debug("total parameters %d", total)
    logger.debug("trainable parameters %d", trainable)

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
