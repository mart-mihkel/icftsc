from transformers import AutoConfig

from icftsc.datasets.common import prepend_system_tokens, randomize_prompt
from icftsc.logging import logger
from icftsc.scripts.common import (
    init_collator,
    init_data,
    init_metrics_fn,
    init_model,
    init_tokenizer,
    train,
)
from icftsc.types import DatasetName, PromptMode, Task


def few_shot(
    model_path: str,
    run_name: str,
    dataset: DatasetName,
    task: Task,
    prompt_mode: PromptMode,
    n_shot: int,
    workers: int,
    batch_size: int,
    mlflow_tracking_uri: str | None,
):
    logger.info("load model config")
    config = AutoConfig.from_pretrained(model_path)

    logger.info("load pretrained tokenizer")
    tokenizer = init_tokenizer(model_path=model_path)
    collate_fn = init_collator(tokenizer=tokenizer, task=task)
    metrics_fn = init_metrics_fn(task=task, tokenizer=tokenizer)

    logger.info("load dataset '%s'", dataset)
    data, info = init_data(
        model_type=config.model_type,
        tokenizer=tokenizer,
        dataset=dataset,
        workers=workers,
        n_shot=n_shot,
        task=task,
    )

    if dataset == "boolq" or dataset == "wic":
        logger.warning("using supergluq dev data, test labels are private")
        data["test"] = data["dev"]

    has_bos = tokenizer.bos_token is not None
    sys = tokenizer(info["system_prompt"], truncation=True, add_special_tokens=False)
    logger.info("prepare system prompt with %d tokens", len(sys["input_ids"]))
    if prompt_mode == "random":
        logger.info("randomize system prompt")
        sys = randomize_prompt(tokenizer=tokenizer, enc=sys)
    elif prompt_mode != "system":
        raise NotImplementedError(f"Prompt mode '{prompt_mode}'")

    data["test"] = data["test"].map(
        prepend_system_tokens,
        batched=True,
        num_proc=workers,
        fn_kwargs={"sys": sys, "has_bos": has_bos},
    )

    logger.info("load pretrained '%s' for '%s'", model_path, task)
    model, _ = init_model(
        model_path=model_path,
        tokenizer=tokenizer,
        head_only=False,
        data_info=info,
        task=task,
    )

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
