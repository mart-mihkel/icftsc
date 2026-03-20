import json
from collections.abc import Callable
from math import ceil
from typing import Any, cast

import torch
from datasets.dataset_dict import DatasetDict
from datasets.splits import Split
from mlflow import end_run, set_experiment, set_tracking_uri, start_run
from torch.nn import Module, Parameter
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollator,
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from pt4sc.constants import bert_model_types, gpt_model_types, t5_model_types
from pt4sc.datasets.estner import init_estner
from pt4sc.datasets.multinerd import DatasetInfo, init_multinerd
from pt4sc.datasets.superglue import init_superglue
from pt4sc.logging import logger
from pt4sc.models import (
    PTDecoderModel,
    PTDecoderModelConfig,
    PTEncoderDecoderModel,
    PTEncoderDecoderModelConfig,
    PTEncoderModel,
    PTEncoderModelConfig,
    PTModel,
)
from pt4sc.types import DatasetName, PrefixInit


def save_params(params: dict[str, Any], run_name: str):
    import json
    import os

    os.makedirs(f"out/{run_name}", exist_ok=True)
    with open(f"out/{run_name}/cli_params.json", "w") as f:
        json.dump(params, f, indent=2)


def init_tokenizer(model_path: str) -> PreTrainedTokenizerFast:
    logger.debug("init tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer = cast(PreTrainedTokenizerFast, tokenizer)

    if tokenizer.pad_token is None:
        logger.warning("tokenizer doesn't have a padding token, using eos")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


def init_model(
    head_only: bool,
    tokenizer: PreTrainedTokenizerFast,
    model_path: str,
    data_info: DatasetInfo,
) -> tuple[PreTrainedModel, dict[str, set[str]]]:
    model, loading_info = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        output_loading_info=True,
        num_labels=len(data_info["id2label"]),
        id2label=data_info["id2label"],
        label2id=data_info["label2id"],
    )

    if model.config.pad_token_id is None:
        logger.warning("model doesn't have a padding token, using eos")
        model.config.pad_token_id = tokenizer.eos_token_id

    if head_only:
        freeze(model=model, skip=loading_info["missing_keys"])

    return model, loading_info


def load_pt_model(checkpoint: str) -> PTModel:
    logger.debug("load pt model from checkpoint")
    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    return AutoModel.from_pretrained(
        checkpoint,
        config=config,
        local_files_only=True,
    )


def init_pt_model(
    prefix_init: PrefixInit,
    tokenizer: PreTrainedTokenizerFast,
    model_path: str,
    data_info: DatasetInfo,
) -> PTModel:
    if "checkpoint" in model_path:
        return load_pt_model(checkpoint=model_path)

    bos = tokenizer.bos_token or ""
    sys = tokenizer(
        f"{bos}{data_info['system_prompt']}",
        add_special_tokens=False,
        truncation=True,
    )

    system_ids = torch.tensor(sys["input_ids"])
    num_virtual_tokens = len(system_ids)

    base, loading_info = init_model(
        head_only=False,
        tokenizer=tokenizer,
        model_path=model_path,
        data_info=data_info,
    )

    model_type = base.config.model_type
    if model_type in bert_model_types:
        logger.debug("init pt encoder model")
        config = PTEncoderModelConfig(
            pretrained_model=model_path,
            num_virtual_tokens=num_virtual_tokens,
            num_labels=len(data_info["id2label"]),
            id2label=data_info["id2label"],
            label2id=data_info["label2id"],
        )

        model = PTEncoderModel(config=config)
    elif model_type in gpt_model_types:
        logger.debug("init pt decoder model")
        config = PTDecoderModelConfig(
            pretrained_model=model_path,
            num_virtual_tokens=num_virtual_tokens,
            num_labels=len(data_info["id2label"]),
            id2label=data_info["id2label"],
            label2id=data_info["label2id"],
        )

        model = PTDecoderModel(config=config)
    elif model_type in t5_model_types:
        logger.debug("init pt encoder-decoder model")
        config = PTEncoderDecoderModelConfig(
            pretrained_model=model_path,
            num_virtual_tokens=num_virtual_tokens,
            num_labels=len(data_info["id2label"]),
            id2label=data_info["id2label"],
            label2id=data_info["label2id"],
        )

        model = PTEncoderDecoderModel(config=config)
    else:
        raise NotImplementedError(f"PT model for base '{model_type}'")

    if model.base.config.pad_token_id is None:
        model.base.config.pad_token_id = tokenizer.eos_token_id

    logger.debug("insert pretrained weights")
    model.base.load_state_dict(base.state_dict(), strict=False)
    del base

    freeze(model=model.base, skip=loading_info["missing_keys"])

    emb = model.base.get_input_embeddings()
    if prefix_init == "random":
        logger.debug("init random prefix")
        model.prefix = Parameter(torch.randn(num_virtual_tokens, emb.embedding_dim))
    elif prefix_init == "pretrained":
        logger.debug("init pretrained prefix")
        model.prefix = Parameter(emb(system_ids).detach())
    else:
        raise NotImplementedError(f"Prefix init '{prefix_init}'")

    return model


def init_data(
    tokenizer: PreTrainedTokenizerFast,
    dataset: DatasetName,
    model_type: str,
    workers: int,
    split: Split | None = None,
) -> tuple[DatasetDict, DatasetInfo]:
    if dataset == "multinerd":
        return init_multinerd(
            model_type=model_type,
            tokenizer=tokenizer,
            workers=workers,
            split=split,
        )

    if dataset == "estner":
        return init_estner(
            model_type=model_type,
            tokenizer=tokenizer,
            workers=workers,
            split=split,
        )

    if dataset == "superglue-boolq":
        return init_superglue(
            model_type=model_type,
            tokenizer=tokenizer,
            workers=workers,
            split=split,
        )

    raise NotImplementedError(f"Dataset '{dataset}'")


def freeze(model: Module, skip: set[str]):
    logger.info("freeze base model, skip layers: %s", skip)
    for name, param in model.named_parameters():
        if name not in skip:
            param.requires_grad = False


def train(
    model: Module,
    data: DatasetDict,
    collate_fn: DataCollator,
    metrics_fn: Callable[[EvalPrediction], dict[str, int | float]],
    run_name: str,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    effective_batch_size: int,
    grad_chkpts: bool,
    mlflow_tracking_uri: str | None,
):
    have_cuda = torch.cuda.is_available()
    optim = "adamw_8bit" if have_cuda else "adamw_torch_fused"
    grad_acc_steps = max(1, ceil(effective_batch_size / batch_size))
    actual_effective_batch_size = batch_size * grad_acc_steps
    train_steps = ceil(len(data["train"]) / actual_effective_batch_size) * epochs
    eval_steps = max(1, train_steps // 5)
    logging_steps = max(1, train_steps // 100)
    out_dir = f"out/{run_name}"

    logger.debug("%shave cuda", "" if have_cuda else "don't ")
    logger.debug(
        "batch size %d, effective batch size %d with %d gradient accumulation steps",
        batch_size,
        actual_effective_batch_size,
        grad_acc_steps,
    )

    logger.debug("%d train samples", len(data["train"]))
    logger.debug("%d dev samples", len(data["dev"]))
    if "test" in data:
        logger.debug("%d test samples", len(data["test"]))
    else:
        logger.debug("0 test samples")

    args = TrainingArguments(
        run_name=run_name,
        report_to="mlflow" if mlflow_tracking_uri else "none",
        output_dir=out_dir,
        save_strategy="no",
        eval_strategy="steps",
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        optim=optim,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 4,
        gradient_accumulation_steps=grad_acc_steps,
        gradient_checkpointing=grad_chkpts,
        bf16_full_eval=have_cuda,
        bf16=have_cuda,
        fp16_full_eval=not have_cuda,
        fp16=not have_cuda,
    )

    trainer = Trainer(
        args=args,
        model=model,
        train_dataset=data["train"],
        eval_dataset=data["dev"],
        data_collator=collate_fn,
        compute_metrics=metrics_fn,
    )

    if mlflow_tracking_uri is not None:
        logger.info(
            "tracking experiment 'pt4sc' run '%s' at %s",
            run_name,
            mlflow_tracking_uri,
        )

        set_tracking_uri(mlflow_tracking_uri)
        set_experiment("pt4sc")
        start_run(run_name=run_name)
    else:
        logger.warning("no experiment tracking configured")

    logger.info("start trainer")
    trainer.train()

    if "test" in data:
        test = cast(Dataset, data["test"])
        metrics = trainer.evaluate(test, metric_key_prefix="test")
        logger.info(json.dumps(metrics, indent=2))
    else:
        logger.warning("skip test evalatuaion, no data")

    if "test-system" in data:
        test_system = cast(Dataset, data["test-system"])
        metrics = trainer.evaluate(test_system, metric_key_prefix="test_system")
        logger.info(json.dumps(metrics, indent=2))
    else:
        logger.warning("skip system prompted test evalatuaion, no data")

    if "test-random" in data:
        test_random = cast(Dataset, data["test-random"])
        metrics = trainer.evaluate(test_random, metric_key_prefix="test_random")
        logger.info(json.dumps(metrics, indent=2))
    else:
        logger.warning("skip random prompt test evalatuaion, no data")

    logger.info("save checkpoint to %s", out_dir)
    trainer.save_model(out_dir)

    if mlflow_tracking_uri is not None:
        end_run()
