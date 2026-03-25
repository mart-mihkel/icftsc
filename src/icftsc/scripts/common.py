import json
from collections.abc import Callable
from math import ceil
from typing import cast

import torch
from datasets.dataset_dict import DatasetDict
from datasets.splits import Split
from mlflow import end_run, set_experiment, set_tracking_uri, start_run
from peft import PeftModel, PromptTuningConfig, TaskType, get_peft_model
from torch.nn import Module
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    DataCollator,
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from icftsc.datasets.boolq import init_boolq, init_boolq_info
from icftsc.datasets.estner import init_estner, init_estner_info
from icftsc.datasets.multinerd import DatasetInfo, init_multinerd, init_multinerd_info
from icftsc.datasets.wic import init_wic, init_wic_info
from icftsc.logging import logger
from icftsc.types import DatasetName, PrefixInit, Task


def init_model(
    head_only: bool,
    tokenizer: PreTrainedTokenizerFast,
    model_path: str,
    data_info: DatasetInfo,
    task: Task,
) -> tuple[PreTrainedModel, dict[str, set[str]]]:
    if task == "seqcls":
        logger.debug("load pretrained model for sequence classification")
        model, loading_info = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            output_loading_info=True,
            num_labels=len(data_info["id2label"]),
            id2label=data_info["id2label"],
            label2id=data_info["label2id"],
        )
    elif task == "causal":
        logger.debug("load pretrained model for causal language modeling")
        loading_info = {"missing_keys": set()}
        model = AutoModelForCausalLM.from_pretrained(model_path)
    elif task == "seq2seq":
        logger.debug("load pretrained model for sequence to sequence")
        loading_info = {"missing_keys": set()}
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    else:
        raise NotImplementedError(f"Task '{task}'")

    if model.config.pad_token_id is None:
        logger.warning("model doesn't have a padding token, using eos")
        model.config.pad_token_id = tokenizer.eos_token_id

    if head_only:
        freeze(model=model, skip=loading_info["missing_keys"])

    return model, loading_info


def init_pt_model(
    prefix_init: PrefixInit,
    tokenizer: PreTrainedTokenizerFast,
    model_path: str,
    task: Task,
    data_info: DatasetInfo,
) -> PeftModel:
    sys = tokenizer(data_info["system_prompt"], truncation=True)
    num_virtual_tokens = len(sys["input_ids"])

    base, _ = init_model(
        head_only=False,
        tokenizer=tokenizer,
        model_path=model_path,
        data_info=data_info,
        task=task,
    )

    if task == "seqcls":
        task_type = TaskType.SEQ_CLS
    elif task == "causal":
        task_type = TaskType.CAUSAL_LM
    elif task == "seq2seq":
        task_type = TaskType.SEQ_2_SEQ_LM
    else:
        raise NotImplementedError(f"Task '{task}'")

    if prefix_init == "pretrained":
        init = "TEXT"
    elif prefix_init == "random":
        init = "RANDOM"
    else:
        raise NotImplementedError(f"Prefix init '{prefix_init}'")

    config = PromptTuningConfig(
        task_type=task_type,
        prompt_tuning_init=init,
        tokenizer_name_or_path=model_path,
        num_virtual_tokens=num_virtual_tokens,
        prompt_tuning_init_text=data_info["system_prompt"],
    )

    return cast(PeftModel, get_peft_model(base, config))


def load_data(
    tokenizer: PreTrainedTokenizerFast,
    dataset: DatasetName,
    model_type: str,
    task: Task,
    workers: int = 0,
    split: Split | None = None,
    n_shot: int = 0,
) -> tuple[DatasetDict, DatasetInfo]:
    if dataset == "multinerd":
        info = init_multinerd_info(
            model_type=model_type,
            tokenizer=tokenizer,
            n_shot=n_shot,
        )

        data = init_multinerd(
            model_type=model_type,
            tokenizer=tokenizer,
            workers=workers,
            split=split,
            task=task,
        )
    elif dataset == "estner":
        info = init_estner_info(
            model_type=model_type,
            tokenizer=tokenizer,
            n_shot=n_shot,
        )

        data = init_estner(
            model_type=model_type,
            tokenizer=tokenizer,
            workers=workers,
            split=split,
            task=task,
        )
    elif dataset == "boolq":
        info = init_boolq_info(
            model_type=model_type,
            tokenizer=tokenizer,
            n_shot=n_shot,
        )

        data = init_boolq(
            model_type=model_type,
            tokenizer=tokenizer,
            workers=workers,
            split=split,
            task=task,
        )
    elif dataset == "wic":
        info = init_wic_info(
            model_type=model_type,
            tokenizer=tokenizer,
            n_shot=n_shot,
        )

        data = init_wic(
            model_type=model_type,
            tokenizer=tokenizer,
            workers=workers,
            split=split,
            task=task,
        )
    else:
        raise NotImplementedError(f"Dataset '{dataset}'")

    return data, info


def freeze(model: Module, skip: set[str]):
    logger.info("freeze base model")
    for name, param in model.named_parameters():
        if name in skip:
            logger.info("skip '%s'", name)
            continue

        param.requires_grad = False


def train(
    model: Module,
    data: DatasetDict,
    collate_fn: DataCollator,
    metrics_fn: Callable[[EvalPrediction, bool], dict[str, int | float]],
    run_name: str,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    grad_chkpts: bool,
    mlflow_tracking_uri: str | None,
):
    have_cuda = torch.cuda.is_available()
    optim = "adamw_8bit" if have_cuda else "adamw_torch_fused"

    train_steps = ceil(len(data["train"]) / batch_size) * epochs
    eval_steps = max(1, train_steps // 3)
    logging_steps = max(1, train_steps // 100)

    out_dir = f"out/{run_name}"
    report_to = "mlflow" if mlflow_tracking_uri else "none"

    logger.debug("using '%s' optimizer", optim)
    logger.debug("using '%s' output dir", out_dir)

    args = TrainingArguments(
        run_name=run_name,
        report_to=report_to,
        output_dir=out_dir,
        save_strategy="no",
        eval_strategy="steps",
        eval_steps=eval_steps,
        batch_eval_metrics=True,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        optim=optim,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_checkpointing=grad_chkpts,
        bf16_full_eval=have_cuda,
        bf16=have_cuda,
    )

    _metrics_fn = cast(Callable[[EvalPrediction], dict[str, int | float]], metrics_fn)
    trainer = Trainer(
        args=args,
        model=model,
        train_dataset=data["train"],
        eval_dataset=data["dev"],
        data_collator=collate_fn,
        compute_metrics=_metrics_fn,
    )

    if mlflow_tracking_uri is not None:
        logger.info(
            "tracking experiment 'icftsc' run '%s' at %s",
            run_name,
            mlflow_tracking_uri,
        )

        set_tracking_uri(mlflow_tracking_uri)
        set_experiment("icftsc")
        start_run(run_name=run_name)
    else:
        logger.warning("no experiment tracking configured")

    logger.info("start trainer")
    trainer.train()

    if "test" in data:
        logger.info("run test evaluation")
        test = cast(Dataset, data["test"])
        metrics = trainer.evaluate(test, metric_key_prefix="test")
        logger.info(json.dumps(metrics, indent=4))
    else:
        logger.warning("skip test evalatuaion")

    logger.info("save checkpoint to %s", out_dir)
    trainer.save_model(out_dir)

    if mlflow_tracking_uri is not None:
        end_run()
