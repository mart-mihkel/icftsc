from collections.abc import Callable
from typing import Any, cast

import torch
from datasets.dataset_dict import DatasetDict
from peft import PeftModel, PromptTuningConfig, TaskType, get_peft_model
from torch.nn import Module
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    DataCollator,
    DistilBertModel,
    EarlyStoppingCallback,
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizerFast,
    T5Gemma2Model,
    TrainerCallback,
)
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from icftsc.constants import (
    decoder_model_types,
    encoder_decoder_model_types,
    encoder_model_types,
)
from icftsc.datasets.util import DatasetInfo
from icftsc.logging import logger
from icftsc.types import Architecture, PrefixInit, Task


def get_arch(model_type: str) -> Architecture:
    if model_type in encoder_model_types:
        return "encoder"

    if model_type in decoder_model_types:
        return "decoder"

    if model_type in encoder_decoder_model_types:
        return "encoder-decoder"

    raise NotImplementedError(f"Model type '{model_type}'")


def get_model(
    tokenizer: PreTrainedTokenizerFast,
    model_path: str,
    data_info: DatasetInfo,
    task: Task,
    head_only: bool,
) -> PreTrainedModel:
    if task == "seqcls":
        logger.debug("load pretrained model for sequence classification")
        model, loading_info = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            output_loading_info=True,
            num_labels=len(data_info["id2label"]),
            id2label=data_info["id2label"],
            label2id=data_info["label2id"],
            device_map="auto",
        )
    elif task == "causal":
        logger.debug("load pretrained model for causal language modeling")
        loading_info = {"missing_keys": set()}
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    elif task == "seq2seq":
        logger.debug("load pretrained model for sequence to sequence")
        loading_info = {"missing_keys": set()}
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, device_map="auto")
    else:
        raise NotImplementedError(f"Task '{task}'")

    if model.config.pad_token_id is None:
        logger.warning("model doesn't have a padding token, using eos")
        model.config.pad_token_id = tokenizer.eos_token_id

    if head_only:
        freeze(model=model, skip=loading_info["missing_keys"])

    return model


def _distiblert_prompt_tuning_kwargs(model: DistilBertModel) -> dict[str, Any]:
    cfg = model.config
    return {
        "token_dim": cfg.dim,
        "num_layers": cfg.n_layers,
        "num_attention_heads": cfg.n_heads,
    }


def _t5gemma2_prompt_tuning_kwargs(model: T5Gemma2Model) -> dict[str, Any]:
    cfg = model.config
    return {
        "token_dim": cfg.encoder.text_config.hidden_size,
        "num_layers": cfg.encoder.text_config.num_hidden_layers,
        "num_attention_heads": cfg.encoder.text_config.num_attention_heads,
    }


def get_pt_model(
    prefix_init: PrefixInit,
    tokenizer: PreTrainedTokenizerFast,
    model_path: str,
    task: Task,
    data_info: DatasetInfo,
) -> PeftModel:
    sys_prompt = data_info["system_prompt"]
    sys_enc = tokenizer(sys_prompt, truncation=True)
    num_virtual_tokens = len(sys_enc["input_ids"])

    base = get_model(tokenizer, model_path, data_info, task, head_only=False)

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

    edge_case_kwargs = {}
    if "distilbert" in model_path:
        base = cast(DistilBertModel, base)
        edge_case_kwargs = _distiblert_prompt_tuning_kwargs(base)
    elif "t5gemma-2" in model_path:
        base = cast(T5Gemma2Model, base)
        edge_case_kwargs = _t5gemma2_prompt_tuning_kwargs(base)

    config = PromptTuningConfig(
        task_type=task_type,
        prompt_tuning_init=init,
        tokenizer_name_or_path=model_path,
        prompt_tuning_init_text=sys_prompt,
        num_virtual_tokens=num_virtual_tokens,
        **edge_case_kwargs,
    )

    logger.debug("init prompt tuning model for %s", model_path)
    return cast(PeftModel, get_peft_model(base, config))


def freeze(model: Module, skip: set[str] | None = None):
    if skip is None:
        skip = set()

    logger.info("freeze base model")
    for name, param in model.named_parameters():
        if name in skip:
            logger.info("skip '%s'", name)
            continue

        param.requires_grad = False


def get_args(
    do_eval: bool,
    epochs: int = 0,
    learning_rate: float = 5e-5,
    batch_size: int = 8,
    run_name: str = "default",
    report_to: str = "none",
) -> TrainingArguments:
    have_cuda = torch.cuda.is_available()
    optim = "adamw_8bit" if have_cuda else "adamw_torch_fused"
    eval_strategy = "epoch" if do_eval else "no"
    out_dir = f"out/{run_name}"

    logger.debug("optimizer '%s'", optim)
    logger.debug("checkpoint dir '%s'", out_dir)
    logger.debug("eval strategy '%s'", eval_strategy)

    return TrainingArguments(
        run_name=run_name,
        report_to=report_to,
        output_dir=out_dir,
        save_strategy="no",
        eval_strategy=eval_strategy,
        eval_on_start=do_eval,
        batch_eval_metrics=True,
        logging_steps=100,
        metric_for_best_model="f1",
        learning_rate=learning_rate,
        optim=optim,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        bf16_full_eval=have_cuda,
        bf16=have_cuda,
    )


def get_trainer(
    model: Module,
    data: DatasetDict,
    collate_fn: DataCollator,
    metrics_fn: Callable[[EvalPrediction, bool], dict[str, int | float]],
    do_eval: bool,
    epochs: int = 0,
    learning_rate: float = 5e-5,
    batch_size: int = 8,
    run_name: str = "default",
    report_to: str = "none",
) -> Trainer:
    args = get_args(do_eval, epochs, learning_rate, batch_size, run_name, report_to)
    _metrics_fn = cast(Callable, metrics_fn)

    callbacks = None
    if do_eval:
        stopper = EarlyStoppingCallback(2, 0.02)
        callbacks: list[TrainerCallback] = [stopper]

    return Trainer(
        args=args,
        model=model,
        callbacks=callbacks,
        data_collator=collate_fn,
        eval_dataset=data["dev"],
        train_dataset=data["train"],
        compute_metrics=_metrics_fn,
    )
