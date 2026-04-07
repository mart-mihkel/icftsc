from dataclasses import dataclass
from typing import Any, cast

import torch
from datasets.dataset_dict import DatasetDict
from datasets.splits import Split
from torch import Tensor
from transformers import (
    AutoTokenizer,
    DataCollator,
    DataCollatorWithPadding,
    PreTrainedTokenizerFast,
)

from icftsc.datasets.boolq import load_boolq
from icftsc.datasets.estner import load_estner
from icftsc.datasets.multinerd import load_multinerd
from icftsc.datasets.wic import load_wic
from icftsc.logging import logger
from icftsc.types import DatasetInfo, DatasetName, Task


@dataclass
class DataCollatorWithPaddingAndLabels:
    tokenizer: PreTrainedTokenizerFast
    pad_to_multiple_of: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Tensor]:
        pad = self.tokenizer.pad_token_id
        mul = self.pad_to_multiple_of

        max_len = max(len(feature["input_ids"]) for feature in features)
        max_len = (max_len + mul - 1) // mul * mul

        labels = []
        inputs = []
        attn = []
        for feature in features:
            _labels = feature.get("labels", [])
            _inputs = feature["input_ids"]
            _attn = feature["attention_mask"]

            labels.append(_labels + [-100] * (max_len - len(_labels)))
            inputs.append(_inputs + [pad] * (max_len - len(_inputs)))
            attn.append(_attn + [0] * (max_len - len(_attn)))

            self.tokenizer.pad

        return {
            "labels": torch.tensor(labels),
            "input_ids": torch.tensor(inputs),
            "attention_mask": torch.tensor(attn),
        }


def load_data(
    tokenizer: PreTrainedTokenizerFast,
    dataset: DatasetName,
    model_type: str,
    task: Task,
    n_shot: int,
    n_train_samples: int | None = None,
    n_dev_samples: int | None = None,
    split: Split | None = None,
) -> tuple[DatasetDict, DatasetInfo]:
    logger.info("load '%s'", dataset)
    if dataset == "multinerd":
        data, info = load_multinerd(tokenizer, model_type, task, n_shot, split=split)
    elif dataset == "estner":
        data, info = load_estner(tokenizer, model_type, task, n_shot, split)
    elif dataset == "boolq":
        data, info = load_boolq(tokenizer, model_type, task, n_shot, split)
    elif dataset == "wic":
        data, info = load_wic(tokenizer, model_type, task, n_shot, split)
    else:
        raise NotImplementedError(f"Dataset '{dataset}'")

    if n_train_samples is not None:
        assert n_train_samples <= len(data["train"]), "requested too many train samples"
        logger.warning(
            "using %d of %d train samples",
            n_train_samples,
            len(data["train"]),
        )

        data["train"] = data["train"].select(range(n_train_samples))

    if n_dev_samples is not None:
        assert n_dev_samples <= len(data["dev"]), "requested too many dev samples"
        logger.warning(
            "using %d of %d dev samples",
            n_dev_samples,
            len(data["dev"]),
        )

        data["dev"] = data["dev"].select(range(n_dev_samples))

    return data, info


def load_tokenizer(model_path: str) -> PreTrainedTokenizerFast:
    logger.info("load pretrained tokenizer for '%s'", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer = cast(PreTrainedTokenizerFast, tokenizer)

    if tokenizer.pad_token is None:
        logger.warning("tokenizer doesn't have a padding token, using eos")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


def get_collator(tokenizer: PreTrainedTokenizerFast, task: Task) -> DataCollator:
    logger.debug("init data collator for %s", task)
    if task == "seqcls":
        return DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    if task == "causal" or task == "seq2seq":
        return DataCollatorWithPaddingAndLabels(
            tokenizer=tokenizer,
            pad_to_multiple_of=8,
        )

    raise NotImplementedError(f"Task '{task}'")
