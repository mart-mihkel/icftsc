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
    split: Split | None = None,
) -> tuple[DatasetDict, DatasetInfo]:
    if dataset == "multinerd":
        return load_multinerd(tokenizer, model_type, task, n_shot, split=split)

    if dataset == "estner":
        return load_estner(tokenizer, model_type, task, n_shot, split)

    if dataset == "boolq":
        return load_boolq(tokenizer, model_type, task, n_shot, split)

    if dataset == "wic":
        return load_wic(tokenizer, model_type, task, n_shot, split)

    raise NotImplementedError(f"Dataset '{dataset}'")


def load_tokenizer(model_path: str) -> PreTrainedTokenizerFast:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer = cast(PreTrainedTokenizerFast, tokenizer)

    if tokenizer.pad_token is None:
        logger.warning("tokenizer doesn't have a padding token, using eos")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


def get_collator(tokenizer: PreTrainedTokenizerFast, task: Task) -> DataCollator:
    if task == "seqcls":
        return DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    if task == "causal" or task == "seq2seq":
        return DataCollatorWithPaddingAndLabels(
            tokenizer=tokenizer,
            pad_to_multiple_of=8,
        )

    raise NotImplementedError(f"Task '{task}'")
