from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, TypedDict, cast

import numpy as np
import torch
from torch import Tensor
from transformers import BatchEncoding, PreTrainedTokenizerFast


class DatasetInfo(TypedDict):
    id2label: dict[int, str]
    label2id: dict[str, int]
    system_prompt: str


@dataclass
class DataCollatorWithPaddingAndLabels:
    tokenizer: PreTrainedTokenizerFast
    pad_to_multiple_of: int = 8

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Tensor]:
        labels = [feature.pop("labels") for feature in features if "labels" in feature]
        batch = self.tokenizer.pad(
            features,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        if not labels:
            return cast(dict, batch)

        max_label_length = max(len(label) for label in labels)
        max_label_length = (
            (max_label_length + self.pad_to_multiple_of - 1)
            // self.pad_to_multiple_of
            * self.pad_to_multiple_of
        )

        padded_labels = []
        for label in labels:
            remainder = max_label_length - len(label)
            padded = label + [-100] * remainder
            padded_labels.append(padded)

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)

        return cast(dict, batch)


def prepend_system_tokens(
    enc: BatchEncoding,
    sys: BatchEncoding,
    has_bos: bool,
) -> BatchEncoding:
    ids: list[list[int]] = []
    attn: list[list[int]] = []
    it = zip(
        cast(Iterable, enc["input_ids"]),
        cast(Iterable, enc["attention_mask"]),
        strict=True,
    )

    for _ids, _attn in it:
        if has_bos:
            _ids = _ids[1:]
            _attn = _attn[1:]

        ids.append(sys["input_ids"] + _ids)
        attn.append(sys["attention_mask"] + _attn)

    out = {"input_ids": ids, "attention_mask": attn, "labels": enc["labels"]}
    return BatchEncoding(out)


def randomize_prompt(
    tokenizer: PreTrainedTokenizerFast,
    enc: BatchEncoding,
) -> BatchEncoding:
    vocab_size = tokenizer.vocab_size
    special_ids = tokenizer.all_special_ids

    random_ids = []
    for token_id in cast(list[int], enc["input_ids"]):
        if token_id in special_ids:
            random_ids.append(token_id)
            continue

        rand_id = np.random.randint(0, vocab_size)
        while rand_id in special_ids:
            rand_id = np.random.randint(0, vocab_size)

        random_ids.append(rand_id)

    out = {"input_ids": random_ids, "attention_mask": enc["attention_mask"]}
    return BatchEncoding(out)
