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
            _labels = feature["labels"]
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


def get_causal_batch(
    tokenizer: PreTrainedTokenizerFast,
    enc: BatchEncoding,
    id2label: dict[int, str],
) -> BatchEncoding:
    out_inputs = []
    out_attn = []
    out_labels = []

    it = zip(
        cast(Iterable, enc["input_ids"]),
        cast(Iterable, enc["attention_mask"]),
        cast(Iterable, enc["labels"]),
        strict=True,
    )

    for _input, _attn, _label in it:
        label_end = [*tokenizer.encode(id2label[_label]), tokenizer.eos_token_id]
        label_padding = [-100] * len(_input)

        full_input = _input + label_end
        full_attn = _attn + [0] * len(label_end)
        full_label = [*label_padding, *label_end]

        out_inputs.append(full_input)
        out_attn.append(full_attn)
        out_labels.append(full_label)

    out = {"input_ids": out_inputs, "attention_mask": out_attn, "labels": out_labels}
    return BatchEncoding(out)
