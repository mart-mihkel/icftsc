from unittest.mock import patch

import pytest
from datasets.dataset_dict import DatasetDict
from transformers import PreTrainedTokenizerFast

from icftsc.datasets.multinerd import _join_spans, load_multinerd


def test_join_spans():
    tokens = ["Kuulus", "kohver", "Eston", "Kohver"]
    tag_ids = [0, 0, 1, 2]
    jtokens, jtags = _join_spans(tokens=tokens, tag_ids=tag_ids)

    assert jtokens == ["Kuulus", "kohver", "Eston Kohver"]
    assert jtags == [-1, -1, 0]


def test_multinerd_seqcls(
    bert_tokenizer: PreTrainedTokenizerFast,
    multinerd: DatasetDict,
):
    with patch("icftsc.datasets.multinerd.load_dataset", return_value=multinerd):
        data, _ = load_multinerd(bert_tokenizer, "encoder", 0, False)

    assert len(data["train"]) > 0
    assert len(data["dev"]) > 0
    assert len(data["test"]) > 0

    train_sample = data["train"][0]

    assert "labels" in train_sample
    assert "input_ids" in train_sample
    assert "attention_mask" in train_sample
    assert isinstance(train_sample["labels"], int)


def test_multinerd_causal(
    gpt2_tokenizer: PreTrainedTokenizerFast,
    multinerd: DatasetDict,
):
    with patch("icftsc.datasets.multinerd.load_dataset", return_value=multinerd):
        data, _ = load_multinerd(gpt2_tokenizer, "decoder", 0, False)

    assert len(data["train"]) > 0
    assert len(data["dev"]) > 0
    assert len(data["test"]) > 0

    train_sample = data["train"][0]
    labels = train_sample["labels"]
    prompt_len = len(train_sample["input_ids"])

    assert "labels" in train_sample
    assert "input_ids" in train_sample
    assert "attention_mask" in train_sample
    assert len(labels) == prompt_len


def test_multinerd_seq2seq(
    t5_tokenizer: PreTrainedTokenizerFast,
    multinerd: DatasetDict,
):
    with patch("icftsc.datasets.multinerd.load_dataset", return_value=multinerd):
        data, _ = load_multinerd(t5_tokenizer, "encoder-decoder", 0, False)

    assert len(data["train"]) > 0
    assert len(data["dev"]) > 0
    assert len(data["test"]) > 0

    train_sample = data["train"][0]

    assert "labels" in train_sample
    assert "input_ids" in train_sample
    assert "attention_mask" in train_sample
    assert all(label >= 0 for label in train_sample["labels"])


def test_multinerd_n_shot(
    bert_tokenizer: PreTrainedTokenizerFast,
    multinerd: DatasetDict,
):
    n_shot = 3
    with patch("icftsc.datasets.multinerd.load_dataset", return_value=multinerd):
        _, info = load_multinerd(bert_tokenizer, "encoder", n_shot, False)

    assert info["system_prompt"].count("sentence:") == n_shot
    assert info["system_prompt"].count("entity:") == n_shot
    assert info["system_prompt"].count("tag:") == n_shot


def test_multinerd_invalid_n_shot(
    bert_tokenizer: PreTrainedTokenizerFast,
    multinerd: DatasetDict,
):
    with (
        pytest.raises(AssertionError, match="requested more examples than exist"),
        patch("datasets.load.load_dataset", return_value=multinerd),
    ):
        load_multinerd(bert_tokenizer, "encoder", 100, False)
