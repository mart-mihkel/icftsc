from unittest.mock import patch

import pytest
from datasets.dataset_dict import DatasetDict
from transformers import PreTrainedTokenizerFast

from icftsc.datasets.wic import load_wic


def test_wic_seqcls(bert_tokenizer: PreTrainedTokenizerFast, wic: DatasetDict):
    with patch("icftsc.datasets.wic.load_dataset", return_value=wic):
        data, _ = load_wic(bert_tokenizer, "bert", "seqcls", 0)

    assert len(data["train"]) > 0
    assert len(data["dev"]) > 0
    assert len(data["test"]) > 0

    train_sample = data["train"][0]

    assert "label" in train_sample
    assert "input_ids" in train_sample
    assert "attention_mask" in train_sample
    assert train_sample["label"] in {0, 1}


def test_wic_causal(gpt2_tokenizer: PreTrainedTokenizerFast, wic: DatasetDict):
    with patch("icftsc.datasets.wic.load_dataset", return_value=wic):
        data, _ = load_wic(gpt2_tokenizer, "gpt2", "causal", 0)

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


def test_wic_seq2seq(t5_tokenizer: PreTrainedTokenizerFast, wic: DatasetDict):
    with patch("icftsc.datasets.wic.load_dataset", return_value=wic):
        data, _ = load_wic(t5_tokenizer, "t5", "seq2seq", 0)

    assert len(data["train"]) > 0
    assert len(data["dev"]) > 0
    assert len(data["test"]) > 0

    train_sample = data["train"][0]

    assert "labels" in train_sample
    assert "input_ids" in train_sample
    assert "attention_mask" in train_sample
    assert all(label >= 0 for label in train_sample["labels"])


def test_wic_n_shot(bert_tokenizer: PreTrainedTokenizerFast, wic: DatasetDict):
    n_shot = 3
    with patch("icftsc.datasets.wic.load_dataset", return_value=wic):
        _, info = load_wic(bert_tokenizer, "bert", "seqcls", n_shot)

    assert info["system_prompt"].count("Sentence 1:") == n_shot
    assert info["system_prompt"].count("Sentence 2:") == n_shot
    assert info["system_prompt"].count("Word:") == n_shot
    assert info["system_prompt"].count("Answer (yes/no):") == n_shot


def test_wic_invalid_model_type(
    bert_tokenizer: PreTrainedTokenizerFast,
    wic: DatasetDict,
):
    with (
        pytest.raises(NotImplementedError, match="Model type 'invalid'"),
        patch("icftsc.datasets.wic.load_dataset", return_value=wic),
    ):
        load_wic(bert_tokenizer, "invalid", "seqcls", 0)


def test_wic_invalid_n_shot(
    bert_tokenizer: PreTrainedTokenizerFast,
    wic: DatasetDict,
):
    with (
        pytest.raises(AssertionError, match="requested more examples than exist"),
        patch("icftsc.datasets.wic.load_dataset", return_value=wic),
    ):
        load_wic(bert_tokenizer, "bert", "seqcls", 100)
