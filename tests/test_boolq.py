from unittest.mock import patch

import pytest
from datasets.dataset_dict import DatasetDict
from transformers import PreTrainedTokenizerFast

from icftsc.datasets.boolq import load_boolq


def test_boolq_seqcls(bert_tokenizer: PreTrainedTokenizerFast, boolq: DatasetDict):
    with patch("icftsc.datasets.boolq.load_dataset", return_value=boolq):
        data, _ = load_boolq(bert_tokenizer, "bert", "seqcls", 0)

    assert len(data["train"]) > 0
    assert len(data["dev"]) > 0
    assert len(data["test"]) > 0

    train_sample = data["train"][0]

    assert "label" in train_sample
    assert "input_ids" in train_sample
    assert "attention_mask" in train_sample
    assert train_sample["label"] in {0, 1}


def test_boolq_causal(gpt2_tokenizer: PreTrainedTokenizerFast, boolq: DatasetDict):
    with patch("icftsc.datasets.boolq.load_dataset", return_value=boolq):
        data, _ = load_boolq(gpt2_tokenizer, "gpt2", "causal", 0)

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


def test_boolq_seq2seq(t5_tokenizer: PreTrainedTokenizerFast, boolq: DatasetDict):
    with patch("icftsc.datasets.boolq.load_dataset", return_value=boolq):
        data, _ = load_boolq(t5_tokenizer, "t5", "seq2seq", 0)

    assert len(data["train"]) > 0
    assert len(data["dev"]) > 0
    assert len(data["test"]) > 0

    train_sample = data["train"][0]

    assert "labels" in train_sample
    assert "input_ids" in train_sample
    assert "attention_mask" in train_sample
    assert all(label >= 0 for label in train_sample["labels"])


def test_boolq_n_shot(bert_tokenizer: PreTrainedTokenizerFast, boolq: DatasetDict):
    n_shot = 3
    with patch("icftsc.datasets.boolq.load_dataset", return_value=boolq):
        _, info = load_boolq(bert_tokenizer, "bert", "seqcls", n_shot)

    assert info["system_prompt"].count("Passage:") == n_shot
    assert info["system_prompt"].count("Question:") == n_shot
    assert info["system_prompt"].count("Answer:") == n_shot


def test_boolq_invalid_model_type(
    bert_tokenizer: PreTrainedTokenizerFast,
    boolq: DatasetDict,
):
    with (
        pytest.raises(NotImplementedError, match="Model type 'invalid'"),
        patch("icftsc.datasets.boolq.load_dataset", return_value=boolq),
    ):
        load_boolq(bert_tokenizer, "invalid", "seqcls", 0)


def test_boolq_invalid_n_shot(
    bert_tokenizer: PreTrainedTokenizerFast,
    boolq: DatasetDict,
):
    with (
        pytest.raises(ValueError, match="Requested more examples than exist"),
        patch("icftsc.datasets.boolq.load_dataset", return_value=boolq),
    ):
        load_boolq(bert_tokenizer, "bert", "seqcls", 100)
