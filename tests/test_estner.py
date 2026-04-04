from unittest.mock import patch

import pytest
from datasets.dataset_dict import DatasetDict
from transformers import PreTrainedTokenizerFast

from icftsc.datasets.estner import _join_spans, label2id, load_estner


def test_join_spans():
    tokens = ["Kuulus", "kohver", "Eston", "Kohver"]
    tags = ["O", "O", "B-PER", "I-PER"]
    jtokens, jtags = _join_spans(tokens=tokens, tags=tags)

    assert jtokens == ["Kuulus", "kohver", "Eston Kohver"]
    assert jtags == ["O", "O", "PER"]


def test_estner_seqcls(bert_tokenizer: PreTrainedTokenizerFast, estner: DatasetDict):
    with patch("icftsc.datasets.estner.load_dataset", return_value=estner):
        data, _ = load_estner(bert_tokenizer, "bert", "seqcls", 0)

    assert len(data["train"]) > 0
    assert len(data["dev"]) > 0
    assert len(data["test"]) > 0

    train_sample = data["train"][0]

    assert "labels" in train_sample
    assert "input_ids" in train_sample
    assert "attention_mask" in train_sample
    assert isinstance(train_sample["labels"], int)
    assert train_sample["labels"] in label2id.values()


def test_estner_causal(gpt2_tokenizer: PreTrainedTokenizerFast, estner: DatasetDict):
    with patch("icftsc.datasets.estner.load_dataset", return_value=estner):
        data, _ = load_estner(gpt2_tokenizer, "gpt2", "causal", 0)

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

    first_non_masked = next((i for i, label in enumerate(labels) if label != -100), -1)

    assert first_non_masked > 0


def test_estner_seq2seq(t5_tokenizer: PreTrainedTokenizerFast, estner: DatasetDict):
    with patch("icftsc.datasets.estner.load_dataset", return_value=estner):
        data, _ = load_estner(t5_tokenizer, "t5", "seq2seq", 0)

    assert len(data["train"]) > 0
    assert len(data["dev"]) > 0
    assert len(data["test"]) > 0

    train_sample = data["train"][0]

    assert "labels" in train_sample
    assert "input_ids" in train_sample
    assert "attention_mask" in train_sample
    assert all(label >= 0 for label in train_sample["labels"])


def test_estner_n_shot(bert_tokenizer: PreTrainedTokenizerFast, estner: DatasetDict):
    n_shot = 3
    with patch("icftsc.datasets.estner.load_dataset", return_value=estner):
        _, info = load_estner(bert_tokenizer, "bert", "seqcls", n_shot)

    assert info["system_prompt"].count("lause:") == n_shot
    assert info["system_prompt"].count("nimeüksus:") == n_shot
    assert info["system_prompt"].count("märgend:") == n_shot


def test_estner_invalid_model_type(
    bert_tokenizer: PreTrainedTokenizerFast,
    estner: DatasetDict,
):
    with (
        pytest.raises(NotImplementedError, match="Model type 'invalid'"),
        patch("icftsc.datasets.estner.load_dataset", return_value=estner),
    ):
        load_estner(bert_tokenizer, "invalid", "seqcls", 0)


def test_estner_invalid_n_shot(
    bert_tokenizer: PreTrainedTokenizerFast,
    estner: DatasetDict,
):
    with (
        pytest.raises(ValueError, match="Requested more examples than exist"),
        patch("icftsc.datasets.estner.load_dataset", return_value=estner),
    ):
        load_estner(bert_tokenizer, "bert", "seqcls", 100)
