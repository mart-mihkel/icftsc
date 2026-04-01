from typing import cast

import pytest
from datasets.splits import Split
from transformers import PreTrainedTokenizerFast

from icftsc.datasets.boolq import load_boolq

split = cast(
    Split,
    {
        "train": "train[:10]",
        "validation": "validation[:10]",
        "test": "test[:10]",
    },
)


def test_boolq_seqcls(bert_tokenizer: PreTrainedTokenizerFast):
    data, _ = load_boolq(
        tokenizer=bert_tokenizer,
        model_type="bert",
        task="seqcls",
        split=split,
        n_shot=0,
    )

    assert len(data["train"]) > 0
    assert len(data["dev"]) > 0
    assert len(data["test"]) > 0

    train_sample = data["train"][0]
    assert "label" in train_sample
    assert "input_ids" in train_sample
    assert "attention_mask" in train_sample
    assert train_sample["label"] in {0, 1}


def test_boolq_causal(gpt2_tokenizer: PreTrainedTokenizerFast):
    data, _ = load_boolq(
        tokenizer=gpt2_tokenizer,
        model_type="gpt2",
        task="causal",
        split=split,
        n_shot=0,
    )

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


def test_boolq_seq2seq(t5_tokenizer: PreTrainedTokenizerFast):
    data, _ = load_boolq(
        tokenizer=t5_tokenizer,
        model_type="t5",
        task="seq2seq",
        split=split,
        n_shot=0,
    )

    assert len(data["train"]) > 0
    assert len(data["dev"]) > 0
    assert len(data["test"]) > 0

    train_sample = data["train"][0]
    assert "labels" in train_sample
    assert "input_ids" in train_sample
    assert "attention_mask" in train_sample
    assert all(label >= 0 for label in train_sample["labels"])


def test_boolq_n_shot(bert_tokenizer: PreTrainedTokenizerFast):
    n_shot = 3
    _, info = load_boolq(
        tokenizer=bert_tokenizer,
        model_type="bert",
        task="seqcls",
        split=split,
        n_shot=n_shot,
    )

    assert info["system_prompt"].count("Passage:") == n_shot
    assert info["system_prompt"].count("Question:") == n_shot
    assert info["system_prompt"].count("Answer:") == n_shot


def test_boolq_invalid_model_type(bert_tokenizer: PreTrainedTokenizerFast):
    with pytest.raises(NotImplementedError, match="Model type 'invalid'"):
        load_boolq(
            tokenizer=bert_tokenizer,
            model_type="invalid",
            task="seqcls",
            split=split,
            n_shot=0,
        )


def test_boolq_invalid_n_shot(bert_tokenizer: PreTrainedTokenizerFast):
    with pytest.raises(ValueError, match="Requested more examples than exist"):
        load_boolq(
            tokenizer=bert_tokenizer,
            model_type="bert",
            task="seqcls",
            split=split,
            n_shot=100,
        )
