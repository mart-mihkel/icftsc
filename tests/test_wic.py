from typing import cast

import pytest
from datasets.splits import Split
from transformers import PreTrainedTokenizerFast

from icftsc.datasets.wic import load_wic

split = cast(
    Split,
    {
        "train": "train[:10]",
        "validation": "validation[:10]",
        "test": "test[:10]",
    },
)


def test_wic_seqcls(mmbert_tokenizer: PreTrainedTokenizerFast):
    data, _ = load_wic(
        tokenizer=mmbert_tokenizer,
        model_type="modernbert",
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


def test_wic_causal(gpt2_tokenizer: PreTrainedTokenizerFast):
    data, _ = load_wic(
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
    assert all(label == -100 for label in labels[: prompt_len - 1])


def test_wic_seq2seq(t5_tokenizer: PreTrainedTokenizerFast):
    data, _ = load_wic(
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


def test_wic_n_shot(mmbert_tokenizer: PreTrainedTokenizerFast):
    n_shot = 3
    _, info = load_wic(
        tokenizer=mmbert_tokenizer,
        model_type="modernbert",
        task="seqcls",
        split=split,
        n_shot=n_shot,
    )

    assert info["system_prompt"].count("Sentence 1:") == n_shot
    assert info["system_prompt"].count("Sentence 2:") == n_shot
    assert info["system_prompt"].count("Word:") == n_shot
    assert info["system_prompt"].count("Answer (yes/no):") == n_shot


def test_wic_invalid_model_type(mmbert_tokenizer: PreTrainedTokenizerFast):
    with pytest.raises(NotImplementedError, match="Model type 'invalid'"):
        load_wic(
            tokenizer=mmbert_tokenizer,
            model_type="invalid",
            task="seqcls",
            split=split,
            n_shot=0,
        )


def test_wic_invalid_n_shot(mmbert_tokenizer: PreTrainedTokenizerFast):
    with pytest.raises(ValueError, match="Requested more examples than exist"):
        load_wic(
            tokenizer=mmbert_tokenizer,
            model_type="modernbert",
            task="seqcls",
            split=split,
            n_shot=100,
        )
