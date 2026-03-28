from typing import cast

import pytest
from datasets.splits import Split
from transformers import PreTrainedTokenizerFast

from icftsc.datasets.multinerd import _join_spans, load_multinerd

split = cast(
    Split,
    {
        "train": "train[:10]",
        "validation": "validation[:10]",
        "test": "test[:10]",
    },
)


def test_join_spans():
    tokens = ["Kuulus", "kohver", "Eston", "Kohver"]
    tag_ids = [0, 0, 1, 2]
    jtokens, jtags = _join_spans(tokens=tokens, tag_ids=tag_ids)

    assert jtokens == ["Kuulus", "kohver", "Eston Kohver"]
    assert jtags == [-1, -1, 0]


def test_multinerd_seqcls(mmbert_tokenizer: PreTrainedTokenizerFast):
    data, _ = load_multinerd(
        tokenizer=mmbert_tokenizer,
        model_type="modernbert",
        filter_en=False,
        task="seqcls",
        split=split,
        subset=1.0,
        n_shot=0,
    )

    assert len(data["train"]) > 0
    assert len(data["dev"]) > 0
    assert len(data["test"]) > 0

    train_sample = data["train"][0]
    assert "labels" in train_sample
    assert "input_ids" in train_sample
    assert "attention_mask" in train_sample
    assert isinstance(train_sample["labels"], int)


def test_multinerd_causal(gpt2_tokenizer: PreTrainedTokenizerFast):
    data, _ = load_multinerd(
        tokenizer=gpt2_tokenizer,
        model_type="gpt2",
        filter_en=False,
        task="causal",
        split=split,
        subset=1.0,
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
    first_non_masked = next((i for i, label in enumerate(labels) if label != -100), -1)
    assert first_non_masked > 0


def test_multinerd_seq2seq(t5_tokenizer: PreTrainedTokenizerFast):
    data, _ = load_multinerd(
        tokenizer=t5_tokenizer,
        model_type="t5",
        filter_en=False,
        task="seq2seq",
        split=split,
        subset=1.0,
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


def test_multinerd_n_shot(mmbert_tokenizer: PreTrainedTokenizerFast):
    n_shot = 3
    _, info = load_multinerd(
        tokenizer=mmbert_tokenizer,
        model_type="modernbert",
        filter_en=False,
        task="seqcls",
        split=split,
        subset=1.0,
        n_shot=n_shot,
    )

    assert info["system_prompt"].count("sentence:") == n_shot
    assert info["system_prompt"].count("entity:") == n_shot
    assert info["system_prompt"].count("tag:") == n_shot


def test_multinerd_invalid_model_type(mmbert_tokenizer: PreTrainedTokenizerFast):
    with pytest.raises(NotImplementedError, match="Model type 'invalid'"):
        load_multinerd(
            tokenizer=mmbert_tokenizer,
            model_type="invalid",
            filter_en=False,
            task="seqcls",
            split=split,
            subset=1.0,
            n_shot=0,
        )


def test_multinerd_invalid_n_shot(mmbert_tokenizer: PreTrainedTokenizerFast):
    with pytest.raises(ValueError, match="Requested more examples than exist"):
        load_multinerd(
            tokenizer=mmbert_tokenizer,
            model_type="modernbert",
            filter_en=False,
            task="seqcls",
            split=split,
            subset=1.0,
            n_shot=100,
        )
