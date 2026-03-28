from typing import cast

import pytest
from datasets.splits import Split
from transformers import PreTrainedTokenizerFast

from icftsc.datasets.estner import _join_spans, label2id, load_estner

split = cast(
    Split,
    {
        "train": "train[:10]",
        "dev": "dev[:10]",
        "test": "test[:10]",
    },
)


def test_join_spans():
    tokens = ["Kuulus", "kohver", "Eston", "Kohver"]
    tags = ["O", "O", "B-PER", "I-PER"]
    jtokens, jtags = _join_spans(tokens=tokens, tags=tags)

    assert jtokens == ["Kuulus", "kohver", "Eston Kohver"]
    assert jtags == ["O", "O", "PER"]


def test_estner_seqcls(mmbert_tokenizer: PreTrainedTokenizerFast):
    data, _ = load_estner(
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
    assert "labels" in train_sample
    assert "input_ids" in train_sample
    assert "attention_mask" in train_sample
    assert isinstance(train_sample["labels"], int)
    assert train_sample["labels"] in label2id.values()


def test_estner_causal(gpt2_tokenizer: PreTrainedTokenizerFast):
    data, _ = load_estner(
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
    first_non_masked = next((i for i, label in enumerate(labels) if label != -100), -1)
    assert first_non_masked > 0


def test_estner_seq2seq(t5_tokenizer: PreTrainedTokenizerFast):
    data, _ = load_estner(
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


def test_estner_n_shot(mmbert_tokenizer: PreTrainedTokenizerFast):
    n_shot = 3
    _, info = load_estner(
        tokenizer=mmbert_tokenizer,
        model_type="modernbert",
        task="seqcls",
        split=split,
        n_shot=n_shot,
    )

    assert info["system_prompt"].count("lause:") == n_shot
    assert info["system_prompt"].count("nimeüksus:") == n_shot
    assert info["system_prompt"].count("märgend:") == n_shot


def test_estner_invalid_model_type(mmbert_tokenizer: PreTrainedTokenizerFast):
    with pytest.raises(NotImplementedError, match="Model type 'invalid'"):
        load_estner(
            tokenizer=mmbert_tokenizer,
            model_type="invalid",
            task="seqcls",
            split=split,
            n_shot=0,
        )


def test_estner_invalid_n_shot(mmbert_tokenizer: PreTrainedTokenizerFast):
    with pytest.raises(ValueError, match="Requested more examples than exist"):
        load_estner(
            tokenizer=mmbert_tokenizer,
            model_type="modernbert",
            task="seqcls",
            split=split,
            n_shot=100,
        )
