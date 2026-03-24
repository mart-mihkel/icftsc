from typing import cast

from datasets.splits import Split
from transformers import PreTrainedTokenizerFast

from icftsc.datasets.estner import _join_spans, init_estner

split = cast(
    Split,
    {
        "train": "train[:100]",
        "dev": "dev[:100]",
        "test": "test[:100]",
    },
)


def test_join_spans():
    tokens = ["Kuulus", "kohver", "Eston", "Kohver"]
    tags = ["O", "O", "B-PER", "I-PER"]
    jtokens, jtags = _join_spans(tokens=tokens, tags=tags)

    assert jtokens == ["Kuulus", "kohver", "Eston Kohver"]
    assert jtags == ["O", "O", "PER"]


def test_estner_mmbert(mmbert_tokenizer: PreTrainedTokenizerFast):
    data = init_estner(
        tokenizer=mmbert_tokenizer,
        model_type="modernbert",
        task="seqcls",
        split=split,
    )

    assert len(data["train"]) > 0
    assert len(data["dev"]) > 0
    assert len(data["test"]) > 0


def test_estner_gpt2(gpt2_tokenizer: PreTrainedTokenizerFast):
    data = init_estner(
        tokenizer=gpt2_tokenizer,
        model_type="gpt2",
        task="causal",
        split=split,
    )

    assert len(data["train"]) > 0
    assert len(data["dev"]) > 0
    assert len(data["test"]) > 0


def test_estner_t5(t5_tokenizer: PreTrainedTokenizerFast):
    data = init_estner(
        tokenizer=t5_tokenizer,
        model_type="t5",
        task="seq2seq",
        split=split,
    )

    assert len(data["train"]) > 0
    assert len(data["dev"]) > 0
    assert len(data["test"]) > 0
