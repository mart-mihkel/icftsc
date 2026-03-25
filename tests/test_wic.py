from typing import cast

from datasets.splits import Split
from transformers import PreTrainedTokenizerFast

from icftsc.datasets.wic import load_wic

split = cast(
    Split,
    {
        "train": "train[:100]",
        "validation": "validation[:100]",
        "test": "test[:100]",
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
