from typing import cast

from datasets.splits import Split
from transformers import PreTrainedTokenizerFast

from icftsc.datasets.boolq import init_boolq

split = cast(
    Split,
    {
        "train": "train[:100]",
        "validation": "validation[:100]",
        "test": "test[:100]",
    },
)


def test_boolq_seqcls(mmbert_tokenizer: PreTrainedTokenizerFast):
    data = init_boolq(
        tokenizer=mmbert_tokenizer,
        model_type="modernbert",
        task="seqcls",
        split=split,
    )

    assert len(data["train"]) > 0
    assert len(data["dev"]) > 0
    assert len(data["test"]) > 0


def test_boolq_causal(gpt2_tokenizer: PreTrainedTokenizerFast):
    data = init_boolq(
        tokenizer=gpt2_tokenizer,
        model_type="gpt2",
        task="causal",
        split=split,
    )

    assert len(data["train"]) > 0
    assert len(data["dev"]) > 0
    assert len(data["test"]) > 0


def test_boolq_seq2seq(t5_tokenizer: PreTrainedTokenizerFast):
    data = init_boolq(
        tokenizer=t5_tokenizer,
        model_type="t5",
        task="seq2seq",
        split=split,
    )

    assert len(data["train"]) > 0
    assert len(data["dev"]) > 0
    assert len(data["test"]) > 0
