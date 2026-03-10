from typing import cast

from transformers import AutoTokenizer, PreTrainedTokenizerFast

from icft.datasets.estner import init_estner


def test_estner_mmbert():
    tokenizer = cast(
        PreTrainedTokenizerFast,
        AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base"),
    )

    data, _ = init_estner(
        tokenizer=tokenizer,
        task="seq-cls",
        prompt_mode="system",
        workers=0,
        split={"train": "train[:1]", "dev": "dev[:1]"},  # type: ignore
    )

    assert len(data["train"]) > 0


def test_estner_gpt2():
    tokenizer = cast(
        PreTrainedTokenizerFast,
        AutoTokenizer.from_pretrained("openai-community/gpt2"),
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    data, _ = init_estner(
        tokenizer=tokenizer,
        task="causal-lm",
        prompt_mode="system",
        workers=0,
        split={"train": "train[:1]", "dev": "dev[:1]"},  # type: ignore
    )

    assert len(data["train"]) > 0


def test_estner_t5():
    tokenizer = cast(
        PreTrainedTokenizerFast,
        AutoTokenizer.from_pretrained("google-t5/t5-small"),
    )

    data, _ = init_estner(
        tokenizer=tokenizer,
        task="seq2seq",
        prompt_mode="system",
        workers=0,
        split={"train": "train[:1]", "dev": "dev[:1]"},  # type: ignore
    )

    assert len(data["train"]) > 0
