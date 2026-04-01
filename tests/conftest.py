from typing import cast

from pytest import fixture
from transformers import AutoTokenizer, PreTrainedTokenizerFast


@fixture
def bert_tokenizer() -> PreTrainedTokenizerFast:
    return cast(
        PreTrainedTokenizerFast,
        AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-bert"),
    )


@fixture
def gpt2_tokenizer() -> PreTrainedTokenizerFast:
    tokenizer = cast(
        PreTrainedTokenizerFast,
        AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2"),
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


@fixture
def t5_tokenizer() -> PreTrainedTokenizerFast:
    return cast(
        PreTrainedTokenizerFast,
        AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5"),
    )
