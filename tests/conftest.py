from typing import cast

from pytest import fixture
from transformers import AutoTokenizer, PreTrainedTokenizerFast


@fixture(scope="session")
def bert_tokenizer() -> PreTrainedTokenizerFast:
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-bert")
    return cast(PreTrainedTokenizerFast, tokenizer)


@fixture(scope="session")
def gpt2_tokenizer() -> PreTrainedTokenizerFast:
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
    tokenizer = cast(PreTrainedTokenizerFast, tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


@fixture(scope="session")
def t5_tokenizer() -> PreTrainedTokenizerFast:
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")
    return cast(PreTrainedTokenizerFast, tokenizer)
