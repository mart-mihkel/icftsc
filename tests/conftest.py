from typing import cast

from datasets.dataset_dict import DatasetDict
from datasets.load import load_dataset
from datasets.splits import Split
from datasets.utils.info_utils import VerificationMode
from pytest import fixture
from transformers import AutoTokenizer, PreTrainedTokenizerFast

_split = {
    "train": "train[:10]",
    "validation": "validation[:10]",
    "test": "test[:10]",
}

_split_estner = {
    "train": "train[:10]",
    "dev": "dev[:10]",
    "test": "test[:10]",
}

_bert = "hf-internal-testing/tiny-random-bert"
_gpt2 = "hf-internal-testing/tiny-random-gpt2"
_t5 = "hf-internal-testing/tiny-random-t5"


@fixture(scope="session")
def bert_tokenizer() -> PreTrainedTokenizerFast:
    tokenizer = AutoTokenizer.from_pretrained(_bert)
    return cast(PreTrainedTokenizerFast, tokenizer)


@fixture(scope="session")
def gpt2_tokenizer() -> PreTrainedTokenizerFast:
    tokenizer = AutoTokenizer.from_pretrained(_gpt2)
    tokenizer = cast(PreTrainedTokenizerFast, tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


@fixture(scope="session")
def t5_tokenizer() -> PreTrainedTokenizerFast:
    tokenizer = AutoTokenizer.from_pretrained(_t5)
    return cast(PreTrainedTokenizerFast, tokenizer)


@fixture(scope="session")
def boolq() -> DatasetDict:
    split = cast(Split, _split)
    data = load_dataset("super_glue", "boolq", split=split)
    return cast(DatasetDict, data)


@fixture(scope="session")
def wic() -> DatasetDict:
    split = cast(Split, _split)
    data = load_dataset("super_glue", "wic", split=split)
    return cast(DatasetDict, data)


@fixture(scope="session")
def estner() -> DatasetDict:
    split = cast(Split, _split_estner)
    data = load_dataset("tartuNLP/EstNER", split=split)
    return cast(DatasetDict, data)


@fixture(scope="session")
def multinerd() -> DatasetDict:
    split = cast(Split, _split)
    data = load_dataset(
        "Babelscape/multinerd",
        verification_mode=VerificationMode.NO_CHECKS,
        split=split,
    )

    return cast(DatasetDict, data)
