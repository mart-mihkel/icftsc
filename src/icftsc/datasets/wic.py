from typing import Literal, TypedDict, cast

from datasets.dataset_dict import DatasetDict
from datasets.load import load_dataset
from datasets.splits import Split
from transformers import BatchEncoding, PreTrainedTokenizerFast

from icftsc.constants import bert_model_types, gpt_model_types, t5_model_types
from icftsc.datasets.common import DatasetInfo, get_causal_batch
from icftsc.logging import logger
from icftsc.types import Task

type WiCLabel = Literal["false", "true"]


class WiCBatch(TypedDict):
    idx: list[int]
    sentence1: list[str]
    sentence2: list[str]
    start1: list[int]
    start2: list[int]
    end1: list[int]
    end2: list[int]
    word: list[str]
    label: list[int]


id2label: dict[int, WiCLabel] = {
    0: "false",
    1: "true",
}

label2id: dict[WiCLabel, int] = {
    "false": 0,
    "true": 1,
}

examples = [
    (
        "Sentence 1: The bank closed at 5 PM.\n"
        "Sentence 2: They sat by the river bank.\n"
        "Word: bank\n"
        "Answer: false\n"
    ),
    (
        "Sentence 1: I need to book a hotel room.\n"
        "Sentence 2: The book on the table is mine.\n"
        "Word: book\n"
        "Answer: false\n"
    ),
    (
        "Sentence 1: The mouse is near the computer.\n"
        "Sentence 2: The mouse ran across the floor.\n"
        "Word: mouse\n"
        "Answer: false\n"
    ),
    (
        "Sentence 1: He plays the guitar very well.\n"
        "Sentence 2: She works as a guitar instructor.\n"
        "Word: guitar\n"
        "Answer: true\n"
    ),
    (
        "Sentence 1: The temperature dropped significantly.\n"
        "Sentence 2: Please drop me a line when you can.\n"
        "Word: drop\n"
        "Answer: false\n"
    ),
    (
        "Sentence 1: I like to read books.\n"
        "Sentence 2: The book club meets weekly.\n"
        "Word: book\n"
        "Answer: true\n"
    ),
    (
        "Sentence 1: Time to get up and face the day.\n"
        "Sentence 2: The face of the mountain was steep.\n"
        "Word: face\n"
        "Answer: false\n"
    ),
    (
        "Sentence 1: She has a kind heart.\n"
        "Sentence 2: They were very kind to help.\n"
        "Word: kind\n"
        "Answer: true\n"
    ),
]


def _bert_sys_prompt(bos: str, sep: str) -> str:
    return f"{bos}Determine if the word has the same meaning in both sentences.{sep}"


def _bert_prompt(
    word: str, sentence1: str, sentence2: str, bos: str, sep: str, eos: str
) -> str:
    return f"{bos}{word}{sep}{sentence1}{sep}{sentence2}{eos}"


def _gpt_sys_prompt(bos: str | None) -> str:
    _bos = bos if bos is not None else ""
    return (
        f"{_bos}"
        "You are a Word-in-Context expert. Given two sentences and a word, "
        "determine if the word has the same meaning in both sentences. "
        'Answer with exactly one word: "true" or "false".\n'
        "Do not provide any explanation.\n"
    )


def _gpt_prompt(word: str, sentence1: str, sentence2: str, bos: str | None) -> str:
    _bos = bos if bos is not None else ""
    return (
        f"{_bos}"
        f"Sentence 1: {sentence1}\n"
        f"Sentence 2: {sentence2}\n"
        f"Word: {word}\n"
        "Answer: "
    )


def _t5_sys_prompt() -> str:
    return (
        "word in context: given two sentences and a word, determine if the "
        'word has the same meaning in both sentences. answer with "true" or "false".\n'
    )


def _t5_prompt(word: str, sentence1: str, sentence2: str) -> str:
    return f"sentence1: {sentence1}\nsentence2: {sentence2}\nword: {word}\nanswer: "


def _get_sys_prompt(tokenizer: PreTrainedTokenizerFast, model_type: str) -> str:
    if model_type in bert_model_types:
        return _bert_sys_prompt(bos=tokenizer.bos_token, sep=tokenizer.sep_token)

    if model_type in gpt_model_types:
        return _gpt_sys_prompt(bos=tokenizer.bos_token)

    if model_type in t5_model_types:
        return _t5_sys_prompt()

    raise NotImplementedError(f"Model type '{model_type}'")


def _get_prompt(
    tokenizer: PreTrainedTokenizerFast,
    model_type: str,
    word: str,
    sentence1: str,
    sentence2: str,
) -> str:
    if model_type in bert_model_types:
        return _bert_prompt(
            word=word,
            sentence1=sentence1,
            sentence2=sentence2,
            bos=tokenizer.bos_token,
            sep=tokenizer.sep_token,
            eos=tokenizer.eos_token,
        )

    if model_type in gpt_model_types:
        return _gpt_prompt(
            word=word,
            sentence1=sentence1,
            sentence2=sentence2,
            bos=tokenizer.bos_token,
        )

    if model_type in t5_model_types:
        return _t5_prompt(word=word, sentence1=sentence1, sentence2=sentence2)

    raise NotImplementedError(f"Model type '{model_type}'")


def _tokenize(
    batch: WiCBatch,
    tokenizer: PreTrainedTokenizerFast,
    model_type: str,
    task: Task,
) -> BatchEncoding:
    prompts: list[str] = []
    labels: list[int] = []

    it = zip(
        batch["word"],
        batch["sentence1"],
        batch["sentence2"],
        batch["label"],
        strict=True,
    )
    for word, sentence1, sentence2, label_id in it:
        prompt = _get_prompt(
            model_type=model_type,
            tokenizer=tokenizer,
            word=word,
            sentence1=sentence1,
            sentence2=sentence2,
        )

        prompts.append(prompt)
        labels.append(label_id)

    enc = tokenizer(prompts, truncation=True, add_special_tokens=False)
    enc["labels"] = labels

    if task == "seqcls":
        return enc

    if task == "causal":
        _id2label = cast(dict[int, str], id2label)
        _id2label[-1] = "private"
        return get_causal_batch(tokenizer=tokenizer, enc=enc, id2label=_id2label)

    if task == "seq2seq":
        _id2label = cast(dict[int, str], id2label)
        _id2label[-1] = "private"
        _eos = tokenizer.eos_token_id
        _labels = [[*tokenizer.encode(id2label[label_id]), _eos] for label_id in labels]
        enc["labels"] = _labels
        return enc

    raise NotImplementedError(f"Task '{task}'")


def init_wic(
    tokenizer: PreTrainedTokenizerFast,
    model_type: str,
    task: Task,
    workers: int = 0,
    split: Split | None = None,
) -> DatasetDict:
    data = cast(DatasetDict, load_dataset("aps/super_glue", "wic", split=split))

    if "validation" in data:
        data["dev"] = data.pop("validation")

    cols = [
        "idx",
        "sentence1",
        "sentence2",
        "start1",
        "start2",
        "end1",
        "end2",
        "word",
        "label",
    ]

    fn_kwargs = {"tokenizer": tokenizer, "model_type": model_type, "task": task}
    data = data.map(
        _tokenize,
        batched=True,
        num_proc=workers,
        remove_columns=cols,
        fn_kwargs=fn_kwargs,
    )

    if "train" in data:
        logger.info("%d train samples", len(data["train"]))

    if "dev" in data:
        logger.info("%d dev samples", len(data["dev"]))

    if "test" in data:
        logger.info("%d test samples", len(data["test"]))

    return data


def init_wic_info(
    tokenizer: PreTrainedTokenizerFast,
    model_type: str,
    n_shot: int = 0,
) -> DatasetInfo:
    if n_shot > len(examples):
        raise ValueError("Requested more examples than exist")

    prompt = _get_sys_prompt(tokenizer=tokenizer, model_type=model_type)

    shots = "".join(examples[:n_shot])
    prompt = f"{prompt}{shots}"

    return DatasetInfo(
        id2label=cast(dict[int, str], id2label),
        label2id=cast(dict[str, int], label2id),
        system_prompt=prompt,
    )
