from typing import Literal, TypedDict, cast

from datasets.dataset_dict import DatasetDict
from datasets.load import load_dataset
from datasets.splits import Split
from transformers import BatchEncoding, PreTrainedTokenizerFast

from icftsc.constants import bert_model_types, gpt_model_types, t5_model_types
from icftsc.logging import logger
from icftsc.types import DatasetInfo, Task

type WiCLabel = Literal["no", "yes"]


class WiCExample(TypedDict):
    idx: int
    sentence1: str
    sentence2: str
    start1: int
    start2: int
    end1: int
    end2: int
    word: str
    label: int


id2label: dict[int, WiCLabel] = {0: "no", 1: "yes"}
label2id: dict[WiCLabel, int] = {"no": 0, "yes": 1}

examples = [
    (
        "Sentence 1: The bank closed at 5 PM.\n"
        "Sentence 2: They sat by the river bank.\n"
        "Word: bank\n"
        "Answer (yes/no): no\n"
    ),
    (
        "Sentence 1: I need to book a hotel room.\n"
        "Sentence 2: The book on the table is mine.\n"
        "Word: book\n"
        "Answer (yes/no): no\n"
    ),
    (
        "Sentence 1: The mouse is near the computer.\n"
        "Sentence 2: The mouse ran across the floor.\n"
        "Word: mouse\n"
        "Answer (yes/no): no\n"
    ),
    (
        "Sentence 1: He plays the guitar very well.\n"
        "Sentence 2: She works as a guitar instructor.\n"
        "Word: guitar\n"
        "Answer (yes/no): yes\n"
    ),
    (
        "Sentence 1: The temperature dropped significantly.\n"
        "Sentence 2: Please drop me a line when you can.\n"
        "Word: drop\n"
        "Answer (yes/no): no\n"
    ),
    (
        "Sentence 1: I like to read books.\n"
        "Sentence 2: The book club meets weekly.\n"
        "Word: book\n"
        "Answer (yes/no): yes\n"
    ),
    (
        "Sentence 1: Time to get up and face the day.\n"
        "Sentence 2: The face of the mountain was steep.\n"
        "Word: face\n"
        "Answer (yes/no): no\n"
    ),
    (
        "Sentence 1: She has a kind heart.\n"
        "Sentence 2: They were very kind to help.\n"
        "Word: kind\n"
        "Answer (yes/no): yes\n"
    ),
]


def _bert_sys_prompt(sep: str) -> str:
    return f"Does the word have the same meaning in both sentences?{sep}"


def _bert_prompt(example: WiCExample, sep: str) -> str:
    return f"{example['word']}{sep}{example['sentence1']}{sep}{example['sentence2']}"


def _gpt_sys_prompt() -> str:
    return "Question: Does the word have the same meaning in both sentences?\n"


def _gpt_prompt(example: WiCExample) -> str:
    return (
        f"Word: {example['word']}\n"
        f"Sentence 1: {example['sentence1']}\n"
        f"Sentence 2: {example['sentence2']}\n"
        "Answer (yes/no):"
    )


def _t5_sys_prompt() -> str:
    return (
        "word in context: "
        "determine if the word has the same meaning in both sentences.\n"
    )


def _t5_prompt(example: WiCExample) -> str:
    return (
        "word in context: does the word have the same meaning in both sentences.\n"
        f"word: {example['word']}\n"
        f"sentence 1: {example['sentence1']}\n"
        f"sentence 2: {example['sentence2']}\n"
        "answer (yes/no):"
    )


def _get_sys_prompt(
    tokenizer: PreTrainedTokenizerFast,
    model_type: str,
    n_shot: int,
) -> str:
    if n_shot > len(examples):
        raise ValueError("Requested more examples than exist")

    if model_type in bert_model_types:
        prompt = _bert_sys_prompt(sep=tokenizer.sep_token)
    elif model_type in gpt_model_types:
        prompt = _gpt_sys_prompt()
    elif model_type in t5_model_types:
        prompt = _t5_sys_prompt()
    else:
        raise NotImplementedError(f"Model type '{model_type}'")

    shots = "\n".join(examples[:n_shot])
    return f"{prompt}\n{shots}"


def _get_prompt(
    tokenizer: PreTrainedTokenizerFast,
    model_type: str,
    example: WiCExample,
) -> str:
    if model_type in bert_model_types:
        return _bert_prompt(example, tokenizer.sep_token)

    if model_type in gpt_model_types:
        return _gpt_prompt(example)

    if model_type in t5_model_types:
        return _t5_prompt(example)

    raise NotImplementedError(f"Model type '{model_type}'")


def _tokenize(
    example: WiCExample,
    tokenizer: PreTrainedTokenizerFast,
    model_type: str,
    task: Task,
    n_shot: int,
) -> BatchEncoding:
    _id2label = id2label | {-1: "private"}

    sys = _get_sys_prompt(tokenizer, model_type, n_shot)
    prompt = _get_prompt(tokenizer, model_type, example)
    label_id = example["label"]
    label = _id2label[label_id]

    prompt_enc = tokenizer(f"{sys}\n{prompt}", truncation=True)
    prompt_len = len(prompt_enc["input_ids"])

    if task == "seqcls":
        prompt_enc["label"] = label_id
        return prompt_enc

    answer = f"{sys}\n{prompt} {label}"
    answer_enc = tokenizer(answer, truncation=True)
    labels_enc = answer_enc["input_ids"].copy()

    if task == "causal":
        labels_enc[:prompt_len] = [-100] * prompt_len
        answer_enc["labels"] = labels_enc
        return answer_enc

    if task == "seq2seq":
        answer_enc["labels"] = labels_enc[(prompt_len - 1) :]
        return answer_enc

    raise NotImplementedError(f"Task '{task}'")


def load_wic(
    tokenizer: PreTrainedTokenizerFast,
    model_type: str,
    task: Task,
    n_shot: int,
    split: Split | None = None,
) -> tuple[DatasetDict, DatasetInfo]:
    data = cast(DatasetDict, load_dataset("super_glue", "wic", split=split))

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

    fn_kwargs = {
        "model_type": model_type,
        "tokenizer": tokenizer,
        "n_shot": n_shot,
        "task": task,
    }

    data = data.map(_tokenize, remove_columns=cols, fn_kwargs=fn_kwargs)

    if "train" in data:
        logger.info("%d train samples", len(data["train"]))

    if "dev" in data:
        logger.info("%d dev samples", len(data["dev"]))

    if "test" in data:
        logger.info("%d test samples", len(data["test"]))

    info = DatasetInfo(
        id2label=cast(dict[int, str], id2label),
        label2id=cast(dict[str, int], label2id),
        system_prompt=_get_sys_prompt(tokenizer, model_type, n_shot),
    )

    return data, info
