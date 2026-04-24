from textwrap import dedent
from typing import Literal, TypedDict, cast

from datasets.dataset_dict import DatasetDict
from datasets.load import load_dataset
from datasets.splits import Split
from transformers import BatchEncoding, PreTrainedTokenizerFast

from icftsc.logging import logger
from icftsc.types import Architecture, DatasetInfo

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


def _enc_sys_prompt(sep: str) -> str:
    return f"Does the word have the same meaning in both sentences?{sep}"


def _enc_prompt(example: WiCExample, sep: str) -> str:
    return f"{example['word']}{sep}{example['sentence1']}{sep}{example['sentence2']}"


def _dec_sys_prompt() -> str:
    return dedent("""
        Determine if the word has the same meaning in both sentences.
        Do not provide any explanation.
        Answer with only yes or no
    """).strip()


def _dec_prompt(example: WiCExample) -> str:
    return dedent(f"""
        Sentence 1: {example["sentence1"]}
        Sentence 2: {example["sentence2"]}
        Word: {example["word"]}
        Answer (yes/no):
    """).strip()


def _encdec_sys_prompt() -> str:
    return dedent("""
        word in context: does the word have the same meaning in both sentences
        output only yes or no
    """).strip()


def _encdec_prompt(example: WiCExample) -> str:
    return dedent(f"""
        sentence 1: {example["sentence1"]}
        sentence 2: {example["sentence2"]}
        word: {example["word"]}
        answer (yes/no):
    """).strip()


def _get_sys_prompt(
    tokenizer: PreTrainedTokenizerFast,
    arch: Architecture,
    n_shot: int,
) -> str:
    assert n_shot <= len(examples), "requested more examples than exist"

    if arch == "encoder":
        prompt = _enc_sys_prompt(sep=tokenizer.sep_token)
    elif arch == "decoder":
        prompt = _dec_sys_prompt()
    elif arch == "encoder-decoder":
        prompt = _encdec_sys_prompt()
    else:
        raise NotImplementedError(f"architecture '{arch}'")

    shots = "\n".join(examples[:n_shot])
    return f"{prompt}\n{shots}"


def _get_prompt(
    tokenizer: PreTrainedTokenizerFast,
    arch: Architecture,
    example: WiCExample,
) -> str:
    if arch == "encoder":
        return _enc_prompt(example, tokenizer.sep_token)

    if arch == "decoder":
        return _dec_prompt(example)

    if arch == "encoder-decoder":
        return _encdec_prompt(example)

    raise NotImplementedError(f"architecture '{arch}'")


def _tokenize(
    example: WiCExample,
    tokenizer: PreTrainedTokenizerFast,
    arch: Architecture,
    n_shot: int,
) -> BatchEncoding:
    _id2label = id2label | {-1: "private"}

    sys = _get_sys_prompt(tokenizer, arch, n_shot)
    prompt = _get_prompt(tokenizer, arch, example)
    label_id = example["label"]
    label = _id2label[label_id]

    if tokenizer.chat_template is not None:
        conv = [
            {"role": "system", "content": sys},
            {"role": "user", "content": prompt},
        ]

        prompt_enc = tokenizer.apply_chat_template(conv, truncation=True)
    else:
        prompt_enc = tokenizer(f"{sys}\n{prompt}", truncation=True)

    prompt_enc = cast(BatchEncoding, prompt_enc)
    prompt_len = len(cast(list[int], prompt_enc["input_ids"]))

    if arch == "encoder":
        prompt_enc["label"] = label_id
        return prompt_enc

    if tokenizer.chat_template is not None:
        conv = [
            {"role": "system", "content": sys},
            {"role": "user", "content": f"{prompt} {label}"},
        ]

        answer_enc = tokenizer.apply_chat_template(conv, truncation=True)
    else:
        answer = f"{sys}\n{prompt} {label}"
        answer_enc = tokenizer(answer, truncation=True)

    answer_enc = cast(BatchEncoding, answer_enc)
    labels_enc = cast(list[int], answer_enc["input_ids"]).copy()

    if arch == "decoder":
        labels_enc[:prompt_len] = [-100] * prompt_len
        answer_enc["labels"] = labels_enc
        return answer_enc

    if arch == "encoder-decoder":
        idx = prompt_len - int(labels_enc[-1] == tokenizer.eos_token_id)
        answer_enc["labels"] = labels_enc[idx:]
        return answer_enc

    raise NotImplementedError(f"architecture '{arch}'")


def load_wic(
    tokenizer: PreTrainedTokenizerFast,
    arch: Architecture,
    n_shot: int,
    split: Split | None = None,
) -> tuple[DatasetDict, DatasetInfo]:
    data = cast(DatasetDict, load_dataset("super_glue", "wic", split=split))

    if "validation" in data:
        logger.debug("rename 'validation' to 'dev'")
        data["dev"] = data.pop("validation")

    logger.debug("tokenize wic")
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

    fn_kwargs = {"tokenizer": tokenizer, "n_shot": n_shot, "arch": arch}
    data = data.map(_tokenize, remove_columns=cols, fn_kwargs=fn_kwargs)
    for subsplit in data:
        logger.debug("tokenized %d %s samples", len(data[subsplit]), subsplit)

    info = DatasetInfo(
        id2label=cast(dict[int, str], id2label),
        label2id=cast(dict[str, int], label2id),
        system_prompt=_get_sys_prompt(tokenizer, arch, n_shot),
    )

    return data, info
