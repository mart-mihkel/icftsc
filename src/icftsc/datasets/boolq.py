from typing import Literal, TypedDict, cast

from datasets.dataset_dict import DatasetDict
from datasets.load import load_dataset
from datasets.splits import Split
from transformers import BatchEncoding, PreTrainedTokenizerFast

from icftsc.logging import logger
from icftsc.types import Architecture, DatasetInfo

type BoolQALabel = Literal["no", "yes"]


class BoolqExample(TypedDict):
    idx: int
    passage: str
    question: str
    label: int


id2label: dict[int, BoolQALabel] = {0: "no", 1: "yes"}
label2id: dict[BoolQALabel, int] = {"no": 0, "yes": 1}

examples = [
    (
        "Passage: The sky appears blue during the day due to Rayleigh scattering.\n"
        "Question: Is the sky blue?\n"
        "Answer: yes\n"
    ),
    (
        "Passage: Fish are animals that live exclusively underwater and breathe "
        "using gills.\n"
        "Question: Can fish breathe on land?\n"
        "Answer: no\n"
    ),
    (
        "Passage: Water freezes at 0 degrees Celsius and boils at 100 degrees "
        "Celsius at sea level.\n"
        "Question: Does water freeze at room temperature?\n"
        "Answer: no\n"
    ),
    (
        "Passage: The Earth orbits around the Sun in approximately 365 days.\n"
        "Question: Does the Earth orbit the Sun?\n"
        "Answer: yes\n"
    ),
    (
        "Passage: Photosynthesis is the process by which plants convert sunlight "
        "into energy.\n"
        "Question: Do plants produce their own food?\n"
        "Answer: yes\n"
    ),
    (
        "Passage: The Great Wall of China is visible from space with naked eye.\n"
        "Question: Is the Great Wall visible from space?\n"
        "Answer: no\n"
    ),
    (
        "Passage: Lightning is a discharge of electricity that occurs during "
        "thunderstorms.\n"
        "Question: Is lightning caused by electricity?\n"
        "Answer: yes\n"
    ),
    (
        "Passage: The human body contains 206 bones in adulthood.\n"
        "Question: Do adults have more than 300 bones?\n"
        "Answer: no\n"
    ),
]


def _enc_sys_prompt(sep: str) -> str:
    return f"Answer the question based on the passage.{sep}"


def _enc_prompt(example: BoolqExample, sep: str) -> str:
    return f"{example['question']}{sep}{example['passage']}"


def _dec_sys_prompt() -> str:
    return (
        "Answer the question based on the passage. Do not provide any explanation, "
        'answer with exactly one word: "yes" or "no".\n'
    )


def _dec_prompt(example: BoolqExample) -> str:
    return (
        f"Passage: {example['passage']}\n"
        f"Question: {example['question']}\n"
        "Answer (yes/no):"
    )


def _encdec_sys_prompt() -> str:
    return (
        "boolean question answering: given a passage and a question, "
        'answer with "yes" or "no".\n'
    )


def _encdec_prompt(example: BoolqExample) -> str:
    return (
        f"passage: {example['passage']}\n"
        f"question: {example['question']}\n"
        "answer (yes/no):"
    )


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
    example: BoolqExample,
) -> str:
    if arch == "encoder":
        return _enc_prompt(example, tokenizer.sep_token)

    if arch == "decoder":
        return _dec_prompt(example)

    if arch == "encoder-decoder":
        return _encdec_prompt(example)

    raise NotImplementedError(f"architecture '{arch}'")


def _tokenize(
    example: BoolqExample,
    tokenizer: PreTrainedTokenizerFast,
    arch: Architecture,
    n_shot: int,
) -> BatchEncoding:
    _id2label = id2label | {-1: "private"}

    sys = _get_sys_prompt(tokenizer, arch, n_shot)
    prompt = _get_prompt(tokenizer, arch, example)
    label_id = example["label"]
    label = _id2label[label_id]

    prompt_enc = tokenizer(f"{sys}\n{prompt}", truncation=True)
    prompt_len = len(prompt_enc["input_ids"])

    if arch == "encoder":
        prompt_enc["label"] = label_id
        return prompt_enc

    answer = f"{sys}\n{prompt} {label}"
    answer_enc = tokenizer(answer, truncation=True)
    labels_enc = answer_enc["input_ids"].copy()

    if arch == "decoder":
        labels_enc[:prompt_len] = [-100] * prompt_len
        answer_enc["labels"] = labels_enc
        return answer_enc

    if arch == "encoder-decoder":
        idx = prompt_len - int(labels_enc[-1] == tokenizer.eos_token_id)
        answer_enc["labels"] = labels_enc[idx:]
        return answer_enc

    raise NotImplementedError(f"architecture '{arch}'")


def load_boolq(
    tokenizer: PreTrainedTokenizerFast,
    arch: Architecture,
    n_shot: int,
    split: Split | None = None,
) -> tuple[DatasetDict, DatasetInfo]:
    data = cast(DatasetDict, load_dataset("super_glue", "boolq", split=split))

    if "validation" in data:
        data["dev"] = data.pop("validation")

    cols = ["question", "passage", "label"]
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
