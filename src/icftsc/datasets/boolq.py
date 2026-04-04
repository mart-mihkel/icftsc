from typing import Literal, TypedDict, cast

from datasets.dataset_dict import DatasetDict
from datasets.load import load_dataset
from datasets.splits import Split
from transformers import BatchEncoding, PreTrainedTokenizerFast

from icftsc.constants import (
    decoder_model_types,
    encoder_decoder_model_types,
    encoder_model_types,
)
from icftsc.logging import logger
from icftsc.types import DatasetInfo, Task

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
    model_type: str,
    n_shot: int,
) -> str:
    assert n_shot <= len(examples), "requested more examples than exist"

    if model_type in encoder_model_types:
        prompt = _enc_sys_prompt(sep=tokenizer.sep_token)
    elif model_type in decoder_model_types:
        prompt = _dec_sys_prompt()
    elif model_type in encoder_decoder_model_types:
        prompt = _encdec_sys_prompt()
    else:
        raise NotImplementedError(f"Model type '{model_type}'")

    shots = "\n".join(examples[:n_shot])
    return f"{prompt}\n{shots}"


def _get_prompt(
    tokenizer: PreTrainedTokenizerFast,
    model_type: str,
    example: BoolqExample,
) -> str:
    if model_type in encoder_model_types:
        return _enc_prompt(example, tokenizer.sep_token)

    if model_type in decoder_model_types:
        return _dec_prompt(example)

    if model_type in encoder_decoder_model_types:
        return _encdec_prompt(example)

    raise NotImplementedError(f"Model type '{model_type}'")


def _tokenize(
    example: BoolqExample,
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
        idx = prompt_len - int(labels_enc[-1] == tokenizer.eos_token_id)
        answer_enc["labels"] = labels_enc[idx:]
        return answer_enc

    raise NotImplementedError(f"Task '{task}'")


def load_boolq(
    tokenizer: PreTrainedTokenizerFast,
    model_type: str,
    task: Task,
    n_shot: int,
    split: Split | None = None,
) -> tuple[DatasetDict, DatasetInfo]:
    data = cast(DatasetDict, load_dataset("super_glue", "boolq", split=split))

    if "validation" in data:
        data["dev"] = data.pop("validation")

    cols = ["question", "passage", "label"]
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
