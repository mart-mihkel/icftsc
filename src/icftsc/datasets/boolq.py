from typing import Literal, TypedDict, cast

from datasets.dataset_dict import DatasetDict
from datasets.load import load_dataset
from datasets.splits import Split
from transformers import BatchEncoding, PreTrainedTokenizerFast

from icftsc.constants import bert_model_types, gpt_model_types, t5_model_types
from icftsc.datasets.common import DatasetInfo, get_causal_batch
from icftsc.logging import logger
from icftsc.types import Task

type BoolQALabel = Literal["false", "true"]


class BoolqBatch(TypedDict):
    idx: list[int]
    passage: list[str]
    question: list[str]
    label: list[int]


id2label: dict[int, BoolQALabel] = {
    0: "false",
    1: "true",
}

label2id: dict[BoolQALabel, int] = {
    "false": 0,
    "true": 1,
}

examples = [
    (
        "Passage: The sky appears blue during the day due to Rayleigh "
        "scattering.\nQuestion: Is the sky blue?\nAnswer: true\n"
    ),
    (
        "Passage: Fish are animals that live exclusively underwater and "
        "breathe using gills.\nQuestion: Can fish breathe on land?\nAnswer: false\n"
    ),
    (
        "Passage: Water freezes at 0 degrees Celsius and boils at 100 "
        "degrees Celsius at sea level.\n"
        "Question: Does water freeze at room temperature?\nAnswer: false\n"
    ),
    (
        "Passage: The Earth orbits around the Sun in approximately 365 "
        "days.\nQuestion: Does the Earth orbit the Sun?\nAnswer: true\n"
    ),
    (
        "Passage: Photosynthesis is the process by which plants convert "
        "sunlight into energy.\nQuestion: Do plants produce their own food?"
        "\nAnswer: true\n"
    ),
    (
        "Passage: The Great Wall of China is visible from space with "
        "naked eye.\nQuestion: Is the Great Wall visible from space?"
        "\nAnswer: false\n"
    ),
    (
        "Passage: Lightning is a discharge of electricity that occurs "
        "during thunderstorms.\nQuestion: Is lightning caused by electricity?"
        "\nAnswer: true\n"
    ),
    (
        "Passage: The human body contains 206 bones in adulthood.\n"
        "Question: Do adults have more than 300 bones?\nAnswer: false\n"
    ),
]


def _bert_sys_prompt(bos: str, sep: str) -> str:
    return f"{bos}Answer the question based on the passage.{sep}"


def _bert_prompt(question: str, passage: str, bos: str, sep: str, eos: str) -> str:
    return f"{bos}{question}{sep}{passage}{eos}"


def _gpt_sys_prompt(bos: str | None) -> str:
    _bos = bos if bos is not None else ""
    return (
        f"{_bos}"
        "You are a Boolean Question Answering expert.Given a passage and a "
        'question, answer with exactly one word: "true" or "false".\nDo '
        "not provide any explanation.\n"
    )


def _gpt_prompt(question: str, passage: str, bos: str | None) -> str:
    _bos = bos if bos is not None else ""
    return f"{_bos}Passage: {passage}\nQuestion: {question}\nAnswer: "


def _t5_sys_prompt() -> str:
    return (
        "boolean question answering: given a passage and a question, "
        'answer with "true" or "false".\n'
    )


def _t5_prompt(question: str, passage: str) -> str:
    return f"passage: {passage}\nquestion: {question}\nanswer: "


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
    question: str,
    passage: str,
) -> str:
    if model_type in bert_model_types:
        return _bert_prompt(
            question=question,
            passage=passage,
            bos=tokenizer.bos_token,
            sep=tokenizer.sep_token,
            eos=tokenizer.eos_token,
        )

    if model_type in gpt_model_types:
        return _gpt_prompt(question=question, passage=passage, bos=tokenizer.bos_token)

    if model_type in t5_model_types:
        return _t5_prompt(question=question, passage=passage)

    raise NotImplementedError(f"Model type '{model_type}'")


def _tokenize(
    batch: BoolqBatch,
    tokenizer: PreTrainedTokenizerFast,
    model_type: str,
    task: Task,
) -> BatchEncoding:
    prompts: list[str] = []
    labels: list[int] = []

    it = zip(batch["passage"], batch["question"], batch["label"], strict=True)
    for passage, question, label_id in it:
        prompt = _get_prompt(
            model_type=model_type,
            tokenizer=tokenizer,
            question=question,
            passage=passage,
        )

        prompts.append(prompt)
        labels.append(label_id)

    enc = tokenizer(prompts, truncation=True, add_special_tokens=False)
    enc["labels"] = labels

    if task == "seqcls":
        return enc

    if task == "causal":
        _id2label = id2label | {-1: "private"}
        return get_causal_batch(tokenizer=tokenizer, enc=enc, id2label=_id2label)

    if task == "seq2seq":
        _id2label = id2label | {-1: "private"}
        eos = tokenizer.eos_token_id
        _labels = [[*tokenizer.encode(_id2label[label_id]), eos] for label_id in labels]
        enc["labels"] = _labels
        return enc

    raise NotImplementedError(f"Task '{task}'")


def init_boolq(
    tokenizer: PreTrainedTokenizerFast,
    model_type: str,
    task: Task,
    workers: int = 0,
    split: Split | None = None,
) -> DatasetDict:
    data = cast(DatasetDict, load_dataset("aps/super_glue", "boolq", split=split))

    if "validation" in data:
        data["dev"] = data.pop("validation")

    cols = ["question", "passage", "label"]
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


def init_boolq_info(
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
