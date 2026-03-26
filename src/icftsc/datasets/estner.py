from typing import Literal, TypedDict, cast

from datasets.dataset_dict import DatasetDict
from datasets.load import load_dataset
from datasets.splits import Split
from transformers import PreTrainedTokenizerFast

from icftsc.constants import bert_model_types, gpt_model_types, t5_model_types
from icftsc.datasets.common import DatasetInfo
from icftsc.logging import logger
from icftsc.types import Task

type EstnerTag = Literal[
    "O",
    "PER",
    "GPE",
    "LOC",
    "ORG",
    "PROD",
    "EVENT",
    "DATE",
    "TIME",
    "TITLE",
    "MONEY",
    "PERCENT",
]


class EstnerExamples(TypedDict):
    doc_id: list[int]
    sent_id: list[int]
    tokens: list[list[str]]
    ner_tags: list[list[str]]
    ner_tags1: list[list[str]]
    ner_tags2: list[list[str]]


_id2label_full: dict[int, str] = {
    0: "O",
    1: "B-PER",
    2: "I-PER",
    3: "B-GPE",
    4: "I-GPE",
    5: "B-LOC",
    6: "I-LOC",
    7: "B-ORG",
    8: "I-ORG",
    9: "B-PROD",
    10: "I-PROD",
    11: "B-EVENT",
    12: "I-EVENT",
    13: "B-DATE",
    14: "I-DATE",
    15: "B-TIME",
    16: "I-TIME",
    17: "B-TITLE",
    18: "I-TITLE",
    19: "B-MONEY",
    20: "I-MONEY",
    21: "B-PERCENT",
    22: "I-PERCENT",
}

id2label: dict[int, EstnerTag] = {
    0: "O",
    1: "PER",
    2: "GPE",
    3: "LOC",
    4: "ORG",
    5: "PROD",
    6: "EVENT",
    7: "DATE",
    8: "TIME",
    9: "TITLE",
    10: "MONEY",
    11: "PERCENT",
}

label2id: dict[EstnerTag, int] = {
    "O": 0,
    "PER": 1,
    "GPE": 2,
    "LOC": 3,
    "ORG": 4,
    "PROD": 5,
    "EVENT": 6,
    "DATE": 7,
    "TIME": 8,
    "TITLE": 9,
    "MONEY": 10,
    "PERCENT": 11,
}

examples = [
    ("lause: Mari töötab Google'is Californias.\nnimeüksus: Mari\nmärgend: PER\n"),
    ("lause: Koosolek toimus ÜRO peakorteris.\nnimeüksus: ÜRO\nmärgend: ORG\n"),
    ("lause: Tallinn on Eesti pealinn.\nnimeüksus: Tallinn\nmärgend: LOC\n"),
    ("lause: Eesti asub Põhja-Euroopas.\nnimeüksus: Eesti\nmärgend: GPE\n"),
    ("lause: Ta jõi hommikul kuuma kohvi.\nnimeüksus: kohvi\nmärgend: PROD\n"),
    ("lause: Võidupüha tähistatakse juunis.\nnimeüksus: Võidupüha\nmärgend: EVENT\n"),
    ("lause: Ta sündis 1990. aastal.\nnimeüksus: 1990. aastal\nmärgend: DATE\n"),
    ("lause: Tulemus näitab 75% kasvu.\nnimeüksus: 75%\nmärgend: PERCENT\n"),
    (
        "lause: Koosolek algab kell kolm pärastlõunal.\n"
        "nimeüksus: kell kolm\n"
        "märgend: TIME\n"
    ),
    (
        "lause: Ta kirjutas raamatu pealkirjaga 'Tarkus'.\n"
        "nimeüksus: Tarkus\n"
        "märgend: TITLE\n"
    ),
    (
        "lause: Ta ostis uue auto 25000 euro eest.\n"
        "nimeüksus: 25000 euro\n"
        "märgend: MONEY\n"
    ),
]


def _bert_sys_prompt(sep: str) -> str:
    return f"Määra nimeüksuse NER märgen lauses.{sep}"


def _bert_prompt(sentence: str, entity: str, sep: str) -> str:
    return f"{sentence}{sep}{entity}"


def _gpt_sys_prompt() -> str:
    return (
        "Määra nimeüksuse NER märgen lauses. Võimalikut märgendid on: PER, ORG, "
        "LOC, GPE, PROD, EVENT, DATE, TIME, TITLE, MONEY, PERCENT, O.\n"
    )


def _gpt_prompt(sentence: str, entity: str) -> str:
    return f"lause: {sentence}\nnimeüksus: {entity}\nmärgend:"


def _t5_sys_prompt() -> str:
    return (
        "ner: tuvasta lauses oleva nimeüksuse NER-märgend.\n"
        "märgendid: PER, ORG, LOC, GPE, PROD, EVENT, DATE, TIME, TITLE, MONEY, "
        "PERCENT, O.\n"
    )


def _t5_prompt(sentence: str, entity: str) -> str:
    return f"lause: {sentence}\nnimeüksus: {entity}\nmärgend:"


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
    sentence: str,
    entity: str,
) -> str:
    if model_type in bert_model_types:
        return _bert_prompt(sentence, entity, tokenizer.sep_token)

    if model_type in gpt_model_types:
        return _gpt_prompt(sentence, entity)

    if model_type in t5_model_types:
        return _t5_prompt(sentence, entity)

    raise NotImplementedError(f"Model type '{model_type}'")


def _tokenize_batch(
    examples: EstnerExamples,
    tokenizer: PreTrainedTokenizerFast,
    model_type: str,
    task: Task,
    n_shot: int,
) -> dict[str, list]:
    all_ids, all_attn, all_labels = [], [], []

    for tokens, tags in zip(examples["tokens"], examples["ner_tags"], strict=True):
        sentence = " ".join(tokens)
        entities, tags = _join_spans(tokens, tags)

        for entity, tag in zip(entities, tags, strict=True):
            sys = _get_sys_prompt(tokenizer, model_type, n_shot)
            prompt = _get_prompt(tokenizer, model_type, sentence, entity)

            prompt_enc = tokenizer(f"{sys}\n{prompt}", truncation=True)
            prompt_len = len(prompt_enc["input_ids"])

            if task == "seqcls":
                all_ids.append(prompt_enc["input_ids"])
                all_attn.append(prompt_enc["attention_mask"])
                all_labels.append(label2id[tag])
                continue

            answer = f"{sys}\n{prompt} {tag}"
            answer_enc = tokenizer(answer, truncation=True)
            labels_enc = answer_enc["input_ids"].copy()

            all_ids.append(answer_enc["input_ids"])
            all_attn.append(answer_enc["attention_mask"])

            if task == "causal":
                labels_enc[:prompt_len] = [-100] * prompt_len
                all_labels.append(labels_enc)
                continue

            if task == "seq2seq":
                all_labels.append(labels_enc[prompt_len:])
                continue

            raise NotImplementedError(f"Task '{task}'")

    return {"input_ids": all_ids, "attention_mask": all_attn, "labels": all_labels}


def _join_spans(
    tokens: list[str],
    tags: list[str],
) -> tuple[list[str], list[EstnerTag]]:
    out_tags = []
    out_tokens = []
    for token, tag in zip(tokens, tags, strict=True):
        if tag.startswith("B-"):
            tag = cast(EstnerTag, tag[2:])
            out_tags.append(tag)
            out_tokens.append(token)
        elif tag.startswith("I-"):
            out_tokens[-1] = f"{out_tokens[-1]} {token}"
        else:
            tag = cast(EstnerTag, tag)
            out_tags.append(tag)
            out_tokens.append(token)

    return out_tokens, out_tags


def load_estner(
    tokenizer: PreTrainedTokenizerFast,
    model_type: str,
    task: Task,
    n_shot: int,
    split: Split | None = None,
) -> tuple[DatasetDict, DatasetInfo]:
    """
    Initialize a modified version of the EstNER dataset.

    The BIO tagging task is converted to a regular NER tagging task by joining
    tokens with B- and I- prefixes into a single span.

    Each token is split into a separate sample containing the entire context
    sentence and the target token. The task is to classify the tag of the token
    in the entire sequence.
    """
    data = cast(DatasetDict, load_dataset("tartuNLP/EstNER", split=split))

    cols = ["doc_id", "sent_id", "tokens", "ner_tags", "ner_tags_2", "ner_tags_3"]
    fn_kwargs = {
        "model_type": model_type,
        "tokenizer": tokenizer,
        "n_shot": n_shot,
        "task": task,
    }

    data = data.map(
        _tokenize_batch,
        batched=True,
        remove_columns=cols,
        fn_kwargs=fn_kwargs,
    )

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
