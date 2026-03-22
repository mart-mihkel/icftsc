from typing import Literal, TypedDict, cast

from datasets.dataset_dict import DatasetDict
from datasets.load import load_dataset
from datasets.splits import Split
from transformers import BatchEncoding, PreTrainedTokenizerFast

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


class EstnerBatch(TypedDict):
    doc_id: list[int]
    sent_id: list[int]
    tokens: list[list[str]]
    ner_tags: list[list[str]]
    ner_tags1: list[list[str]]
    ner_tags2: list[list[str]]


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
    (
        "lause: Koosolek algab kell kolm pärastlõunal.\n"
        "nimeüksus: kell kolm\nmärgend: TIME\n"
    ),
    (
        "lause: Ta kirjutas raamatu pealkirjaga 'Tarkus'.\n"
        "nimeüksus: Tarkus\nmärgend: TITLE\n"
    ),
    (
        "lause: Ta ostis uue auto 25000 euro eest.\n"
        "nimeüksus: 25000 euro\nmärgend: MONEY\n"
    ),
    ("lause: Tulemus näitab 75% kasvu.\nnimeüksus: 75%\nmärgend: PERCENT\n"),
]


def _bert_sys_prompt(bos: str, sep: str) -> str:
    return f"{bos}Määra nimeüksuse NER märgen lauses.{sep}"


def _bert_prompt(sentence: str, entity: str, bos: str, sep: str, eos: str) -> str:
    return f"{bos}{sentence}{sep}{entity}{eos}"


def _gpt_sys_prompt(bos: str | None) -> str:
    _bos = bos if bos is not None else ""
    return (
        f"{_bos}"
        "Sa oled nimeüksuste tuvastamise (NER) ekspert. Sulle antakse lause ja "
        "sihtüksus ning pead tagastama õige märgendi. Kasuta täpselt ühte "
        "järgmistest märgenditest: PER, ORG, LOC, GPE, PROD, EVENT, DATE, TIME "
        "TITLE, MONEY, PERCENT, O. Vasta ainult märgendiga ilma selgituseta.\n"
    )


def _gpt_prompt(sentence: str, entity: str, bos: str | None) -> str:
    _bos = bos if bos is not None else ""
    return f"{_bos}lause: {sentence}\nnimeüksus: {entity}\nmärgend: "


def _t5_sys_prompt() -> str:
    return (
        "ner: tuvasta lauses oleva nimeüksuse NER-märgend.\nmärgendid: "
        "PER, ORG, LOC, GPE, PROD, EVENT, DATE, TIME, TITLE, MONEY, PERCENT, "
        "O.\n"
    )


def _t5_prompt(sentence: str, entity: str) -> str:
    return f"lause: {sentence}\nimeüksus: {entity}\nmärgend: "


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
    sentence: str,
    entity: str,
) -> str:
    if model_type in bert_model_types:
        return _bert_prompt(
            sentence=sentence,
            entity=entity,
            bos=tokenizer.bos_token,
            sep=tokenizer.sep_token,
            eos=tokenizer.eos_token,
        )

    if model_type in gpt_model_types:
        return _gpt_prompt(sentence=sentence, entity=entity, bos=tokenizer.bos_token)

    if model_type in t5_model_types:
        return _t5_prompt(sentence=sentence, entity=entity)

    raise NotImplementedError(f"Model type '{model_type}'")


def _tokenize(
    batch: EstnerBatch,
    tokenizer: PreTrainedTokenizerFast,
    model_type: str,
    task: Task,
) -> BatchEncoding:
    prompts: list[str] = []
    labels: list[int] = []

    for tokens, tags in zip(batch["tokens"], batch["ner_tags"], strict=True):
        sentence = " ".join(tokens)
        tokens, tags = _join_spans(tokens=tokens, tags=tags)

        for token, tag in zip(tokens, tags, strict=True):
            prompt = _get_prompt(
                model_type=model_type,
                tokenizer=tokenizer,
                sentence=sentence,
                entity=token,
            )

            prompts.append(prompt)
            labels.append(label2id[tag])

    enc = tokenizer(prompts, truncation=True, add_special_tokens=False)
    if task == "seqcls":
        enc["labels"] = labels
    elif task == "causal":
        enc["labels"] = [
            [-100] * len(prompt_ids)
            + tokenizer.encode(id2label[tag_id])
            + [tokenizer.eos_token_id]
            for prompt_ids, tag_id in zip(enc["input_ids"], labels, strict=True)
        ]
    elif task == "seq2seq":
        enc["labels"] = [
            [*tokenizer.encode(id2label[tag_id]), tokenizer.eos_token_id]
            for tag_id in labels
        ]
    else:
        raise NotImplementedError(f"Task '{task}'")

    return enc


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


def init_estner(
    tokenizer: PreTrainedTokenizerFast,
    model_type: str,
    task: Task,
    workers: int = 0,
    split: Split | None = None,
) -> DatasetDict:
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


def init_estner_info(
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
