from typing import Literal, TypedDict, cast

from datasets.dataset_dict import DatasetDict
from datasets.load import load_dataset
from datasets.splits import Split
from datasets.utils.info_utils import VerificationMode
from transformers import PreTrainedTokenizerFast

from icftsc.constants import (
    decoder_model_types,
    encoder_decoder_model_types,
    encoder_model_types,
)
from icftsc.logging import logger
from icftsc.types import DatasetInfo, Task

type MultinerdLang = Literal[
    "zh",
    "nl",
    "en",
    "fr",
    "de",
    "it",
    "pl",
    "pt",
    "ru",
    "es",
]

type MultinerdTag = Literal[
    "PER",
    "ORG",
    "LOC",
    "ANIM",
    "BIO",
    "CEL",
    "DIS",
    "EVE",
    "FOOD",
    "INST",
    "MEDIA",
    "MYTH",
    "PLANT",
    "TIME",
    "VEHI",
]


class MultinerdExamples(TypedDict):
    tokens: list[list[str]]
    ner_tags: list[list[int]]
    lang: list[MultinerdLang]


_id2label_full: dict[int, str] = {
    0: "O",
    1: "B-PER",
    2: "I-PER",
    3: "B-ORG",
    4: "I-ORG",
    5: "B-LOC",
    6: "I-LOC",
    7: "B-ANIM",
    8: "I-ANIM",
    9: "B-BIO",
    10: "I-BIO",
    11: "B-CEL",
    12: "I-CEL",
    13: "B-DIS",
    14: "I-DIS",
    15: "B-EVE",
    16: "I-EVE",
    17: "B-FOOD",
    18: "I-FOOD",
    19: "B-INST",
    20: "I-INST",
    21: "B-MEDIA",
    22: "I-MEDIA",
    23: "B-MYTH",
    24: "I-MYTH",
    25: "B-PLANT",
    26: "I-PLANT",
    27: "B-TIME",
    28: "I-TIME",
    29: "B-VEHI",
    30: "I-VEHI",
}

id2label: dict[int, MultinerdTag] = {
    0: "PER",
    1: "ORG",
    2: "LOC",
    3: "ANIM",
    4: "BIO",
    5: "CEL",
    6: "DIS",
    7: "EVE",
    8: "FOOD",
    9: "INST",
    10: "MEDIA",
    11: "MYTH",
    12: "PLANT",
    13: "TIME",
    14: "VEHI",
}

label2id: dict[MultinerdTag, int] = {
    "PER": 0,
    "ORG": 1,
    "LOC": 2,
    "ANIM": 3,
    "BIO": 4,
    "CEL": 5,
    "DIS": 6,
    "EVE": 7,
    "FOOD": 8,
    "INST": 9,
    "MEDIA": 10,
    "MYTH": 11,
    "PLANT": 12,
    "TIME": 13,
    "VEHI": 14,
}

examples = [
    ("sentence: John works at Google in California.\nentity: John\ntag: PER\n"),
    ("sentence: Paris is the capital of France.\nentity: Paris\ntag: LOC\n"),
    ("sentence: The dog chased the cat across the garden.\nentity: dog\ntag: ANIM\n"),
    ("sentence: I love eating sushi and pasta for dinner.\nentity: sushi\ntag: FOOD\n"),
    ("sentence: The car drove quickly down the highway.\nentity: car\ntag: VEHI\n"),
    (
        "sentence: The meeting was held at the United Nations headquarters.\n"
        "entity: United Nations\n"
        "tag: ORG\n"
    ),
    (
        "sentence: Evolution shaped the diversity of life on Earth.\n"
        "entity: Evolution\n"
        "tag: BIO\n"
    ),
    (
        "sentence: Einstein developed the theory of relativity.\n"
        "entity: Einstein\n"
        "tag: CEL\n"
    ),
    (
        "sentence: The patient was diagnosed with diabetes last year.\n"
        "entity: diabetes\n"
        "tag: DIS\n"
    ),
    (
        "sentence: The Olympics will be held in Tokyo next summer.\n"
        "entity: Olympics\n"
        "tag: EVE\n"
    ),
    (
        "sentence: The telescope was invented several centuries ago.\n"
        "entity: telescope\n"
        "tag: INST\n"
    ),
    (
        "sentence: I watched an interesting movie on Netflix last night.\n"
        "entity: Netflix\n"
        "tag: MEDIA\n"
    ),
    (
        "sentence: The dragon guarded the ancient treasure in the cave.\n"
        "entity: dragon\n"
        "tag: MYTH\n"
    ),
    (
        "sentence: Roses bloom beautifully in the garden during spring.\n"
        "entity: Roses\n"
        "tag: PLANT\n"
    ),
    (
        "sentence: The meeting is scheduled for Monday morning.\n"
        "entity: Monday\n"
        "tag: TIME\n"
    ),
]


def _enc_sys_prompt(sep: str) -> str:
    return f"Identify the NER tag of the entity in the sentence.{sep}"


def _enc_prompt(sentence: str, entity: str, sep: str) -> str:
    return f"{sentence}{sep}{entity}"


def _dec_sys_prompt() -> str:
    return (
        "Identify the NER tag of the entity in the sentence. Possible tags are: "
        "PER, ORG, LOC, ANIM, BIO, CEL, DIS, EVE, FOOD, INST, MEDIA, MYTH, "
        "PLANT, TIME, VEHI.\n"
    )


def _dec_prompt(sentence: str, entity: str) -> str:
    return f"sentence: {sentence}\nentity: {entity}\ntag:"


def _encdec_sys_prompt() -> str:
    return (
        "ner: identify the ner tag of the entity in the sentence.\ntags: PER "
        "ORG, LOC, ANIM, BIO, CEL, DIS, EVE, FOOD, INST, MEDIA, MYTH, PLANT, "
        "TIME, VEHI.\n"
    )


def _encdec_prompt(sentence: str, entity: str) -> str:
    return f"sentence: {sentence}\nentity: {entity}\ntag:"


def _get_sys_prompt(
    tokenizer: PreTrainedTokenizerFast,
    model_type: str,
    n_shot: int,
) -> str:
    if n_shot > len(examples):
        raise ValueError("Requested more examples than exist")

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
    sentence: str,
    entity: str,
) -> str:
    if model_type in encoder_model_types:
        return _enc_prompt(sentence, entity, tokenizer.sep_token)

    if model_type in decoder_model_types:
        return _dec_prompt(sentence, entity)

    if model_type in encoder_decoder_model_types:
        return _encdec_prompt(sentence, entity)

    raise NotImplementedError(f"Model type '{model_type}'")


def _tokenize_batch(
    examples: MultinerdExamples,
    tokenizer: PreTrainedTokenizerFast,
    model_type: str,
    task: Task,
    n_shot: int,
) -> dict[str, list]:
    all_ids, all_attn, all_labels = [], [], []

    for tokens, tag_ids in zip(examples["tokens"], examples["ner_tags"], strict=True):
        sentence = " ".join(tokens)
        entities, tag_ids = _join_spans(tokens, tag_ids)

        for entity, tag_id in zip(entities, tag_ids, strict=True):
            if tag_id == -1:
                continue

            sys = _get_sys_prompt(tokenizer, model_type, n_shot)
            prompt = _get_prompt(tokenizer, model_type, sentence, entity)

            prompt_enc = tokenizer(f"{sys}\n{prompt}", truncation=True)
            prompt_len = len(prompt_enc["input_ids"])

            if task == "seqcls":
                all_ids.append(prompt_enc["input_ids"])
                all_attn.append(prompt_enc["attention_mask"])
                all_labels.append(tag_id)
                continue

            answer = f"{sys}\n{prompt} {id2label[tag_id]}"
            answer_enc = tokenizer(answer, truncation=True)
            labels_enc = answer_enc["input_ids"].copy()

            all_ids.append(answer_enc["input_ids"])
            all_attn.append(answer_enc["attention_mask"])

            if task == "causal":
                labels_enc[:prompt_len] = [-100] * prompt_len
                all_labels.append(labels_enc)
                continue

            if task == "seq2seq":
                all_labels.append(labels_enc[(prompt_len - 1) :])
                continue

            raise NotImplementedError(f"Task '{task}'")

    return {"input_ids": all_ids, "attention_mask": all_attn, "labels": all_labels}


def _join_spans(
    tokens: list[str],
    tag_ids: list[int],
) -> tuple[list[MultinerdTag], list[int]]:
    out_ids = []
    out_tokens = []
    for token, tag_id in zip(tokens, tag_ids, strict=True):
        tag = _id2label_full[tag_id]

        if tag.startswith("B-"):
            tag = cast(MultinerdTag, tag[2:])
            out_ids.append(label2id[tag])
            out_tokens.append(token)
        elif tag.startswith("I-"):
            out_tokens[-1] = f"{out_tokens[-1]} {token}"
        elif tag == "O":
            out_ids.append(-1)
            out_tokens.append(token)
        else:
            tag = cast(MultinerdTag, tag)
            out_ids.append(label2id[tag])
            out_tokens.append(token)

    return out_tokens, out_ids


def _filter_english(batch: MultinerdExamples) -> list[bool]:
    return [lang == "en" for lang in batch["lang"]]


def load_multinerd(
    tokenizer: PreTrainedTokenizerFast,
    model_type: str,
    task: Task,
    n_shot: int,
    subset: float = 0.1,
    filter_en: bool = True,
    split: Split | None = None,
) -> tuple[DatasetDict, DatasetInfo]:
    """
    Initialize a modified subset of the MultiNERD dataset.

    The BIO tagging task is converted to a regular NER tagging task by joining
    tokens with B- and I- prefixes into a single span. O tags are dropped
    entirely.

    Each token is split into a separate sample containing the entire context
    sentence and the target token. The task is to classify the tag of the token
    in the entire sequence.
    """
    data = load_dataset(
        "Babelscape/multinerd",
        split=split,
        verification_mode=VerificationMode.NO_CHECKS,
    )

    data = cast(DatasetDict, data)

    if "validation" in data:
        data["dev"] = data.pop("validation")

    if filter_en:
        logger.info("using english only subset")
        data = data.filter(_filter_english, batched=True)

    logger.warning("using %d%% of dev and train data", int(subset * 100))
    if "train" in data:
        idx_train = range(int(subset * len(data["train"])))
        data["train"] = data["train"].select(idx_train)

    if "dev" in data:
        idx_dev = range(int(subset * len(data["dev"])))
        data["dev"] = data["dev"].select(idx_dev)

    cols = ["tokens", "ner_tags", "lang"]
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
