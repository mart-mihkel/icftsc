from textwrap import dedent
from typing import Literal, TypedDict, cast

from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from transformers import BatchEncoding, PreTrainedTokenizerFast

from icftsc.logging import logger
from icftsc.types import Architecture, DatasetInfo

_permalink = (
    "https://raw.githubusercontent.com/estnltk/estnltk-model-data/"
    "c6f373a49d29a5a4b5e92326e5309b173bfcd3d1"
    "/phrase_removal/benchmarks/obl_phrases/obl_max/data/obl_all_hand_annotated.csv"
)

type OblLabel = Literal["bound", "unnatural", "redundant comma", "free", "dubious"]


class OblExample(TypedDict):
    id: int
    fpath: str
    sentence: str
    remove_start: int
    remove_end: int
    removed: str
    label: OblLabel
    short_sent: str
    cons_score: float
    ual: float
    la: float


id2label: dict[int, OblLabel] = {
    0: "bound",
    1: "unnatural",
    2: "redundant comma",
    3: "free",
    4: "dubious",
}

label2id: dict[OblLabel, int] = {
    "bound": 0,
    "unnatural": 1,
    "redundant comma": 2,
    "free": 3,
    "dubious": 4,
}

examples = [
    (
        "Sentence: Esimesel korral oli vene rahvas sõja vastu , aga seekord "
        "ollakse siiski poolt .\n"
        "Span: Esimesel korral\n"
        "Answer: bound\n"
    ),
    (
        "Sentence: Suusakeskus hakkas poolest talvest tööle ja muutus üsna "
        "ruttu populaarseks .\n"
        "Span: poolest talvest\n"
        "Answer: free\n"
    ),
    (
        "Sentence: Hemodialüüsis patsientide jälgimisel jääb maha tohutu suur "
        "hulk meditsiinilist infot , mistõttu arstidel on mustrite nägemine "
        "üle pikema aja üsnagi problemaatiline .\n"
        "Span: Hemodialüüsis patsientide jälgimisel\n"
        "Answer: unnatural\n"
    ),
    (
        "Sentence: Päev hiljem , reede õhtul pidasid Tallinna Põhja "
        "politseiosakonna inspektorid suure vaevaga kinni 1997. aasta "
        "universaalkerega tumesinise Ford Mondeo .\n"
        "Span: Päev hiljem\n"
        "Answer: redundant comma\n"
    ),
    (
        "Sentence: 1990. aastal siirdus Gerda , kes vanemate ja passi järgi "
        "on soomlane , oma esimesele kodumaale .\n"
        "Span: vanemate ja passi järgi\n"
        "Answer: dubious\n"
    ),
]


def _enc_sys_prompt(sep: str) -> str:
    return dedent(f"""
        What is the relation between the span and the sentence,
        is it bound, free, dubious, unnatural or redundant comma?{sep}
    """).strip()


def _enc_prompt(example: OblExample, sep: str) -> str:
    return f"{example['removed']}{sep}{example['sentence']}"


def _dec_sys_prompt() -> str:
    return dedent("""
        You are an obligation labeling model. Classify the removed span into one label:

        - bound: required for grammatical structure, removal breaks grammar
        - unnatural: grammatical but awkward or non-native
        - redundant comma: unnecessary or incorrect comma
        - free: optional, removable without loss of meaning or grammar
        - dubious: unclear or borderline incorrect

        Output only the label.
    """).strip()


def _dec_prompt(example: OblExample) -> str:
    return dedent(f"""
        Sentence: {example["sentence"]}
        Span: {example["removed"]}
        Label:
    """).strip()


def _encdec_sys_prompt() -> str:
    return dedent("""
        obligation labeling: classify the removed text into one of five labels.

        bound: required for grammatical structure; removal breaks grammar
        unnatural: grammatically optional but awkward or non-native
        redundant comma: unnecessary or incorrect comma
        free: optional; removable without changing meaning or grammar
        dubious: unclear or borderline case

        output only the label
    """).strip()


def _encdec_prompt(example: OblExample) -> str:
    return dedent(f"""
        sentence: {example["sentence"]}
        span: {example["removed"]}
        label:
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
    example: OblExample,
) -> str:
    if arch == "encoder":
        return _enc_prompt(example, tokenizer.sep_token)

    if arch == "decoder":
        return _dec_prompt(example)

    if arch == "encoder-decoder":
        return _encdec_prompt(example)

    raise NotImplementedError(f"architecture '{arch}'")


def _tokenize(
    example: OblExample,
    tokenizer: PreTrainedTokenizerFast,
    arch: Architecture,
    n_shot: int,
) -> BatchEncoding:
    sys = _get_sys_prompt(tokenizer, arch, n_shot)
    prompt = _get_prompt(tokenizer, arch, example)
    label = example["label"]

    prompt_enc = tokenizer(f"{sys}\n{prompt}", truncation=True)
    prompt_len = len(prompt_enc["input_ids"])

    if arch == "encoder":
        prompt_enc["label"] = label2id[label]
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


def load_obl(
    tokenizer: PreTrainedTokenizerFast,
    arch: Architecture,
    n_shot: int,
) -> tuple[DatasetDict, DatasetInfo]:
    logger.debug("load obl csv from github permalink")
    raw = Dataset.from_csv(_permalink, sep=";").rename_column("type", "label")

    split1 = raw.train_test_split(test_size=1000, seed=0)
    split2 = split1["train"].train_test_split(test_size=500, seed=0)
    data = DatasetDict(
        {
            "train": split2["train"],
            "dev": split2["test"],
            "test": split1["test"],
        }
    )

    logger.debug("tokenize obl")
    cols = [
        "id",
        "fpath",
        "sentence",
        "remove_start",
        "remove_end",
        "removed",
        "label",
        "short_sent",
        "cons_score",
        "ual",
        "la",
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
