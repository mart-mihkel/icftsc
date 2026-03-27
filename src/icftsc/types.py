from typing import Literal, TypedDict

type DatasetName = Literal["multinerd", "estner", "boolq", "wic"]
type PromptMode = Literal["system", "random"]
type PrefixInit = Literal["pretrained", "random"]
type Task = Literal["seqcls", "causal", "seq2seq"]


class DatasetInfo(TypedDict):
    id2label: dict[int, str]
    label2id: dict[str, int]
    system_prompt: str
