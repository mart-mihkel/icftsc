from typing import Literal, TypedDict

type Architecture = Literal["encoder", "decoder", "encoder-decoder"]
type DatasetName = Literal["multinerd", "estner", "boolq", "wic"]
type PrefixInit = Literal["pretrained", "random"]
type Task = Literal["seqcls", "causal", "seq2seq"]


class DatasetInfo(TypedDict):
    id2label: dict[int, str]
    label2id: dict[str, int]
    system_prompt: str
