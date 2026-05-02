from typing import Literal, TypedDict

type Architecture = Literal["encoder", "decoder", "encoder-decoder"]
type DatasetName = Literal["multinerd", "estner", "boolq", "wic", "obl"]
type PrefixInit = Literal["pretrained", "random"]


class DatasetInfo(TypedDict):
    id2label: dict[int, str]
    label2id: dict[str, int]
    system_prompt: str
