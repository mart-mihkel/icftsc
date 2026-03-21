from typing import Literal

type DatasetName = Literal["multinerd", "estner", "superglue-boolq"]
type PrefixInit = Literal["pretrained", "random"]
type Task = Literal["seqcls", "causal", "seq2seq"]
