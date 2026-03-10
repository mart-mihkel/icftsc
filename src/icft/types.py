from typing import Literal

type Task = Literal["seq2seq", "seq-cls", "causal-lm"]
type DatasetName = Literal["multinerd", "estner"]
type PromptMode = Literal["system", "random", "none"]
type PrefixInit = Literal["pretrained", "labels", "random"]
