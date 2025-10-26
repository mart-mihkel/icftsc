from typing import Literal

type ICFTTask = Literal["seq2seq", "seq-cls", "causal-lm"]
type ICFTDataset = Literal["multinerd"]
type PromptMode = Literal["ner", "random", "none"]
type PrefixInit = Literal["pretrained", "labels", "random"]
