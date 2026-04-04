from unittest.mock import patch

from datasets.dataset_dict import DatasetDict
from peft import PeftModel
from transformers import PreTrainedModel, PreTrainedTokenizerFast

from icftsc.datasets.boolq import load_boolq
from icftsc.datasets.estner import load_estner
from icftsc.datasets.multinerd import load_multinerd
from icftsc.datasets.util import get_collator
from icftsc.datasets.wic import load_wic


def test_gpt2_wic_forward(
    gpt2: PreTrainedModel,
    gpt2_tokenizer: PreTrainedTokenizerFast,
    wic: DatasetDict,
):
    with patch("icftsc.datasets.wic.load_dataset", return_value=wic):
        data, _ = load_wic(gpt2_tokenizer, "gpt2", "causal", 0)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(gpt2_tokenizer, "causal")

    batch = collator(examples)
    out = gpt2(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_gpt2_boolq_forward(
    gpt2: PreTrainedModel,
    gpt2_tokenizer: PreTrainedTokenizerFast,
    boolq: DatasetDict,
):
    with patch("icftsc.datasets.boolq.load_dataset", return_value=boolq):
        data, _ = load_boolq(gpt2_tokenizer, "gpt2", "causal", 0)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(gpt2_tokenizer, "causal")

    batch = collator(examples)
    out = gpt2(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_gpt2_estner_forward(
    gpt2: PreTrainedModel,
    gpt2_tokenizer: PreTrainedTokenizerFast,
    estner: DatasetDict,
):
    with patch("icftsc.datasets.estner.load_dataset", return_value=estner):
        data, _ = load_estner(gpt2_tokenizer, "gpt2", "causal", 0)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(gpt2_tokenizer, "causal")

    batch = collator(examples)
    out = gpt2(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_gpt2_multinerd_forward(
    gpt2: PreTrainedModel,
    gpt2_tokenizer: PreTrainedTokenizerFast,
    multinerd: DatasetDict,
):
    with patch("icftsc.datasets.multinerd.load_dataset", return_value=multinerd):
        data, _ = load_multinerd(gpt2_tokenizer, "gpt2", "causal", 0, False)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(gpt2_tokenizer, "causal")

    batch = collator(examples)
    out = gpt2(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_pt_gpt2_wic_forward(
    pt_gpt2: PeftModel,
    gpt2_tokenizer: PreTrainedTokenizerFast,
    wic: DatasetDict,
):
    with patch("icftsc.datasets.wic.load_dataset", return_value=wic):
        data, _ = load_wic(gpt2_tokenizer, "gpt2", "causal", 0)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(gpt2_tokenizer, "causal")

    batch = collator(examples)
    out = pt_gpt2(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_pt_gpt2_boolq_forward(
    pt_gpt2: PeftModel,
    gpt2_tokenizer: PreTrainedTokenizerFast,
    boolq: DatasetDict,
):
    with patch("icftsc.datasets.boolq.load_dataset", return_value=boolq):
        data, _ = load_boolq(gpt2_tokenizer, "gpt2", "causal", 0)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(gpt2_tokenizer, "causal")

    batch = collator(examples)
    out = pt_gpt2(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_pt_gpt2_estner_forward(
    pt_gpt2: PeftModel,
    gpt2_tokenizer: PreTrainedTokenizerFast,
    estner: DatasetDict,
):
    with patch("icftsc.datasets.estner.load_dataset", return_value=estner):
        data, _ = load_estner(gpt2_tokenizer, "gpt2", "causal", 0)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(gpt2_tokenizer, "causal")

    batch = collator(examples)
    out = pt_gpt2(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_pt_gpt2_multinerd_forward(
    pt_gpt2: PeftModel,
    gpt2_tokenizer: PreTrainedTokenizerFast,
    multinerd: DatasetDict,
):
    with patch("icftsc.datasets.multinerd.load_dataset", return_value=multinerd):
        data, _ = load_multinerd(gpt2_tokenizer, "gpt2", "causal", 0, False)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(gpt2_tokenizer, "causal")

    batch = collator(examples)
    out = pt_gpt2(**batch)

    assert out.loss is not None
    assert out.logits is not None
