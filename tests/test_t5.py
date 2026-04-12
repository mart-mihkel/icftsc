from unittest.mock import patch

from datasets.dataset_dict import DatasetDict
from peft import PeftModel
from transformers import PreTrainedModel, PreTrainedTokenizerFast

from icftsc.datasets.boolq import load_boolq
from icftsc.datasets.estner import load_estner
from icftsc.datasets.multinerd import load_multinerd
from icftsc.datasets.util import get_collator
from icftsc.datasets.wic import load_wic

_arch = "encoder-decoder"


def test_t5_wic_forward(
    t5: PreTrainedModel,
    t5_tokenizer: PreTrainedTokenizerFast,
    wic: DatasetDict,
) -> None:
    with patch("icftsc.datasets.wic.load_dataset", return_value=wic):
        data, _ = load_wic(t5_tokenizer, _arch, 0)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(t5_tokenizer, _arch)

    batch = collator(examples)
    out = t5(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_t5_boolq_forward(
    t5: PreTrainedModel,
    t5_tokenizer: PreTrainedTokenizerFast,
    boolq: DatasetDict,
) -> None:
    with patch("icftsc.datasets.boolq.load_dataset", return_value=boolq):
        data, _ = load_boolq(t5_tokenizer, _arch, 0)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(t5_tokenizer, _arch)

    batch = collator(examples)
    out = t5(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_t5_estner_forward(
    t5: PreTrainedModel,
    t5_tokenizer: PreTrainedTokenizerFast,
    estner: DatasetDict,
) -> None:
    with patch("icftsc.datasets.estner.load_dataset", return_value=estner):
        data, _ = load_estner(t5_tokenizer, _arch, 0)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(t5_tokenizer, _arch)

    batch = collator(examples)
    out = t5(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_t5_multinerd_forward(
    t5: PreTrainedModel,
    t5_tokenizer: PreTrainedTokenizerFast,
    multinerd: DatasetDict,
) -> None:
    with patch("icftsc.datasets.multinerd.load_dataset", return_value=multinerd):
        data, _ = load_multinerd(t5_tokenizer, _arch, 0, False)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(t5_tokenizer, _arch)

    batch = collator(examples)
    out = t5(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_pt_t5_wic_forward(
    pt_t5: PeftModel,
    t5_tokenizer: PreTrainedTokenizerFast,
    wic: DatasetDict,
) -> None:
    with patch("icftsc.datasets.wic.load_dataset", return_value=wic):
        data, _ = load_wic(t5_tokenizer, _arch, 0)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(t5_tokenizer, _arch)

    batch = collator(examples)
    out = pt_t5(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_pt_t5_boolq_forward(
    pt_t5: PeftModel,
    t5_tokenizer: PreTrainedTokenizerFast,
    boolq: DatasetDict,
) -> None:
    with patch("icftsc.datasets.boolq.load_dataset", return_value=boolq):
        data, _ = load_boolq(t5_tokenizer, _arch, 0)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(t5_tokenizer, _arch)

    batch = collator(examples)
    out = pt_t5(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_pt_t5_estner_forward(
    pt_t5: PreTrainedModel,
    t5_tokenizer: PreTrainedTokenizerFast,
    estner: DatasetDict,
) -> None:
    with patch("icftsc.datasets.estner.load_dataset", return_value=estner):
        data, _ = load_estner(t5_tokenizer, _arch, 0)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(t5_tokenizer, _arch)

    batch = collator(examples)
    out = pt_t5(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_pt_t5_multinerd_forward(
    pt_t5: PreTrainedModel,
    t5_tokenizer: PreTrainedTokenizerFast,
    multinerd: DatasetDict,
) -> None:
    with patch("icftsc.datasets.multinerd.load_dataset", return_value=multinerd):
        data, _ = load_multinerd(t5_tokenizer, _arch, 0, False)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(t5_tokenizer, _arch)

    batch = collator(examples)
    out = pt_t5(**batch)

    assert out.loss is not None
    assert out.logits is not None
