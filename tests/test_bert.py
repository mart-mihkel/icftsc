from typing import cast
from unittest import skip
from unittest.mock import patch

from datasets.dataset_dict import DatasetDict
from peft import PeftModel
from torch.nn import Linear
from transformers import (
    BertForSequenceClassification,
    PreTrainedTokenizerFast,
)

from icftsc.datasets.boolq import load_boolq
from icftsc.datasets.estner import load_estner
from icftsc.datasets.multinerd import load_multinerd
from icftsc.datasets.util import get_collator
from icftsc.datasets.wic import load_wic

_arch = "encoder"


def test_bert_wic_forward(
    bert: BertForSequenceClassification,
    bert_tokenizer: PreTrainedTokenizerFast,
    wic: DatasetDict,
) -> None:
    with patch("icftsc.datasets.wic.load_dataset", return_value=wic):
        data, _ = load_wic(bert_tokenizer, _arch, 0)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(bert_tokenizer, _arch)

    batch = collator(examples)
    out = bert(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_bert_boolq_forward(
    bert: BertForSequenceClassification,
    bert_tokenizer: PreTrainedTokenizerFast,
    boolq: DatasetDict,
) -> None:
    with patch("icftsc.datasets.boolq.load_dataset", return_value=boolq):
        data, _ = load_boolq(bert_tokenizer, _arch, 0)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(bert_tokenizer, _arch)

    batch = collator(examples)
    out = bert(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_bert_estner_forward(
    bert: BertForSequenceClassification,
    bert_tokenizer: PreTrainedTokenizerFast,
    estner: DatasetDict,
) -> None:
    with patch("icftsc.datasets.estner.load_dataset", return_value=estner):
        data, info = load_estner(bert_tokenizer, _arch, 0)

    num_labels = len(info["id2label"])
    bert.num_labels = num_labels
    bert.classifier = Linear(bert.config.hidden_size, num_labels)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(bert_tokenizer, _arch)

    batch = collator(examples)
    out = bert(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_bert_multinerd_forward(
    bert: BertForSequenceClassification,
    bert_tokenizer: PreTrainedTokenizerFast,
    multinerd: DatasetDict,
) -> None:
    with patch("icftsc.datasets.multinerd.load_dataset", return_value=multinerd):
        data, info = load_multinerd(bert_tokenizer, _arch, 0, False)

    num_labels = len(info["id2label"])
    bert.num_labels = num_labels
    bert.classifier = Linear(bert.config.hidden_size, num_labels)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(bert_tokenizer, _arch)

    batch = collator(examples)
    print(batch)
    out = bert(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_pt_bert_wic_forward(
    pt_bert: PeftModel,
    bert_tokenizer: PreTrainedTokenizerFast,
    wic: DatasetDict,
) -> None:
    with patch("icftsc.datasets.wic.load_dataset", return_value=wic):
        data, _ = load_wic(bert_tokenizer, _arch, 0)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(bert_tokenizer, _arch)

    batch = collator(examples)
    out = pt_bert(**batch)

    assert out.loss is not None
    assert out.logits is not None


@skip("sequence truncation based on number of virtual tokens in not implemented")
def test_pt_bert_boolq_forward(
    pt_bert: PeftModel,
    bert_tokenizer: PreTrainedTokenizerFast,
    boolq: DatasetDict,
) -> None:
    with patch("icftsc.datasets.boolq.load_dataset", return_value=boolq):
        data, _ = load_boolq(bert_tokenizer, _arch, 0)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(bert_tokenizer, _arch)

    batch = collator(examples)
    out = pt_bert(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_pt_bert_estner_forward(
    pt_bert: PeftModel,
    bert_tokenizer: PreTrainedTokenizerFast,
    estner: DatasetDict,
) -> None:
    with patch("icftsc.datasets.estner.load_dataset", return_value=estner):
        data, info = load_estner(bert_tokenizer, _arch, 0)

    num_labels = len(info["id2label"])
    bert = cast(BertForSequenceClassification, pt_bert.base_model)
    bert.num_labels = num_labels
    bert.classifier = Linear(pt_bert.config.hidden_size, num_labels)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(bert_tokenizer, _arch)

    batch = collator(examples)
    out = pt_bert(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_pt_bert_multinerd_forward(
    pt_bert: PeftModel,
    bert_tokenizer: PreTrainedTokenizerFast,
    multinerd: DatasetDict,
) -> None:
    with patch("icftsc.datasets.multinerd.load_dataset", return_value=multinerd):
        data, info = load_multinerd(bert_tokenizer, _arch, 0, False)

    num_labels = len(info["id2label"])
    bert = cast(BertForSequenceClassification, pt_bert.base_model)
    bert.num_labels = num_labels
    bert.classifier = Linear(pt_bert.config.hidden_size, num_labels)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(bert_tokenizer, _arch)

    batch = collator(examples)
    print(batch)
    out = pt_bert(**batch)

    assert out.loss is not None
    assert out.logits is not None
