from transformers import DataCollatorWithPadding, PreTrainedTokenizerFast

from icftsc.modeling.pt import PTModelConfig
from icftsc.modeling.seqcls import (
    PTBertForSequenceClassification,
    PTGPTForSequenceClassification,
    PTT5ForSequenceClassification,
)


def test_pt_mmbert_seqcls(mmbert_tokenizer: PreTrainedTokenizerFast):
    cls = mmbert_tokenizer.cls_token_id
    data = [
        {"input_ids": [cls, 1, 2], "label": 0},
        {"input_ids": [cls, 3], "label": 1},
    ]

    collate_fn = DataCollatorWithPadding(
        tokenizer=mmbert_tokenizer,
        pad_to_multiple_of=8,
    )

    config = PTModelConfig(
        pretrained_model="jhu-clsp/mmBERT-base",
        num_virtual_tokens=10,
        task="seqcls",
        num_labels=2,
        id2label={0: "0", 1: "1"},
        label2id={"0": 0, "1": 1},
    )

    model = PTBertForSequenceClassification(config=config)
    out = model(**collate_fn(data))

    assert out.loss is not None
    assert out.logits is not None
    assert out.logits.shape == (2, 2)


def test_pt_gpt2_seqcls(gpt2_tokenizer: PreTrainedTokenizerFast):
    data = [
        {"input_ids": [1, 2], "label": 0},
        {"input_ids": [3], "label": 1},
    ]

    collate_fn = DataCollatorWithPadding(
        tokenizer=gpt2_tokenizer,
        pad_to_multiple_of=8,
    )

    config = PTModelConfig(
        pretrained_model="openai-community/gpt2",
        num_virtual_tokens=10,
        task="seqcls",
        num_labels=2,
        id2label={0: "0", 1: "1"},
        label2id={"0": 0, "1": 1},
    )

    model = PTGPTForSequenceClassification(config=config)
    model.base.config.pad_token_id = gpt2_tokenizer.eos_token_id
    out = model(**collate_fn(data))

    assert out.loss is not None
    assert out.logits is not None
    assert out.logits.shape == (2, 2)


def test_pt_t5_seqcls(t5_tokenizer: PreTrainedTokenizerFast):
    data = [
        {"input_ids": [1, 2], "label": 0},
        {"input_ids": [3], "label": 1},
    ]

    collate_fn = DataCollatorWithPadding(
        tokenizer=t5_tokenizer,
        pad_to_multiple_of=8,
    )

    config = PTModelConfig(
        pretrained_model="google-t5/t5-small",
        num_virtual_tokens=10,
        task="seqcls",
        num_labels=2,
        id2label={0: "0", 1: "1"},
        label2id={"0": 0, "1": 1},
    )

    model = PTT5ForSequenceClassification(config=config)
    model.base.config.pad_token_id = t5_tokenizer.eos_token_id
    out = model(**collate_fn(data))

    assert out.loss is not None
    assert out.logits is not None
    assert out.logits.shape == (2, 2)
