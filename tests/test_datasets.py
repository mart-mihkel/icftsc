from transformers import PreTrainedTokenizerFast

from icftsc.datasets.common import DataCollatorWithPaddingAndLabels


def test_collator_with_labels(gpt2_tokenizer: PreTrainedTokenizerFast):
    features = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [10, 20]},
        {"input_ids": [4, 5], "attention_mask": [1, 1], "labels": [30]},
    ]

    collator = DataCollatorWithPaddingAndLabels(
        tokenizer=gpt2_tokenizer,
        pad_to_multiple_of=8,
    )

    batch = collator(features)

    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "labels" in batch

    assert batch["input_ids"].shape[1] % 8 == 0
    assert batch["labels"].shape[1] % 8 == 0
