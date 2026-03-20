import numpy as np
from transformers import EvalPrediction, PreTrainedTokenizerFast

from pt4sc.metrics import (
    compute_metrics_causal_lm,
    compute_metrics_seq2seq,
    compute_metrics_seq_cls,
)


def test_seq_cls():
    logits = np.array([[2.0, 1.0, 0.0], [2.0, 1.0, 0.0]])
    labels = np.array([0, 0])
    eval_pred = EvalPrediction(logits, labels)
    metrics = compute_metrics_seq_cls(eval_pred)

    assert metrics["accuracy"] == 1.0


def test_seq2seq(t5_tokenizer: PreTrainedTokenizerFast):
    logits = np.array(
        [
            [[1.0, 4.0, 3.0, 2.0, 5.0], [1.0, 4.0, 3.0, 2.0, 5.0]],
            [[1.0, 4.0, 3.0, 2.0, 5.0], [1.0, 4.0, 3.0, 2.0, 5.0]],
        ]
    )

    labels = np.array([[5, 5], [5, 5]])
    eval_pred = EvalPrediction(logits, labels)
    compute_metrics_seq2seq(eval_pred, tokenizer=t5_tokenizer)


def test_causal_lm(gpt2_tokenizer: PreTrainedTokenizerFast):
    logits = np.array(
        [
            [[5.0, 4.0, 3.0, 2.0, 1.0], [5.0, 4.0, 3.0, 2.0, 1.0]],
            [[5.0, 4.0, 3.0, 2.0, 1.0], [5.0, 4.0, 3.0, 2.0, 1.0]],
        ]
    )

    labels = np.array([[0, 1], [0, 1]])
    eval_pred = EvalPrediction(logits, labels)

    compute_metrics_causal_lm(eval_pred, tokenizer=gpt2_tokenizer)
