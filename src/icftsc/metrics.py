import evaluate
import numpy as np
from scipy.special import log_softmax
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.fft import Tensor
from transformers import EvalPrediction, PreTrainedTokenizerFast

from icftsc.logging import logger

_bleu = evaluate.load("bleu")
_rouge = evaluate.load("rouge")

# global state during `trainer.evaluate` to collect batched eval loop outputs
_labels: list[np.ndarray] = []
_preds: list[np.ndarray] = []


def _batch_to_numpy(eval_pred: EvalPrediction) -> tuple[np.ndarray, np.ndarray]:
    batch_labels = eval_pred.label_ids
    if isinstance(batch_labels, tuple):
        batch_labels = batch_labels[0]

    if isinstance(batch_labels, Tensor):
        batch_labels = batch_labels.detach().cpu().numpy()

    batch_logits = eval_pred.predictions
    if isinstance(batch_logits, tuple):
        batch_logits = batch_logits[0]

    if isinstance(batch_logits, Tensor):
        batch_logits = batch_logits.detach().cpu().numpy()

    batch_preds = np.argmax(batch_logits, axis=-1)

    return batch_labels, batch_preds


def _compute_classification_metrics(
    labels: np.ndarray | list,
    preds: np.ndarray | list,
) -> dict[str, float]:
    logger.debug("compute classification metrics")

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="macro", zero_division=0)
    recall = recall_score(labels, preds, average="macro", zero_division=0)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def _compute_perplexity(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    logger.debug("compute perplexity")

    idx = np.arange(labels.shape[0])
    log_probs = log_softmax(logits, axis=-1)[idx, labels]
    perplexity = np.exp(-log_probs.mean())
    return {"perplexity": perplexity}


def _compute_bleu(
    references: list[str],
    predictions: list[str],
) -> dict[str, float]:
    logger.debug("compute BLEU")

    res = _bleu.compute(predictions=predictions, references=references)  # type: ignore
    if res is None:
        logger.warning("BLEU evaluation was run in a child process")
        return {}

    return {"bleu": res["bleu"]}


def _compute_rouge(
    references: list[str],
    predictions: list[str],
) -> dict[str, float]:
    logger.debug("compute ROUGE")

    res = _rouge.compute(  # type: ignore
        predictions=predictions,
        references=references,
        rouge_types=["rouge1", "rouge2"],
    )

    if res is None:
        logger.warning("ROUGE evaluation was run in a child process")
        return {}

    return res


def compute_metrics_seq_cls(
    eval_pred: EvalPrediction,
    compute_result: bool = True,
) -> dict[str, float]:
    global _labels, _preds

    labels, preds = _batch_to_numpy(eval_pred)
    _labels.extend(labels)
    _preds.extend(preds)

    if not compute_result:
        return {}

    all_labels = np.array(_labels)
    all_preds = np.array(_preds)
    mask = all_labels != -100

    _labels = []
    _preds = []

    return _compute_classification_metrics(all_labels[mask], all_preds[mask])


def compute_metrics_seq2seq(
    eval_pred: EvalPrediction,
    tokenizer: PreTrainedTokenizerFast,
    compute_result: bool = True,
) -> dict[str, float]:
    global _labels, _preds

    labels, preds = _batch_to_numpy(eval_pred)
    _labels.extend(labels)
    _preds.extend(preds)

    if not compute_result:
        return {}

    labels = []
    preds = []
    for label, pred in zip(_labels, _preds, strict=True):
        mask = label != -100
        labels.append(label[mask])
        preds.append(pred[mask])

    _labels = []
    _preds = []

    references = tokenizer.batch_decode(labels, skip_special_tokens=True)
    predictions = tokenizer.batch_decode(preds, skip_special_tokens=True)

    logger.info(predictions)

    return _compute_classification_metrics(references, predictions)


def compute_metrics_causal_lm(
    eval_pred: EvalPrediction,
    tokenizer: PreTrainedTokenizerFast,
    compute_result: bool = True,
) -> dict[str, float]:
    global _labels, _preds

    labels, preds = _batch_to_numpy(eval_pred)
    _labels.extend(labels)
    _preds.extend(preds)

    if not compute_result:
        return {}

    labels = []
    preds = []
    for label, pred in zip(_labels, _preds, strict=True):
        mask = label != -100
        labels.append(label[mask])
        preds.append(pred[mask])

    _labels = []
    _preds = []

    references = tokenizer.batch_decode(labels, skip_special_tokens=True)
    predictions = tokenizer.batch_decode(preds, skip_special_tokens=True)

    logger.info(predictions)

    return _compute_classification_metrics(references, predictions)
