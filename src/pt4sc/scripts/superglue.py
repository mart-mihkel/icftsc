import json
from pathlib import Path
from typing import cast

import numpy as np
import torch
from datasets.splits import Split
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoModel, DataCollatorWithPadding, Trainer

from pt4sc.datasets.superglue import id2label
from pt4sc.logging import logger
from pt4sc.scripts.common import init_data, init_tokenizer


def _run_predict(trainer: Trainer, data: Dataset, out: Path):
    res = trainer.predict(data)
    preds = np.argmax(res.predictions, axis=-1)

    jsonl = [
        json.dumps({"idx": idx, "label": id2label[pred]}) + "\n"
        for idx, pred in zip(data["idx"], preds, strict=True)
    ]

    logger.info("save predictions to %s", out)
    with open(out, "w") as f:
        f.writelines(jsonl)


def predict(checkpoint: str):
    path = Path(checkpoint)
    with open(path / "cli_params.json") as f:
        params = json.load(f)

    config = AutoConfig.from_pretrained(checkpoint)
    tokenizer = init_tokenizer(model_path=checkpoint)
    data, _ = init_data(
        model_type=config.model_type,
        tokenizer=tokenizer,
        dataset="superglue-boolq",
        workers=params["workers"],
        split=cast(Split, {"test": "test"}),
    )

    logger.debug("load model from checkpoint")
    model = AutoModel.from_pretrained(checkpoint)

    logger.debug("load trainer from checkpoint")
    args = torch.load(path / "training_args.bin", weights_only=False)
    args.eval_strategy = "no"

    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    trainer = Trainer(args=args, model=model, data_collator=collate_fn)

    logger.debug("run no prompt predictions")
    _run_predict(
        trainer=trainer,
        data=cast(Dataset, data["test"].remove_columns("labels")),
        out=path / "boolq.jsonl",
    )

    logger.debug("run system prompt predictions")
    _run_predict(
        trainer=trainer,
        data=cast(Dataset, data["test-system"].remove_columns("labels")),
        out=path / "boolq-system.jsonl",
    )

    logger.debug("run random system prompt predictions")
    _run_predict(
        trainer=trainer,
        data=cast(Dataset, data["test-random"].remove_columns("labels")),
        out=path / "boolq-random.jsonl",
    )
