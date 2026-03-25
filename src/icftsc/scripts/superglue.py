import json
from pathlib import Path
from typing import cast

import numpy as np
import torch
from datasets.splits import Split
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoModel, DataCollatorWithPadding, Trainer

from icftsc.datasets.boolq import id2label
from icftsc.datasets.common import load_tokenizer
from icftsc.logging import logger
from icftsc.scripts.common import load_data


def predict_boolq(checkpoint: str):
    path = Path(checkpoint)
    with open(path / "cli_params.json") as f:
        params = json.load(f)

    config = AutoConfig.from_pretrained(checkpoint)
    tokenizer = load_tokenizer(model_path=checkpoint)
    data, _ = load_data(
        tokenizer=tokenizer,
        dataset="boolq",
        model_type=config.model_type,
        task=params["task"],
        n_shot=params["n_shot"],
        split=cast(Split, {"test": "test"}),
    )

    logger.info("load model from checkpoint")
    model = AutoModel.from_pretrained(checkpoint)

    logger.info("load trainer from checkpoint")
    args = torch.load(path / "training_args.bin", weights_only=False)
    args.eval_strategy = "no"

    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    trainer = Trainer(args=args, model=model, data_collator=collate_fn)

    logger.info("run predictions")
    res = trainer.predict(cast(Dataset, data["test"].remove_columns("labels")))
    preds = np.argmax(res.predictions, axis=-1)

    jsonl = [
        json.dumps({"idx": idx, "label": id2label[pred]}) + "\n"
        for idx, pred in zip(data["idx"], preds, strict=True)
    ]

    logger.info("save predictions to %s", path)
    with open(path, "w") as f:
        f.writelines(jsonl)
