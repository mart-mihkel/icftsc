import marimo

__generated_with = "0.19.7"
app = marimo.App()


@app.cell
def _():
    from typing import cast

    import marimo as mo
    import numpy as np
    from transformers import AutoTokenizer

    from icft.datasets.multinerd import Multinerd

    return AutoTokenizer, Multinerd, cast, mo, np


@app.cell
def _(AutoTokenizer):
    _pretrained_model = "jhu-clsp/mmBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(_pretrained_model)
    sep_token = tokenizer.special_tokens_map["sep_token"]
    cls_token = tokenizer.special_tokens_map["cls_token"]
    return cls_token, sep_token, tokenizer


@app.cell
def _(Multinerd, mo):
    print(Multinerd.SYSTEM_PROMPT)
    mo.md("MultiNERD system prompt")
    return


@app.cell
def _(Multinerd, cls_token, mo, sep_token, tokenizer):
    _tokens = tokenizer(
        f"{cls_token} {Multinerd.SYSTEM_PROMPT} {sep_token}",
        add_special_tokens=False,
    )

    _ids = _tokens["input_ids"]

    print(_ids)
    print()
    print(tokenizer.decode(_ids))

    mo.md("Tokenized")
    return


@app.cell
def _(Multinerd, cast, cls_token, mo, np, sep_token, tokenizer):
    _base_tokens = tokenizer(Multinerd.SYSTEM_PROMPT, add_special_tokens=False)

    _ids = cast(list[int], _base_tokens["input_ids"])
    _vocab_size = tokenizer.vocab_size
    _random_ids = np.random.randint(0, _vocab_size - 1, size=len(_ids))

    _cls_id = tokenizer.convert_tokens_to_ids(cls_token)
    _sep_id = tokenizer.convert_tokens_to_ids(sep_token)

    _ids = [_cls_id] + _random_ids.tolist() + [_sep_id]

    print(_ids)
    print()
    print(tokenizer.decode(_ids))

    mo.md("Random gibberish")
    return


@app.cell
def _(cls_token, mo, tokenizer):
    _tokens = tokenizer(cls_token, add_special_tokens=False)
    _ids = _tokens["input_ids"]

    print(_ids)
    print()
    print(tokenizer.decode(_ids))

    mo.md("No prompt")
    return


if __name__ == "__main__":
    app.run()
