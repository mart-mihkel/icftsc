import marimo

__generated_with = "0.19.7"
app = marimo.App()


@app.cell
def _():
    from transformers import AutoModelForSequenceClassification

    return (AutoModelForSequenceClassification,)


@app.cell
def _(AutoModelForSequenceClassification):
    model, info = AutoModelForSequenceClassification.from_pretrained(
        "distilbert/distilbert-base-uncased",
        output_loading_info=True,
    )
    return info, model


@app.cell
def _(info, model):
    for name, param in model.named_parameters():
        param.requires_grad = name in info["missing_keys"]
    return


@app.cell
def _(model):
    _total = sum(p.numel() for p in model.parameters())
    _trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"total params:     {_total / 1e6:.2f}M")
    print(f"trainable params: {_trainable / 1e6:.2f}M")
    return


if __name__ == "__main__":
    app.run()
