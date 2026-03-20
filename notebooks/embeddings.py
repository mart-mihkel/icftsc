import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    from typing import cast

    from torch import Tensor
    from transformers import (
        AutoModel,
        AutoTokenizer,
        PreTrainedModel,
        PreTrainedTokenizer,
    )

    return (
        AutoModel,
        AutoTokenizer,
        PreTrainedModel,
        PreTrainedTokenizer,
        Tensor,
        cast,
    )


@app.cell
def _(PreTrainedModel, PreTrainedTokenizer, Tensor):
    def encode(
        text: str,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
    ) -> Tensor:
        prefix_tokenized = tokenizer(text, return_tensors="pt")
        token_ids = prefix_tokenized["input_ids"][0]
        return model.get_input_embeddings().forward(token_ids)

    return (encode,)


@app.cell
def _(PreTrainedModel, PreTrainedTokenizer, Tensor, cast):
    def decode(
        embeddings: Tensor,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
    ) -> str | list[str]:
        voc_embeddings = cast(Tensor, model.get_input_embeddings().weight)
        similarity = embeddings @ voc_embeddings.T
        token_ids = similarity.argmax(dim=1)
        return tokenizer.decode(token_ids=token_ids)

    return (decode,)


@app.cell
def _(AutoModel, AutoTokenizer):
    model_path = "hf-internal-testing/tiny-random-bert"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    return model, tokenizer


@app.cell
def _(decode, encode, model, tokenizer):
    embeddings = encode(
        text="I am a horse",
        model=model,
        tokenizer=tokenizer,
    )

    decoded = decode(
        embeddings=embeddings,
        model=model,
        tokenizer=tokenizer,
    )

    print(decoded)
    return


if __name__ == "__main__":
    app.run()
