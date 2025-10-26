import marimo

__generated_with = "0.20.2"
app = marimo.App()


@app.cell
def _():
    from datasets import load_dataset

    return (load_dataset,)


@app.cell
def _(load_dataset):
    (data,) = load_dataset("Babelscape/multinerd", split=["train[:1]"])
    return (data,)


@app.cell
def _(data):
    data[0]
    return


if __name__ == "__main__":
    app.run()
