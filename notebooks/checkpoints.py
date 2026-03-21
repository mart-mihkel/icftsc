import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import torch
    from torch.nn import Parameter
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    from icftsc.datasets.common import DatasetInfo
    from icftsc.scripts.common import init_pt_model

    return (
        AutoConfig,
        AutoModel,
        AutoTokenizer,
        DatasetInfo,
        Parameter,
        init_pt_model,
        torch,
    )


@app.cell
def _():
    path = "/tmp/icftsc-test"
    prefix_init = "pretrained"
    model_path = "hf-internal-testing/tiny-random-bert"
    return model_path, path, prefix_init


@app.cell
def _(AutoTokenizer, DatasetInfo, init_pt_model, model_path, prefix_init):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    _data_info = DatasetInfo(
        id2label={0: "0", 1: "1"}, label2id={"0": 0, "1": 1}, system_prompt="Test"
    )

    model = init_pt_model(
        model_path=model_path,
        tokenizer=tokenizer,
        prefix_init=prefix_init,
        data_info=_data_info,
    )
    return (model,)


@app.cell
def _(Parameter, model, path, torch):
    model.prefix = Parameter(torch.ones_like(model.prefix))
    model.save_pretrained(path)
    return


@app.cell
def _(AutoConfig, AutoModel, path):
    config = AutoConfig.from_pretrained(path)
    loaded_model = AutoModel.from_pretrained(path, config=config)
    loaded_model.prefix
    return


if __name__ == "__main__":
    app.run()
