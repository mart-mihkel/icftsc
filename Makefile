HF_CACHE = ~/.cache/huggingface
DOC_VIEWER = zathura
REMOTE_HOST ?=
REMOTE_DIR ?=

install:
	uv sync
	uv pip install --editable .

pre-commit:
	uv run ruff check --fix
	uv run ruff format
	uv run ty check

marimo:
	uv run marimo edit notebooks

trackio:
	uv run trackio show --project icft

typst:
	typst watch typesetting/main.typ --open $(DOC_VIEWER)

upload:
	rsync -rv --exclude-from '.gitignore' . $(REMOTE_HOST):$(REMOTE_DIR)

download-out:
	rsync -rv $(REMOTE_HOST):$(REMOTE_DIR)/out .

download-log:
	rsync -rv $(REMOTE_HOST):$(HF_CACHE)/trackio $(HF_CACHE)
