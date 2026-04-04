REMOTE ?=

pre-commit:
	@uv run ruff check --fix
	@uv run ruff format
	@uv run ty check

test:
	@uv run pytest --quiet --numprocesses 4

sync:
	rsync --verbose --archive --delete \
		--exclude-from .gitignore \
		--exclude .pytest_cache \
		--exclude .ruff_cache \
		--exclude .git \
		. $(REMOTE)
