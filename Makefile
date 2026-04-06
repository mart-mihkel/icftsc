REMOTE ?=
MAX_JOBS=4
BACKEND=cpu

install:
	@MAX_JOBS=$(MAX_JOBS) uv sync --compile-bytecode --extra $(BACKEND)

check:
	@uv run --no-sync ruff check --fix
	@uv run --no-sync ruff format
	@uv run --no-sync ty check

test:
	@uv run --no-sync pytest --quiet --numprocesses 4

sync:
	rsync --verbose --archive --delete \
		--exclude-from .gitignore \
		--exclude .pytest_cache \
		--exclude .ruff_cache \
		--exclude .git \
		. $(REMOTE)
