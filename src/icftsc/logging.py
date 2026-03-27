import logging

import accelerate
import datasets
import evaluate
import httpx
import numpy
import polars
import torch
import transformers
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install

_suppress = [
    transformers,
    accelerate,
    datasets,
    evaluate,
    polars,
    torch,
    httpx,
    numpy,
]

_console = Console()
_handler = RichHandler(
    show_path=False,
    console=_console,
    rich_tracebacks=True,
    tracebacks_show_locals=True,
    tracebacks_suppress=_suppress,
)

install(show_locals=True, console=_console, suppress=_suppress)

logging.basicConfig(format="%(message)s", handlers=[_handler])
logger = logging.getLogger("icftsc")
