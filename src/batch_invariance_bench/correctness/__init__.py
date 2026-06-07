"""Correctness/accuracy harness.

run_accuracy and run_correctness are lazy so importing correctness.score does not
pull in the engines (and vLLM).
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from batch_invariance_bench.correctness.runner import (
        run_accuracy,
        run_correctness,
    )

__all__ = ["run_accuracy", "run_correctness"]

_LAZY = {
    "run_accuracy": ("batch_invariance_bench.correctness.runner", "run_accuracy"),
    "run_correctness": (
        "batch_invariance_bench.correctness.runner",
        "run_correctness",
    ),
}


def __getattr__(name: str):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module, attr = target
    return getattr(importlib.import_module(module), attr)
