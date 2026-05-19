"""Set environment variables for a while, then put them back.

vLLM reads some settings from os.environ when an engine is built. These helpers
let an engine set them in setup() and undo it in teardown().
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator, Mapping


def apply_env(updates: Mapping[str, str]) -> dict[str, str | None]:
    """Set the given env vars. Returns the old values for restore_env()."""
    prev: dict[str, str | None] = {}
    for key, value in updates.items():
        prev[key] = os.environ.get(key)
        os.environ[key] = value
    return prev


def restore_env(prev: Mapping[str, str | None]) -> None:
    """Undo apply_env() using the values it returned."""
    for key, old in prev.items():
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


@contextmanager
def scoped_environ(updates: Mapping[str, str]) -> Iterator[None]:
    """Context-manager form of apply_env / restore_env."""
    prev = apply_env(updates)
    try:
        yield
    finally:
        restore_env(prev)
