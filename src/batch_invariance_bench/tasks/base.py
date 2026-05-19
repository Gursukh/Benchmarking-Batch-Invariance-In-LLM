from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypedDict

from datasets import load_dataset

if TYPE_CHECKING:
    import pandas as pd


class Item(TypedDict, total=False):
    id: str
    prompt: str
    reference: str | int | float | dict | list


class Task(ABC):
    name: str

    @abstractmethod
    def load(self) -> list[Item]:
        """Return the task's prompts as a list of Items."""

    def score(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Score a results frame, returning it with verdict columns added.

        Optional. Tasks scored downstream (like IFEval) raise NotImplementedError.
        """
        raise NotImplementedError(f"task {self.name!r} has no built-in scorer")


class HFTask(Task):
    """A Task backed by a Hugging Face dataset.

    Subclasses set hf_dataset and default_split and implement _to_item(). This
    base handles load_dataset, the optional limit, and building the Items.
    """

    hf_dataset: str
    default_split: str = "test"

    def __init__(self, split: str | None = None, limit: int | None = None) -> None:
        self.split = split or self.default_split
        self.limit = limit

    def load(self) -> list[Item]:
        ds = load_dataset(self.hf_dataset, split=self.split)
        if self.limit:
            ds = ds.select(range(min(self.limit, len(ds))))
        return [self._to_item(row, i) for i, row in enumerate(ds)]

    @abstractmethod
    def _to_item(self, row: dict, idx: int) -> Item:
        """Build one Item from a dataset row."""
