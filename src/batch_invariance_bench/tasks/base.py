from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypedDict


class Item(TypedDict, total=False):
    id: str
    prompt: str
    reference: str | int | float | dict


class ScoreResult(TypedDict):
    correct: list[bool]
    extracted: list[str]


class Task(ABC):
    name: str

    @abstractmethod
    def load(self) -> list[Item]: ...

    @abstractmethod
    def score(self, item: Item, completions: list[str]) -> ScoreResult: ...
