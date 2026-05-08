from __future__ import annotations

from abc import ABC, abstractmethod


class Engine(ABC):
    """One model + one batch-invariance setup. Subclass per cell to benchmark."""

    name: str

    @abstractmethod
    def setup(self) -> None: ...

    @abstractmethod
    def generate(
        self,
        prompts: list[str],
        n: int,
        sampling: dict | None = None,
    ) -> list[list[str]]:
        """n completions per prompt, shape [len(prompts)][n]."""

    @abstractmethod
    def teardown(self) -> None: ...
