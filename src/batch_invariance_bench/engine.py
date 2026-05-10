from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Sample:
    """One generated completion. logprobs[i] is the chosen token's logprob at step i."""

    text: str
    token_ids: list[int]
    logprobs: list[float]
    n_prompt_tokens: int
    n_output_tokens: int
    finish_reason: str
    stop_reason: str


class Engine(ABC):
    """A model + a particular batch-invariance configuration."""

    name: str

    @abstractmethod
    def setup(self) -> None: ...

    @abstractmethod
    def generate(
        self,
        prompts: list[str],
        n: int,
        sampling: dict | None = None,
    ) -> list[list[Sample]]:
        """Returns n samples for each prompt, indexed [prompt_idx][sample_idx]."""

    @abstractmethod
    def teardown(self) -> None: ...
