from __future__ import annotations

import re

from datasets import load_dataset

from batch_invariance_bench.tasks.base import Item, ScoreResult, Task
from batch_invariance_bench.tasks.math500 import _extract_boxed


PROMPT_TEMPLATE = (
    "Solve the following AIME problem. The answer is a non-negative integer. "
    "Show your reasoning, then write the final integer answer inside "
    "\\boxed{{}}.\n\nProblem: {problem}"
)


def _extract_int(text: str) -> int | None:
    boxed = _extract_boxed(text)
    candidate = boxed if boxed is not None else text
    m = re.findall(r"-?\d+", candidate)
    if not m:
        return None
    try:
        return int(m[-1])
    except ValueError:
        return None


class AIME(Task):
    """AIME problems — integer answers in [0, 999]."""

    name = "aime"

    def __init__(
        self,
        hf_dataset: str = "Maxwell-Jia/AIME_2024",
        split: str = "train",
        problem_field: str = "Problem",
        answer_field: str = "Answer",
        id_field: str = "ID",
        limit: int | None = None,
    ) -> None:
        self._hf_dataset = hf_dataset
        self._split = split
        self._problem_field = problem_field
        self._answer_field = answer_field
        self._id_field = id_field
        self._limit = limit

    def load(self) -> list[Item]:
        ds = load_dataset(self._hf_dataset, split=self._split)
        if self._limit:
            ds = ds.select(range(min(self._limit, len(ds))))
        items: list[Item] = []
        for i, row in enumerate(ds):
            items.append(
                Item(
                    id=str(row.get(self._id_field, i)),
                    prompt=PROMPT_TEMPLATE.format(problem=row[self._problem_field]),
                    reference=int(row[self._answer_field]),
                )
            )
        return items

    def score(self, item: Item, completions: list[str]) -> ScoreResult:
        gold = int(item["reference"])
        extracted_ints = [_extract_int(c) for c in completions]
        extracted = ["" if x is None else str(x) for x in extracted_ints]
        correct = [x == gold for x in extracted_ints]
        return ScoreResult(correct=correct, extracted=extracted)
