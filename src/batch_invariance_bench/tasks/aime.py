from __future__ import annotations

from datasets import load_dataset

from batch_invariance_bench.tasks.base import Item, Task


PROMPT_TEMPLATE = (
    "Solve the following AIME problem. The answer is a non-negative integer. "
    "Show your reasoning, then write the final integer answer inside "
    "\\boxed{{}}.\n\nProblem: {problem}"
)


class AIME(Task):
    """AIME problems. Answers are non-negative integers in [0, 999]."""

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
