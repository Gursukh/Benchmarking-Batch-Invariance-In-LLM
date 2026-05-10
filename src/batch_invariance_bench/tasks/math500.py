from __future__ import annotations

from datasets import load_dataset

from batch_invariance_bench.tasks.base import Item, Task


PROMPT_TEMPLATE = (
    "Solve the following math problem. Show your reasoning, then write the "
    "final answer inside \\boxed{{}}.\n\nProblem: {problem}"
)


class MATH500(Task):
    name = "math500"

    def __init__(self, split: str = "test", limit: int | None = None) -> None:
        self._split = split
        self._limit = limit

    def load(self) -> list[Item]:
        ds = load_dataset("HuggingFaceH4/MATH-500", split=self._split)
        if self._limit:
            ds = ds.select(range(min(self._limit, len(ds))))
        items: list[Item] = []
        for row in ds:
            items.append(
                Item(
                    id=str(row.get("unique_id", row.get("problem")[:64])),
                    prompt=PROMPT_TEMPLATE.format(problem=row["problem"]),
                    reference=row["answer"],
                )
            )
        return items
