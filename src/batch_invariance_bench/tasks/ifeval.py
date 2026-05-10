from __future__ import annotations

from datasets import load_dataset

from batch_invariance_bench.tasks.base import Item, Task


class IFEval(Task):
    """Google IFEval prompts. Scoring is handled downstream, not here."""

    name = "ifeval"

    def __init__(self, split: str = "train", limit: int | None = None) -> None:
        self._split = split
        self._limit = limit

    def load(self) -> list[Item]:
        ds = load_dataset("google/IFEval", split=self._split)
        if self._limit:
            ds = ds.select(range(min(self._limit, len(ds))))
        items: list[Item] = []
        for row in ds:
            items.append(
                Item(
                    id=str(row["key"]),
                    prompt=row["prompt"],
                    reference={
                        "instruction_id_list": list(row["instruction_id_list"]),
                        "kwargs": list(row["kwargs"]),
                    },
                )
            )
        return items
