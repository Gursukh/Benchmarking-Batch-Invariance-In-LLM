from __future__ import annotations

from batch_invariance_bench.tasks.base import HFTask, Item


class IFEval(HFTask):
    """Google IFEval prompts. Scoring is handled downstream, not here."""

    name = "ifeval"
    hf_dataset = "google/IFEval"
    default_split = "train"

    def _to_item(self, row: dict, idx: int) -> Item:
        return Item(
            id=str(row["key"]),
            prompt=row["prompt"],
            reference={
                "instruction_id_list": list(row["instruction_id_list"]),
                "kwargs": list(row["kwargs"]),
            },
        )
