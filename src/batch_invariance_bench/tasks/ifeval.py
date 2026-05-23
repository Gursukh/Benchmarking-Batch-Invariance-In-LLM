from __future__ import annotations

from typing import TYPE_CHECKING

from batch_invariance_bench.tasks.base import HFTask, Item

if TYPE_CHECKING:
    import pandas as pd


class IFEval(HFTask):
    """Google IFEval prompts and the instruction following grader.

    Each reference keeps the instruction id list and kwargs. score needs
    the instruction_following_eval package and raises if it is missing.
    """

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
                # Some instructions check against the original prompt.
                "prompt": row["prompt"],
            },
        )

    def score(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Score an IFEval frame. Adds correct, n_instr_passed, n_instr_total."""
        
        raise NotImplementedError("IFEval scoring not implemented yet")