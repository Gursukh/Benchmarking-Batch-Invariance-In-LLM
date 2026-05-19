from __future__ import annotations

from typing import TYPE_CHECKING

from batch_invariance_bench.tasks.base import HFTask, Item

if TYPE_CHECKING:
    import pandas as pd


PROMPT_TEMPLATE = (
    "Solve the following math problem. Show your reasoning, then write the "
    "final answer inside \\boxed{{}}.\n\nProblem: {problem}"
)


class MATH500(HFTask):
    name = "math500"
    hf_dataset = "HuggingFaceH4/MATH-500"
    default_split = "test"

    def _to_item(self, row: dict, idx: int) -> Item:
        return Item(
            id=str(row.get("unique_id", row.get("problem", "")[:64])),
            prompt=PROMPT_TEMPLATE.format(problem=row["problem"]),
            reference=row["answer"],
        )

    def score(self, df: "pd.DataFrame") -> "pd.DataFrame":
        from batch_invariance_bench.correctness.score import score_frame

        return score_frame(df, references=self.references())

    def references(self) -> dict[str, str]:
        """Map of problem_id to answer for this split."""
        return {it["id"]: str(it["reference"]) for it in self.load()}
