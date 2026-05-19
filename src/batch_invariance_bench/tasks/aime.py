from __future__ import annotations

from typing import TYPE_CHECKING

from batch_invariance_bench.tasks.base import HFTask, Item

if TYPE_CHECKING:
    import pandas as pd


PROMPT_TEMPLATE = (
    "Solve the following AIME problem. The answer is a non-negative integer. "
    "Show your reasoning, then write the final integer answer inside "
    "\\boxed{{}}.\n\nProblem: {problem}"
)


class AIME(HFTask):
    """AIME problems. Answers are non-negative integers in [0, 999]."""

    name = "aime"
    hf_dataset = "Maxwell-Jia/AIME_2024"
    default_split = "train"

    # Dataset field names; override in a subclass for a different source.
    problem_field = "Problem"
    answer_field = "Answer"
    id_field = "ID"

    def _to_item(self, row: dict, idx: int) -> Item:
        return Item(
            id=str(row.get(self.id_field, idx)),
            prompt=PROMPT_TEMPLATE.format(problem=row[self.problem_field]),
            reference=int(row[self.answer_field]),
        )

    def score(self, df: "pd.DataFrame") -> "pd.DataFrame":
        # AIME answers are integers, so the boxed-answer scorer handles them.
        from batch_invariance_bench.correctness.score import score_frame

        return score_frame(df, references=self.references())

    def references(self) -> dict[str, str]:
        """Map of problem_id to answer for this split."""
        return {it["id"]: str(it["reference"]) for it in self.load()}
