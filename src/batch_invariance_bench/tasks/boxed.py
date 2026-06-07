from __future__ import annotations

from batch_invariance_bench.tasks.base import BoxedTask, Item


MATH500_PROMPT = (
    "Solve the following math problem. Show your reasoning, then write the "
    "final answer inside \\boxed{{}}.\n\nProblem: {problem}"
)

AIME_PROMPT = (
    "Solve the following AIME problem. The answer is a non-negative integer. "
    "Show your reasoning, then write the final integer answer inside "
    "\\boxed{{}}.\n\nProblem: {problem}"
)


class MATH500(BoxedTask):
    name = "math500"
    hf_dataset = "HuggingFaceH4/MATH-500"
    default_split = "test"

    def _to_item(self, row: dict, idx: int) -> Item:
        return Item(
            id=str(row.get("unique_id", row.get("problem", "")[:64])),
            prompt=MATH500_PROMPT.format(problem=row["problem"]),
            reference=row["answer"],
        )


class AIME(BoxedTask):
    """AIME problems; answers are integers in [0, 999]."""

    name = "aime"
    hf_dataset = "Maxwell-Jia/AIME_2024"
    default_split = "train"

    problem_field = "Problem"
    answer_field = "Answer"
    id_field = "ID"

    def _to_item(self, row: dict, idx: int) -> Item:
        return Item(
            id=str(row.get(self.id_field, idx)),
            prompt=AIME_PROMPT.format(problem=row[self.problem_field]),
            reference=int(row[self.answer_field]),
        )
