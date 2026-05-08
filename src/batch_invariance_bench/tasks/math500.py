from __future__ import annotations

import re

from datasets import load_dataset

from batch_invariance_bench.tasks.base import Item, ScoreResult, Task


PROMPT_TEMPLATE = (
    "Solve the following math problem. Show your reasoning, then write the "
    "final answer inside \\boxed{{}}.\n\nProblem: {problem}"
)


def _extract_boxed(text: str) -> str | None:
    """Contents of the last \\boxed{...}, brace-matched. None if absent."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    i = idx + len("\\boxed{")
    depth = 1
    out = []
    while i < len(text) and depth:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(c)
        i += 1
    return "".join(out) if depth == 0 else None


_NORMALIZE_SUBS = [
    (r"\\left", ""),
    (r"\\right", ""),
    (r"\\!", ""),
    (r"\\,", ""),
    (r"\\;", ""),
    (r"\\:", ""),
    (r"\\\\", ""),
    (r"\\dfrac", r"\\frac"),
    (r"\\tfrac", r"\\frac"),
    (r"\s+", ""),
]


def _normalize(s: str) -> str:
    s = s.strip()
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1]
    for pat, repl in _NORMALIZE_SUBS:
        s = re.sub(pat, repl, s)
    return s.rstrip(".")


def _equal(a: str, b: str) -> bool:
    if _normalize(a) == _normalize(b):
        return True
    try:
        return float(a) == float(b)
    except (ValueError, TypeError):
        return False


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

    def score(self, item: Item, completions: list[str]) -> ScoreResult:
        gold = str(item["reference"])
        extracted = [_extract_boxed(c) or "" for c in completions]
        correct = [_equal(e, gold) if e else False for e in extracted]
        return ScoreResult(correct=correct, extracted=extracted)
