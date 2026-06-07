"""Boxed-answer scoring: pull the last \\boxed{...}, join to references by
problem_id, check equivalence with math_verify. For MATH-500 / AIME; IFEval has
its own grader.
"""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd
from math_verify import parse as _mv_parse, verify as _mv_verify


_BOXED_PREFIXES = ("\\boxed", "\\fbox")


def extract_boxed(text: str) -> str | None:
    """Last \\boxed{...} or \\fbox{...}, handling nested braces."""
    if not isinstance(text, str) or not text:
        return None
    candidates: list[str] = []
    for prefix in _BOXED_PREFIXES:
        start = 0
        while True:
            i = text.find(prefix, start)
            if i < 0:
                break
            j = i + len(prefix)
            while j < len(text) and text[j] == " ":
                j += 1
            if j >= len(text) or text[j] != "{":
                m = re.match(r"\s*([^\s$]+)", text[j:])
                if m:
                    candidates.append(m.group(1))
                start = j
                continue
            depth = 0
            k = j
            content_start = j + 1
            while k < len(text):
                ch = text[k]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidates.append(text[content_start:k])
                        break
                k += 1
            start = k + 1
    if candidates:
        return candidates[-1].strip()
    # Fallback for "final answer: ..." without a box.
    m = re.search(
        r"(?:final answer|answer)\s*[:=]\s*\$?([^\n$]+?)\$?\s*$",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    return None


def is_correct(pred: str | None, ref: str | None) -> tuple[bool, str | None]:
    """(correct, error) via math_verify; error is None on success.

    Splitting them keeps parse failures out of the accuracy aggregate.
    """
    if pred is None or ref is None:
        return False, None
    try:
        gold = _mv_parse(f"${ref}$")
        guess = _mv_parse(f"${pred}$")
        return bool(_mv_verify(gold, guess)), None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def score_frame(df: pd.DataFrame, references: dict[str, str]) -> pd.DataFrame:
    """Add pred, reference, correct, score_error to a copy of df.

    Needs problem_id and completion_text; references maps problem_id to answer.
    """
    out = df.copy()
    out["pred"] = out["completion_text"].map(extract_boxed)
    out["reference"] = out["problem_id"].map(references)
    verdicts = [is_correct(p, r) for p, r in zip(out["pred"], out["reference"])]
    out["correct"] = [v[0] for v in verdicts]
    out["score_error"] = [v[1] if v[1] is not None else "" for v in verdicts]
    return out


def accuracy_table(
    scored: pd.DataFrame,
    by: Iterable[str] = ("gpu_name", "engine_label", "batch_size"),
) -> pd.DataFrame:
    by = list(by)
    g = scored.groupby(by)["correct"]
    return (
        g.agg(n="size", correct="sum", accuracy="mean")
        .reset_index()
        .sort_values(by)
        .reset_index(drop=True)
    )
