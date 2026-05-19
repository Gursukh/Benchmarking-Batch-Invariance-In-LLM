"""Boxed-answer scoring for the correctness harness.

Reads a results CSV, pulls the answer from the last \\boxed{...} in
completion_text, joins it to per-problem references by problem_id, and checks
equivalence with math_verify.

Works for any task with a \\boxed{} answer (MATH-500, AIME). The caller passes
references; load_references() is a default for the MATH-500 split.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import pandas as pd
from math_verify import parse as _mv_parse, verify as _mv_verify


# answer extraction

_BOXED_PREFIXES = ("\\boxed", "\\fbox")


def extract_boxed(text: str) -> str | None:
    """Return the contents of the last \\boxed{...} or \\fbox{...}.

    Brace matching keeps nested braces, so \\boxed{\\frac{1}{2}} gives \\frac{1}{2}.
    """
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
    # Fall back to "final answer: ..." style completions with no box.
    m = re.search(
        r"(?:final answer|answer)\s*[:=]\s*\$?([^\n$]+?)\$?\s*$",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    return None


# equivalence


def is_correct(pred: str | None, ref: str | None) -> bool:
    """True if pred and ref are equivalent per math_verify."""
    if pred is None or ref is None:
        return False
    try:
        gold = _mv_parse(f"${ref}$")
        guess = _mv_parse(f"${pred}$")
        return bool(_mv_verify(gold, guess))
    except Exception:
        return False


# references


@lru_cache(maxsize=4)
def load_references(split: str = "test") -> dict[str, str]:
    """Map of unique_id to answer from HuggingFaceH4/MATH-500.

    A default for MATH-500. For other tasks pass Task.references() instead.
    """
    from datasets import load_dataset  # heavy, import only when needed

    ds = load_dataset("HuggingFaceH4/MATH-500", split=split)
    return {str(row["unique_id"]): str(row["answer"]) for row in ds}


# engine labels

_ENGINE_LABELS = (
    ("TMBatchInvariant", "TM"),
    ("Fxpr", "FXPR"),
    ("FXPR", "FXPR"),
    ("Default", "Default"),
)


def engine_label(engine: str) -> str:
    """Short label for an engine key, e.g. VLLMTMBatchInvariant gives TM."""
    for needle, label in _ENGINE_LABELS:
        if needle in engine:
            return label
    return engine


# scoring entry points


def score_frame(
    df: pd.DataFrame,
    references: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Return a copy of df with pred, reference and correct columns.

    Needs problem_id and completion_text columns. references defaults to the
    MATH-500 references.
    """
    if references is None:
        references = load_references()
    out = df.copy()
    out["pred"] = out["completion_text"].map(extract_boxed)
    out["reference"] = out["problem_id"].map(references)
    out["correct"] = [is_correct(p, r) for p, r in zip(out["pred"], out["reference"])]
    return out


def score_csv(
    path: str | Path,
    references: dict[str, str] | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    return score_frame(df, references)


def score_dir(
    results_dir: str | Path,
    *,
    pattern: str = "*.math500.csv",
    references: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Score every matching CSV in a directory into one combined frame.

    Adds engine_label and source columns; gpu_name, engine, task and run_id are
    already in the CSV. references defaults to MATH-500, so the default pattern
    only matches MATH-500 files. Pass both to score another task.
    """
    if references is None:
        references = load_references()
    frames = []
    for csv in sorted(Path(results_dir).glob(pattern)):
        scored = score_csv(csv, references)
        scored["engine_label"] = scored["engine"].map(engine_label)
        scored["source"] = csv.name
        frames.append(scored)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


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
