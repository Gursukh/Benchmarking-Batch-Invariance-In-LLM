"""MATH-500 scoring utilities.

Pipeline: read a results CSV, extract the predicted answer from the last
`\\boxed{...}` in `completion_text`, join against the HF MATH-500 references
on `problem_id` (= `unique_id`), and check symbolic equivalence with
`math_verify`.

`math_verify` is a hard dependency — import fails fast if it's missing.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import pandas as pd

from math_verify import parse as _mv_parse, verify as _mv_verify


# ---------------------------------------------------------------------------
# answer extraction
# ---------------------------------------------------------------------------

_BOXED_PREFIXES = ("\\boxed", "\\fbox")


def extract_boxed(text: str) -> str | None:
    """Return the contents of the *last* `\\boxed{...}` (or `\\fbox{...}`).

    Uses brace matching, so nested `{}` are preserved
    (`\\boxed{\\frac{1}{2}}` -> `\\frac{1}{2}`).
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
    # Final-answer-style completions with no box.
    m = re.search(
        r"(?:final answer|answer)\s*[:=]\s*\$?([^\n$]+?)\$?\s*$",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# equivalence
# ---------------------------------------------------------------------------


def is_correct(pred: str | None, ref: str | None) -> bool:
    """True iff `pred` is symbolically equivalent to `ref` per math_verify."""
    if pred is None or ref is None:
        return False
    try:
        gold = _mv_parse(f"${ref}$")
        guess = _mv_parse(f"${pred}$")
        return bool(_mv_verify(gold, guess))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# references
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4)
def load_references(split: str = "test") -> dict[str, str]:
    """Return `{unique_id: answer}` from HuggingFaceH4/MATH-500."""
    from datasets import load_dataset  # local import: heavy

    ds = load_dataset("HuggingFaceH4/MATH-500", split=split)
    return {str(row["unique_id"]): str(row["answer"]) for row in ds}


# ---------------------------------------------------------------------------
# scoring entry points
# ---------------------------------------------------------------------------


def score_frame(
    df: pd.DataFrame,
    references: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Return a copy of `df` with `pred`, `reference`, `correct` columns.

    Expects columns `problem_id` and `completion_text`.
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
) -> pd.DataFrame:
    """Score every matching CSV in a directory and return one concatenated frame.

    Adds `gpu`, `engine_slug`, `engine_label`, and `source` columns so accuracy
    can be sliced per configuration.
    """
    refs = load_references()
    frames = []
    for csv in sorted(Path(results_dir).glob(pattern)):
        scored = score_csv(csv, refs)
        gpu, engine_slug, task = _parse_name(csv.name)
        scored["gpu"] = gpu
        scored["engine_slug"] = engine_slug
        scored["engine_label"] = _short_engine(engine_slug)
        scored["task"] = task
        scored["source"] = csv.name
        frames.append(scored)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def accuracy_table(scored: pd.DataFrame, by: Iterable[str] = ("gpu", "engine_label", "batch_size")) -> pd.DataFrame:
    by = list(by)
    g = scored.groupby(by)["correct"]
    return (
        g.agg(n="size", correct="sum", accuracy="mean")
        .reset_index()
        .sort_values(by)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# tiny helpers
# ---------------------------------------------------------------------------


def _parse_name(fname: str) -> tuple[str, str, str]:
    stem = fname.rsplit(".csv", 1)[0]
    parts = stem.split(".")
    return parts[0], ".".join(parts[1:-1]), parts[-1]


def _short_engine(slug: str) -> str:
    if "TMBatchInvariant" in slug:
        return "TM"
    if "Fxpr" in slug or "FXPR" in slug:
        return "FXPR"
    if "Default" in slug:
        return "Default"
    return slug
