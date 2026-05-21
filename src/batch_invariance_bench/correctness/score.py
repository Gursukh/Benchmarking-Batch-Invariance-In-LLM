"""Boxed-answer scoring for the correctness harness.

Reads a results CSV, pulls the answer from the last \\boxed{...} in
completion_text, joins to references by problem_id, and checks equivalence
with math_verify. Works for any task with a \\boxed{} answer (MATH-500, AIME).
IFEval uses a separate scorer (score_ifeval).
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import pandas as pd
from math_verify import parse as _mv_parse, verify as _mv_verify


_BOXED_PREFIXES = ("\\boxed", "\\fbox")


def extract_boxed(text: str) -> str | None:
    """Contents of the last \\boxed{...} or \\fbox{...}, with nested braces."""
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
    # Fallback for "final answer: ..." completions without a box.
    m = re.search(
        r"(?:final answer|answer)\s*[:=]\s*\$?([^\n$]+?)\$?\s*$",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    return None


def is_correct(pred: str | None, ref: str | None) -> tuple[bool, str | None]:
    """Check equivalence via math_verify.

    Returns (correct, error). error is None on success, else a short string;
    splitting these keeps parse failures out of the accuracy aggregate.
    """
    if pred is None or ref is None:
        return False, None
    try:
        gold = _mv_parse(f"${ref}$")
        guess = _mv_parse(f"${pred}$")
        return bool(_mv_verify(gold, guess)), None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


@lru_cache(maxsize=4)
def load_references(task: str = "math500", split: str = "test") -> dict[str, str]:
    """Map of problem_id to reference. Supported: math500, aime."""
    from datasets import load_dataset  # heavy, lazy

    key = task.lower()
    if key == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split=split)
        return {str(row["unique_id"]): str(row["answer"]) for row in ds}
    if key == "aime":
        # Reuse the AIME task so references stay consistent with generation.
        from batch_invariance_bench.tasks.aime import AIME

        items = AIME().load()
        return {str(it["id"]): str(it["reference"]) for it in items}
    raise ValueError(
        f"no built-in references for task {task!r}; pass references= explicitly"
    )


def score_frame(
    df: pd.DataFrame,
    references: dict[str, str] | None = None,
    *,
    task: str | None = None,
) -> pd.DataFrame:
    """Add pred, reference, correct, score_error columns to a copy of df.

    Needs problem_id and completion_text. Either references or task must be
    given; if df has a task column it must be uniform and match.
    """
    if references is None:
        if task is None:
            if "task" in df.columns:
                unique = df["task"].dropna().unique().tolist()
                if len(unique) != 1:
                    raise ValueError(
                        f"score_frame: df.task has {len(unique)} unique values "
                        f"{unique!r}; pass task= explicitly"
                    )
                task = str(unique[0])
            else:
                raise ValueError(
                    "score_frame: pass either references= or task="
                )
        references = load_references(task)
    elif task is not None and "task" in df.columns:
        unique = df["task"].dropna().unique().tolist()
        if len(unique) == 1 and str(unique[0]).lower() != task.lower():
            raise ValueError(
                f"score_frame: task={task!r} but df.task={unique[0]!r}"
            )

    out = df.copy()
    out["pred"] = out["completion_text"].map(extract_boxed)
    out["reference"] = out["problem_id"].map(references)
    verdicts = [is_correct(p, r) for p, r in zip(out["pred"], out["reference"])]
    out["correct"] = [v[0] for v in verdicts]
    out["score_error"] = [v[1] if v[1] is not None else "" for v in verdicts]
    return out


# Older CSVs predate the engine_label column; derive it from the class name.
_LEGACY_LABELS = (
    ("TMBatchInvariant", "TM"),
    ("Fxpr", "FXPR"),
    ("Default", "Default"),
)


def _legacy_engine_label(engine: str) -> str:
    for needle, label in _LEGACY_LABELS:
        if needle in engine:
            return label
    return engine


def score_dir(
    results_dir: str | Path,
    *,
    pattern: str = "*.math500.csv",
    references: dict[str, str] | None = None,
    task: str | None = None,
) -> pd.DataFrame:
    """Score every matching CSV in a directory into one combined frame.

    Adds a source column. Falls back to a name-substring engine_label on CSVs
    that predate the column. Either references or task must be given, or each
    CSV must carry a task column.
    """
    frames = []
    for csv in sorted(Path(results_dir).glob(pattern)):
        df = pd.read_csv(csv)
        scored = score_frame(df, references, task=task)
        if "engine_label" not in scored.columns:
            scored["engine_label"] = scored["engine"].map(_legacy_engine_label)
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


def score_ifeval(df: pd.DataFrame) -> pd.DataFrame:
    """Score an IFEval frame via the IFEval task's grader."""
    from batch_invariance_bench.tasks.ifeval import IFEval

    return IFEval().score(df)
