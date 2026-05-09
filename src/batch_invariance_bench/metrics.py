from __future__ import annotations

import math
from statistics import mean, pstdev


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from Chen et al. (HumanEval): n samples, c correct."""
    if k > n:
        return float("nan")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def length_stats(completions: list[str]) -> tuple[float, float]:
    """(mean, pstdev) of completion lengths in characters."""
    if not completions:
        return 0.0, 0.0
    lens = [len(c) for c in completions]
    return float(mean(lens)), float(pstdev(lens)) if len(lens) > 1 else 0.0
