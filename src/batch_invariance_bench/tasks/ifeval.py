from __future__ import annotations

import warnings

from datasets import load_dataset

from batch_invariance_bench.tasks.base import Item, ScoreResult, Task


def _load_checker():
    # Google's checkers live in `instruction_following_eval` upstream and
    # are also vendored in lm-eval-harness — try both.
    try:
        from instruction_following_eval import instructions_registry  # type: ignore
    except ImportError:
        try:
            from lm_eval.tasks.ifeval import instructions_registry  # type: ignore
        except ImportError:
            return None

    def _check(instruction_id: str, kwargs: dict, response: str) -> bool:
        cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        inst = cls(instruction_id)
        inst.build_description(**(kwargs or {}))
        return bool(inst.check_following(response))

    return _check


class IFEval(Task):
    """Google IFEval — strict mode: a completion passes iff every check passes."""

    name = "ifeval"

    def __init__(self, split: str = "train", limit: int | None = None) -> None:
        self._split = split
        self._limit = limit
        self._checker = _load_checker()
        if self._checker is None:
            warnings.warn(
                "IFEval scoring requires `instruction_following_eval` or "
                "`lm-eval-harness`; no checker found. Scores will be all False.",
                stacklevel=2,
            )

    def load(self) -> list[Item]:
        ds = load_dataset("google/IFEval", split=self._split)
        if self._limit:
            ds = ds.select(range(min(self._limit, len(ds))))
        items: list[Item] = []
        for row in ds:
            items.append(
                Item(
                    id=str(row["key"]),
                    prompt=row["prompt"],
                    reference={
                        "instruction_id_list": list(row["instruction_id_list"]),
                        "kwargs": list(row["kwargs"]),
                    },
                )
            )
        return items

    def score(self, item: Item, completions: list[str]) -> ScoreResult:
        ref = item["reference"]
        ids: list[str] = ref["instruction_id_list"]
        kwargs_list: list[dict] = ref["kwargs"]

        if self._checker is None:
            return ScoreResult(
                correct=[False] * len(completions),
                extracted=[""] * len(completions),
            )

        correct: list[bool] = []
        for c in completions:
            ok = True
            for iid, kw in zip(ids, kwargs_list):
                kw = {k: v for k, v in (kw or {}).items() if v is not None}
                try:
                    if not self._checker(iid, kw, c):
                        ok = False
                        break
                except Exception:
                    ok = False
                    break
            correct.append(ok)
        return ScoreResult(correct=correct, extracted=[""] * len(completions))
