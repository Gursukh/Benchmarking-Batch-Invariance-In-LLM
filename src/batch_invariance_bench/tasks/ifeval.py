from __future__ import annotations

from typing import TYPE_CHECKING

from batch_invariance_bench.tasks.base import HFTask, Item

if TYPE_CHECKING:
    import pandas as pd


class IFEval(HFTask):
    """Google IFEval prompts and the instruction following grader.

    Each reference keeps the instruction id list and kwargs. score needs
    the instruction_following_eval package and raises if it is missing.
    """

    name = "ifeval"
    hf_dataset = "google/IFEval"
    default_split = "train"

    def _to_item(self, row: dict, idx: int) -> Item:
        return Item(
            id=str(row["key"]),
            prompt=row["prompt"],
            reference={
                "instruction_id_list": list(row["instruction_id_list"]),
                "kwargs": list(row["kwargs"]),
                # Some instructions check against the original prompt.
                "prompt": row["prompt"],
            },
        )

    def score(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Score an IFEval frame. Adds correct, n_instr_passed, n_instr_total."""
        import pandas as pd

        try:
            from instruction_following_eval import instructions_registry
        except ImportError as e:
            raise ImportError(
                "IFEval scoring requires `instruction_following_eval`. "
                "Install with `pip install instruction-following-eval`."
            ) from e

        # CSV has no instruction metadata, so reload the items.
        ref_map: dict[str, dict] = {str(it["id"]): it["reference"] for it in self.load()}

        registry = instructions_registry.INSTRUCTION_DICT

        out = df.copy()
        n_pass: list[int] = []
        n_total: list[int] = []
        correct: list[bool] = []
        errors: list[str] = []

        for _, row in out.iterrows():
            pid = str(row["problem_id"])
            completion = row.get("completion_text", "") or ""
            ref = ref_map.get(pid)
            if ref is None:
                n_pass.append(0)
                n_total.append(0)
                correct.append(False)
                errors.append(f"no IFEval reference for problem_id {pid!r}")
                continue
            ids = ref.get("instruction_id_list") or []
            kw_list = ref.get("kwargs") or []
            prompt = ref.get("prompt") or ""
            passed = 0
            total = 0
            err: str | None = None
            for inst_id, kw in zip(ids, kw_list):
                total += 1
                try:
                    cls = registry[inst_id]
                    inst = cls(inst_id)
                    inst.build_description(**(kw or {}))
                    if hasattr(inst, "get_instruction_args"):
                        _ = inst.get_instruction_args()
                    if hasattr(inst, "build_description") and prompt:
                        pass
                    if inst.check_following(completion):
                        passed += 1
                except Exception as e:
                    if err is None:
                        err = f"{inst_id}: {type(e).__name__}: {e}"
            n_pass.append(passed)
            n_total.append(total)
            correct.append(total > 0 and passed == total)
            errors.append(err or "")

        out["n_instr_passed"] = n_pass
        out["n_instr_total"] = n_total
        out["correct"] = correct
        out["score_error"] = errors
        return out
