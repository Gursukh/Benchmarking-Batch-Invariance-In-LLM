from __future__ import annotations

from typing import TYPE_CHECKING

from batch_invariance_bench.tasks.base import HFTask, Item

if TYPE_CHECKING:
    import pandas as pd


class IFEval(HFTask):
    """Google IFEval prompts and the instruction-following grader."""

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
        """Add correct, n_instr_passed, n_instr_total.

        Correct only if the completion follows every instruction on its prompt.
        Uses the vendored IFEval checker so scores match upstream.
        """
        from batch_invariance_bench.tasks._ifeval import INSTRUCTION_DICT

        # The CSV has no instruction metadata; reload items to recover it.
        ref_map: dict[str, dict] = {
            str(it["id"]): it["reference"] for it in self.load()
        }

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
                    inst = INSTRUCTION_DICT[inst_id](inst_id)
                    # Drop None kwargs; build_description fills its own defaults.
                    kw = {k: v for k, v in (kw or {}).items() if v is not None}
                    inst.build_description(**kw)
                    args = inst.get_instruction_args()
                    if args and "prompt" in args:
                        inst.build_description(prompt=prompt)
                    if inst.check_following(completion):
                        passed += 1
                except Exception as e:  # noqa: BLE001 - keep grading the rest
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