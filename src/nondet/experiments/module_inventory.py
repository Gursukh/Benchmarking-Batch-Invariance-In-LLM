"""Log every module tagged with op class, layer index and param count.

softmax is a functional op inside eager attention, not an nn.Module, so one
synthetic row is added per attention layer to cover that class.
"""

from __future__ import annotations

from collections import Counter

from nondet.utils import results, runners

FILENAME = "module_inventory.csv"
FIELDS = ["module_name", "leaf", "module_type", "op_class", "layer_idx", "n_params"]


def _softmax_rows(model) -> list[dict]:
    """One functional-softmax row per decoder layer that owns a self_attn module."""
    rows = []
    for name, _ in model.named_modules():
        if runners.classify(name) == "attention":  # a self_attn instance
            rows.append(
                {
                    "module_name": f"{name}.softmax",
                    "leaf": "softmax",
                    "module_type": "F.softmax (eager_attention)",
                    "op_class": "softmax",
                    "layer_idx": runners.layer_index_of(name),
                    "n_params": 0,
                }
            )
    return rows


def main():
    ap = results.base_parser()
    args = ap.parse_args()

    model_id = args.model
    env = results.env_from_args(args)

    runner = runners.HFRunner(model_id, dtype=args.dtype).load()
    inv = runners.inventory(runner.model) + _softmax_rows(runner.model)
    rows = [{**env, **r} for r in inv]
    runner.free()

    path = results.write_rows(FILENAME, rows, results.ENV_FIELDS + FIELDS)

    counts = Counter(r["op_class"] or "(unclassified)" for r in inv)
    print(f"[module_inventory] {len(rows)} modules -> {path}")
    for cls in (*runners.OP_CLASSES, "(unclassified)"):
        if counts.get(cls):
            print(f"  {cls:<14} {counts[cls]}")


if __name__ == "__main__":
    main()
