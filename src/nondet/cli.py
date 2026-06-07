"""Run a Chapter 3 experiment: nondet <experiment> [args...]."""

from __future__ import annotations

import importlib
import sys

EXPERIMENTS = [
    "measurement_neutrality",
    "baseline_divergence",
    "module_inventory",
]


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in EXPERIMENTS:
        print("usage: nondet <experiment> [args...]\n\nexperiments:")
        for name in EXPERIMENTS:
            print(f"  {name}")
        sys.exit(0 if len(sys.argv) < 2 else 2)
    name = sys.argv[1]
    sys.argv = [f"nondet {name}", *sys.argv[2:]]  # rest goes to the experiment's argparse
    importlib.import_module(f"nondet.experiments.{name}").main()


if __name__ == "__main__":
    main()
