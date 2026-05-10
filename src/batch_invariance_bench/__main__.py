"""Edit `engines` to register the cells you want; then `python -m batch_invariance_bench [--out PATH]`."""

from __future__ import annotations

import argparse

from batch_invariance_bench.engines.vllm_default import VLLMDefault
from batch_invariance_bench.engines.vllm_tm_batch_invariant import VLLMTMBatchInvariant
from batch_invariance_bench.runner import run
from batch_invariance_bench.tasks import AIME, IFEval, MATH500


engines = [
    VLLMDefault(),
    VLLMTMBatchInvariant(),
]

tasks = [MATH500(), AIME(), IFEval()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="batch_invariance_bench")
    parser.add_argument("--out", default=None, help="Output directory (default: results/). One CSV per (gpu, engine, task) is written here.")
    args = parser.parse_args()
    print(f"wrote {run(engines=engines, tasks=tasks, n=1, out_path=args.out)}")
