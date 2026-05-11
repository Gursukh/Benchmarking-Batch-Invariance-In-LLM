"""Edit the `engines` list below, then run `python -m batch_invariance_bench`."""

from __future__ import annotations

import argparse

from batch_invariance_bench.engines.vllm_default import VLLMDefault
from batch_invariance_bench.engines.vllm_fxpr import VLLMFxpr
from batch_invariance_bench.engines.vllm_tm_batch_invariant import VLLMTMBatchInvariant
from batch_invariance_bench.runner import run
from batch_invariance_bench.tasks import AIME, IFEval, MATH500


engines = [
    VLLMDefault(),
    VLLMTMBatchInvariant(),
    VLLMFxpr(),
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="batch_invariance_bench")
    parser.add_argument(
        "--out",
        default=None,
        help="Where to write the output CSVs (default: results/).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap problems per task. Handy for a quick smoke test.",
    )
    parser.add_argument(
        "--batch-sizes",
        default=None,
        help="Comma-separated batch sizes (e.g. '1,2,4'). Default: 1,2,4,6,8,16.",
    )
    args = parser.parse_args()

    tasks = [MATH500(limit=args.limit), AIME(limit=args.limit), IFEval(limit=args.limit)]

    kwargs: dict = {"engines": engines, "tasks": tasks, "out_path": args.out}
    if args.batch_sizes:
        kwargs["batch_sizes"] = tuple(int(x) for x in args.batch_sizes.split(","))

    print(f"wrote {run(**kwargs)}")
