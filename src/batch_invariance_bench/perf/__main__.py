"""LLMPerf sweep entrypoint. `python -m batch_invariance_bench.perf --help`."""

from __future__ import annotations

import argparse

from batch_invariance_bench.perf.configs import ENGINE_ALIASES
from batch_invariance_bench.perf.runner import run


def main() -> None:
    parser = argparse.ArgumentParser(prog="batch_invariance_bench.perf")
    parser.add_argument("--out", default="data/perf")
    parser.add_argument("--engines", default="default,tm",
                        help=f"Comma-separated aliases from {{{', '.join(ENGINE_ALIASES)}}}.")
    parser.add_argument("--concurrency", default="1,4,16,64",
                        help="Comma-separated concurrency levels.")
    parser.add_argument("--max-requests", type=int, default=100,
                        help="Completed requests per cell.")
    args = parser.parse_args()

    try:
        engines = [ENGINE_ALIASES[a.strip()] for a in args.engines.split(",") if a.strip()]
    except KeyError as e:
        parser.error(f"unknown engine alias {e.args[0]!r}; known: {list(ENGINE_ALIASES)}")
    concurrency = tuple(int(x) for x in args.concurrency.split(","))

    out = run(
        engines=engines,
        concurrency=concurrency,
        max_requests=args.max_requests,
        out_path=args.out,
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
