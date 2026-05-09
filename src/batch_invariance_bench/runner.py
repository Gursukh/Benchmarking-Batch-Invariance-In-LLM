from __future__ import annotations

import datetime as _dt
import json
import time
import uuid
from pathlib import Path
from typing import Iterable, Sequence

from batch_invariance_bench.engine import Engine
from batch_invariance_bench.io import (
    append_rows,
    default_output_path,
    gpu_info,
    vllm_version,
)
from batch_invariance_bench.metrics import length_stats, pass_at_k
from batch_invariance_bench.tasks.base import Item, Task


def _chunked(seq: Sequence[Item], size: int) -> Iterable[Sequence[Item]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def run(
    engines: Sequence[Engine],
    tasks: Sequence[Task],
    batch_sizes: Sequence[int] = (1, 2, 4, 6, 8, 16),
    n: int = 1,
    seed: int = 0,
    sampling: dict | None = None,
    out_path: str | Path | None = None,
) -> Path:
    """Sweep engines x tasks x batch_sizes; one CSV row per problem-cell."""
    out = Path(out_path) if out_path else default_output_path()
    run_id = uuid.uuid4().hex[:12]
    arch, gpu_name = gpu_info()
    vllm_v = vllm_version()
    sampling = {**(sampling or {}), "seed": seed}
    print(f"[run] out={out} engines={len(engines)} tasks={len(tasks)} bs={list(batch_sizes)} n={n}", flush=True)

    for engine in engines:
        t0 = time.perf_counter()
        print(f"[{engine.name}] setup...", flush=True)
        engine.setup()
        print(f"[{engine.name}] ready ({time.perf_counter() - t0:.1f}s)", flush=True)
        try:
            for task in tasks:
                items = task.load()
                for bs in batch_sizes:
                    t_bs = time.perf_counter()
                    print(f"[{engine.name} | {task.name} | bs={bs}] {len(items)} items", flush=True)
                    for batch in _chunked(items, bs):
                        prompts = [it["prompt"] for it in batch]
                        completions = engine.generate(prompts, n=n, sampling=sampling)
                        rows = []
                        for item, comps in zip(batch, completions):
                            res = task.score(item, comps)
                            c = sum(res["correct"])
                            mean_len, std_len = length_stats(comps)
                            rows.append(
                                {
                                    "run_id": run_id,
                                    "gpu_arch": arch,
                                    "gpu_name": gpu_name,
                                    "engine": engine.name,
                                    "task": task.name,
                                    "batch_size": bs,
                                    "problem_id": item["id"],
                                    "n_samples": n,
                                    "pass_at_1": pass_at_k(n, c, 1),
                                    "pass_at_4": pass_at_k(n, c, 4),
                                    "mean_resp_len": mean_len,
                                    "std_resp_len": std_len,
                                    "completions_json": json.dumps(comps),
                                    "correct_mask": json.dumps(res["correct"]),
                                    "seed": seed,
                                    "vllm_version": vllm_v,
                                    "timestamp": _dt.datetime.utcnow().isoformat(),
                                }
                            )
                        append_rows(out, rows)
                    print(f"[{engine.name} | {task.name} | bs={bs}] done ({time.perf_counter() - t_bs:.1f}s)", flush=True)
        finally:
            engine.teardown()
            print(f"[{engine.name}] teardown", flush=True)

    return out
