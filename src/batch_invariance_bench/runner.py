from __future__ import annotations

import datetime as _dt
import json
import time
import uuid
from pathlib import Path
from typing import Iterable, Sequence

from batch_invariance_bench.engine import Engine, Sample
from batch_invariance_bench.io import (
    _slug,
    append_rows,
    default_output_path,
    gpu_info,
    vllm_version,
)
from batch_invariance_bench.tasks.base import Item, Task


def _chunked(seq: Sequence[Item], size: int) -> Iterable[Sequence[Item]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def run(
    engines: Sequence[Engine],
    tasks: Sequence[Task],
    batch_sizes: Sequence[int] = (1, 2, 4, 6, 8, 16),
    n: int = 1,
    sampling: dict | None = None,
    out_path: str | Path | None = None,
) -> Path:
    """Run every (engine, task, batch_size) combination and dump raw outputs.

    The engine is rebuilt between batch sizes so KV cache and compiled-graph
    state can't leak across the sweep. Scoring is intentionally not done here.
    """
    out_dir = Path(out_path) if out_path else default_output_path().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:12]
    arch, gpu_name = gpu_info()
    vllm_v = vllm_version()
    print(
        f"[run] out_dir={out_dir} engines={len(engines)} tasks={len(tasks)} "
        f"bs={list(batch_sizes)} n={n} run_id={run_id}",
        flush=True,
    )

    for engine in engines:
        for task in tasks:
            items = task.load()
            task_out = (
                out_dir
                / f"{_slug(gpu_name)}.{_slug(engine.name)}.{_slug(task.name)}.csv"
            )

            for bs in batch_sizes:
                t0 = time.perf_counter()
                print(f"[{engine.name} | {task.name} | bs={bs}] setup...", flush=True)
                engine.setup()
                print(
                    f"[{engine.name} | {task.name} | bs={bs}] ready "
                    f"({time.perf_counter() - t0:.1f}s)",
                    flush=True,
                )
                try:
                    total = len(items)
                    idx = 0
                    t_bs = time.perf_counter()
                    print(
                        f"[{engine.name} | {task.name} | bs={bs}] {total} items",
                        flush=True,
                    )
                    for batch in _chunked(items, bs):
                        prompts = [it["prompt"] for it in batch]
                        completions = engine.generate(
                            prompts, n=n, sampling=sampling
                        )
                        rows = []
                        for item, samples in zip(batch, completions):
                            for sample_idx, s in enumerate(samples):
                                rows.append(_row(
                                    run_id=run_id,
                                    arch=arch,
                                    gpu_name=gpu_name,
                                    engine_name=engine.name,
                                    vllm_v=vllm_v,
                                    task_name=task.name,
                                    problem_id=str(item["id"]),
                                    bs=bs,
                                    sample_idx=sample_idx,
                                    sample=s,
                                ))
                            idx += 1
                            print(f"\r  [{idx}/{total}]", end="", flush=True)
                        append_rows(task_out, rows)
                    print(
                        f"\r[{engine.name} | {task.name} | bs={bs}] done "
                        f"{total}/{total} ({time.perf_counter() - t_bs:.1f}s)",
                        flush=True,
                    )
                finally:
                    engine.teardown()
                    print(
                        f"[{engine.name} | {task.name} | bs={bs}] teardown",
                        flush=True,
                    )

    return out_dir


def _row(
    *,
    run_id: str,
    arch: str,
    gpu_name: str,
    engine_name: str,
    vllm_v: str,
    task_name: str,
    problem_id: str,
    bs: int,
    sample_idx: int,
    sample: Sample,
) -> dict:
    return {
        "run_id": run_id,
        "gpu_arch": arch,
        "gpu_name": gpu_name,
        "engine": engine_name,
        "vllm_version": vllm_v,
        "task": task_name,
        "problem_id": problem_id,
        "batch_size": bs,
        "sample_idx": sample_idx,
        "completion_text": sample.text,
        "completion_token_ids": json.dumps(sample.token_ids, separators=(",", ":")),
        "output_logprobs": json.dumps(sample.logprobs, separators=(",", ":")),
        "n_prompt_tokens": sample.n_prompt_tokens,
        "n_output_tokens": sample.n_output_tokens,
        "finish_reason": sample.finish_reason,
        "stop_reason": sample.stop_reason,
        "timestamp": _now(),
    }
