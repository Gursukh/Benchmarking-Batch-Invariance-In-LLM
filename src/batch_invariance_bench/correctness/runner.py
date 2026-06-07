"""Correctness harness.

run_correctness sweeps batch sizes and dumps raw outputs (token ids, logprobs)
for offline divergence analysis. run_accuracy runs each task once at a single
concurrency and scores inline via the task's grader. Both build each engine once.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from batch_invariance_bench.common.csvio import (
    append_csv_rows,
    default_output_dir,
    now,
    slug,
)
from batch_invariance_bench.common.gpu import gpu_info, vllm_version
from batch_invariance_bench.correctness.score import accuracy_table
from batch_invariance_bench.engines.base import Engine, Sample
from batch_invariance_bench.tasks.base import Item, Task


OUTPUT_COLUMNS = [
    "run_id",
    "gpu_arch",
    "gpu_name",
    "engine",
    "engine_label",
    "vllm_version",
    "task",
    "problem_id",
    "batch_size",
    "sample_idx",
    "completion_text",
    "completion_token_ids",
    "output_logprobs",
    "n_prompt_tokens",
    "n_output_tokens",
    "finish_reason",
    "stop_reason",
    "timestamp",
]


def _run_meta() -> tuple[str, str, str, str]:
    """(run_id, gpu_arch, gpu_name, vllm_version)."""
    arch, gpu_name = gpu_info()
    return uuid.uuid4().hex[:12], arch, gpu_name, vllm_version()


def _engine_meta(engine: Engine) -> tuple[str, str]:
    """(name, label), tolerant of non-vLLM engines."""
    return getattr(engine, "name", engine.key), getattr(engine, "label", "") or ""


def _chunked(seq: Sequence[Item], size: int) -> Iterable[Sequence[Item]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def run_correctness(
    engines: Sequence[Engine],
    tasks: Sequence[Task],
    batch_sizes: Sequence[int] = (1, 2, 4, 6, 8, 16),
    n: int = 1,
    sampling: dict | None = None,
    out_path: str | Path | None = None,
) -> Path:
    """Run every (engine, task, batch_size) combo and dump raw outputs.

    The engine is built once and reused across the whole sweep to skip per-bs
    startup cost; vLLM keeps no decode state between generate() calls, so batch
    sizes can't leak. Scoring is left to the caller.
    """
    out_dir = default_output_dir(out_path) if out_path else default_output_dir()
    run_id, arch, gpu_name, vllm_v = _run_meta()
    print(
        f"[run] out_dir={out_dir} engines={len(engines)} tasks={len(tasks)} "
        f"bs={list(batch_sizes)} n={n} run_id={run_id}",
        flush=True,
    )

    for engine in engines:
        engine_name, engine_label = _engine_meta(engine)

        t0 = time.perf_counter()
        print(f"[{engine_name}] setup...", flush=True)
        engine.setup()
        print(
            f"[{engine_name}] ready ({time.perf_counter() - t0:.1f}s)",
            flush=True,
        )
        try:
            for task in tasks:
                items = task.load()
                task_out = (
                    out_dir
                    / f"{slug(gpu_name)}.{run_id}.{slug(engine_name)}.{slug(task.name)}.csv"
                )

                for bs in batch_sizes:
                    total = len(items)
                    idx = 0
                    t_bs = time.perf_counter()
                    print(
                        f"[{engine_name} | {task.name} | bs={bs}] {total} items",
                        flush=True,
                    )
                    for batch in _chunked(items, bs):
                        prompts = [it["prompt"] for it in batch]
                        completions = engine.generate(prompts, n=n, sampling=sampling)
                        rows = []
                        for item, samples in zip(batch, completions):
                            for sample_idx, s in enumerate(samples):
                                rows.append(
                                    _row(
                                        run_id=run_id,
                                        arch=arch,
                                        gpu_name=gpu_name,
                                        engine_name=engine_name,
                                        engine_label=engine_label,
                                        vllm_v=vllm_v,
                                        task_name=task.name,
                                        problem_id=str(item["id"]),
                                        bs=bs,
                                        sample_idx=sample_idx,
                                        sample=s,
                                    )
                                )
                            idx += 1
                            print(f"\r  [{idx}/{total}]", end="", flush=True)
                        append_csv_rows(task_out, rows, OUTPUT_COLUMNS)
                    print(
                        f"\r[{engine_name} | {task.name} | bs={bs}] done "
                        f"{total}/{total} ({time.perf_counter() - t_bs:.1f}s)",
                        flush=True,
                    )
        finally:
            engine.teardown()
            print(f"[{engine_name}] teardown", flush=True)

    return out_dir


def _row(
    *,
    run_id: str,
    arch: str,
    gpu_name: str,
    engine_name: str,
    engine_label: str,
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
        "engine_label": engine_label,
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
        "timestamp": now(),
    }


def _setup_engine_at_concurrency(
    engine: Engine, concurrency: int, mean_input_tokens: int
) -> None:
    """Set max_num_seqs=concurrency, then setup().

    Floors max_num_batched_tokens at 2048 for small concurrencies. Only touches
    vllm_kwargs when the engine has them.
    """
    vllm_kwargs = getattr(engine, "vllm_kwargs", None)
    if isinstance(vllm_kwargs, dict):
        vllm_kwargs["max_num_seqs"] = concurrency
        vllm_kwargs["max_num_batched_tokens"] = max(
            concurrency * mean_input_tokens, 2048
        )
    engine.setup()


def run_accuracy(
    engine: Engine,
    concurrency: int = 16,
    tasks: Sequence[Task] = (),
    n: int = 1,
    sampling: dict | None = None,
    mean_input_tokens: int = 550,
    out_path: str | Path | None = None,
) -> pd.DataFrame:
    """Run each task once at the given concurrency and score it.

    All of a task's prompts go in one generate() call; vLLM runs up to
    `concurrency` sequences at once (max_num_seqs). Each task uses its own scorer.
    Returns a per-(engine, task) accuracy table and writes the scored rows to
    out_dir/{gpu}.{run_id}.{engine}.accuracy.csv. Defaults to MATH-500 + IFEval.
    """
    if not tasks:
        from batch_invariance_bench.tasks.boxed import MATH500
        from batch_invariance_bench.tasks.ifeval import IFEval

        tasks = (MATH500(), IFEval())

    out_dir = default_output_dir(out_path) if out_path else default_output_dir()
    run_id, arch, gpu_name, vllm_v = _run_meta()
    engine_name, engine_label = _engine_meta(engine)

    print(
        f"[accuracy] out_dir={out_dir} engine={engine_name} "
        f"concurrency={concurrency} tasks={[t.name for t in tasks]} "
        f"n={n} run_id={run_id}",
        flush=True,
    )

    t0 = time.perf_counter()
    print(f"[{engine_name}] setup (max_num_seqs={concurrency})...", flush=True)
    _setup_engine_at_concurrency(engine, concurrency, mean_input_tokens)
    print(f"[{engine_name}] ready ({time.perf_counter() - t0:.1f}s)", flush=True)

    scored_frames: list[pd.DataFrame] = []
    n_tasks = len(tasks)
    try:
        for task_idx, task in enumerate(tasks, start=1):
            items = task.load()
            prompts = [it["prompt"] for it in items]
            t_task = time.perf_counter()
            print(
                f"[{engine_name} | task {task_idx}/{n_tasks} {task.name}] "
                f"generating {len(prompts)} prompts (n={n}) "
                f"at concurrency {concurrency}...",
                flush=True,
            )
            # use_tqdm shows vLLM's progress bar for the single generate() call.
            completions = engine.generate(
                prompts, n=n, sampling=sampling, use_tqdm=True
            )
            print(
                f"[{engine_name} | {task.name}] generation done "
                f"({time.perf_counter() - t_task:.1f}s); scoring...",
                flush=True,
            )

            rows: list[dict] = []
            for item, samples in zip(items, completions):
                for sample_idx, s in enumerate(samples):
                    rows.append(
                        {
                            "run_id": run_id,
                            "gpu_arch": arch,
                            "gpu_name": gpu_name,
                            "engine": engine_name,
                            "engine_label": engine_label,
                            "vllm_version": vllm_v,
                            "task": task.name,
                            "concurrency": concurrency,
                            "problem_id": str(item["id"]),
                            "sample_idx": sample_idx,
                            "completion_text": s.text,
                            "n_output_tokens": s.n_output_tokens,
                            "timestamp": now(),
                        }
                    )
            df = pd.DataFrame(rows)
            scored = task.score(df)
            scored_frames.append(scored)
            acc = float(scored["correct"].mean()) if len(scored) else float("nan")
            print(
                f"[{engine_name} | {task.name}] done {len(scored)} rows, "
                f"accuracy={acc:.4f} ({time.perf_counter() - t_task:.1f}s)",
                flush=True,
            )
    finally:
        engine.teardown()
        print(f"[{engine_name}] teardown", flush=True)

    scored_all = pd.concat(scored_frames, ignore_index=True)
    csv_path = (
        out_dir / f"{slug(gpu_name)}.{run_id}.{slug(engine_name)}.accuracy.csv"
    )
    scored_all.to_csv(csv_path, index=False)

    table = accuracy_table(scored_all, by=["engine_label", "task", "concurrency"])

    print(f"\n[accuracy] {engine_name} @ concurrency={concurrency}", flush=True)
    for _, r in table.iterrows():
        line = (
            f"  {r['task']:<10} n={int(r['n']):<5} "
            f"correct={int(r['correct']):<5} accuracy={r['accuracy']:.4f}"
        )
        # IFEval also reports instruction-level accuracy.
        task_rows = scored_all[scored_all["task"] == r["task"]]
        if "n_instr_total" in task_rows.columns:
            tot = task_rows["n_instr_total"].sum()
            passed = task_rows["n_instr_passed"].sum()
            if tot:
                line += f" instr_accuracy={passed / tot:.4f}"
        print(line, flush=True)
    print(f"[accuracy] wrote {csv_path}", flush=True)

    return table
