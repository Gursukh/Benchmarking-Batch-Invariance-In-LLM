"""Capture a torch profiler trace from an in-process vLLM engine."""

from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

from vllm import SamplingParams
from vllm.inputs import TokensPrompt

from batch_invariance_bench.engines.base import VLLMBase


def _load_trace(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        return json.load(f)


def _rank_kernels(events: list[dict]) -> list[tuple[str, float, int]]:
    agg: dict[str, list[float]] = {}
    for ev in events:
        if ev.get("ph") != "X" or ev.get("cat") != "kernel":
            continue
        a = agg.setdefault(ev.get("name", "?"), [0.0, 0])
        a[0] += ev.get("dur", 0)
        a[1] += 1
    rows = [(n, us, int(c)) for n, (us, c) in agg.items()]
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows


def _format_table(rows: list[tuple[str, float, int]], limit: int) -> str:
    total = sum(r[1] for r in rows) or 1.0
    out = [f"{'kernel':<74}{'total_ms':>12}{'count':>10}{'cuda%':>8}"]
    for name, us, cnt in rows[:limit]:
        out.append(
            f"{name[:74]:<74}{us / 1000:>12.3f}{cnt:>10}{100 * us / total:>7.1f}%"
        )
    return "\n".join(out)


def profile_engine(
    engine: VLLMBase,
    *,
    num_requests: int = 4,
    input_tokens: int = 256,
    output_tokens: int = 32,
    enforce_eager: bool = True,
    row_limit: int = 30,
    out_path: str | Path = "data/profile",
) -> Path:
    """Profile prefill and decode with the torch profiler.

    Writes the trace and a ranked kernels.txt under out_path/engine_name.
    Run from a top level script, not a notebook, so engine core stderr is
    kept. enforce_eager turns off CUDA graphs so single kernels show up.
    """
    trace_dir = Path(out_path) / engine.name
    trace_dir.mkdir(parents=True, exist_ok=True)

    saved = dict(engine.vllm_kwargs)
    engine.vllm_kwargs["profiler_config"] = {
        "profiler": "torch",
        "torch_profiler_dir": str(trace_dir.resolve()),
    }
    if enforce_eager:
        engine.vllm_kwargs["enforce_eager"] = True

    try:
        engine.setup()
        try:
            vocab = engine.tokenizer.vocab_size
            prompts = [
                TokensPrompt(prompt_token_ids=[i % vocab for i in range(input_tokens)])
                for _ in range(num_requests)
            ]
            params = SamplingParams(
                max_tokens=output_tokens, temperature=0, ignore_eos=True
            )
            engine.llm.generate(prompts, params, use_tqdm=False)  # warmup
            engine.llm.start_profile(profile_prefix=engine.name)
            try:
                engine.llm.generate(prompts, params, use_tqdm=False)
            finally:
                engine.llm.stop_profile()
            # Trace is written async; wait for it to land.
            deadline = time.time() + 60.0
            while time.time() < deadline and not list(trace_dir.glob("*.json*")):
                time.sleep(1.0)
        finally:
            engine.teardown()
    finally:
        engine.vllm_kwargs = saved

    traces = sorted(trace_dir.glob("*.json*"))
    if not traces:
        print(f"[profile] no trace written to {trace_dir}", flush=True)
        return trace_dir

    merged: dict[str, list[float]] = {}
    for t in traces:
        try:
            events = _load_trace(t).get("traceEvents", [])
        except Exception as e:
            print(f"[profile] could not parse {t.name}: {e}", flush=True)
            continue
        for name, us, cnt in _rank_kernels(events):
            a = merged.setdefault(name, [0.0, 0])
            a[0] += us
            a[1] += cnt
    ranked = sorted(
        ((n, us, int(c)) for n, (us, c) in merged.items()),
        key=lambda r: r[1],
        reverse=True,
    )
    table = _format_table(ranked, row_limit)
    print(f"\n=== {engine.name}: top {row_limit} CUDA kernels ===\n{table}", flush=True)
    (trace_dir / f"{engine.name}.kernels.txt").write_text(table)
    print(f"[profile] trace(s) + kernels.txt under {trace_dir}", flush=True)
    return trace_dir
