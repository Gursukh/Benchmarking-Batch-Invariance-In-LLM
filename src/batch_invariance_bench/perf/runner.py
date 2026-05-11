from __future__ import annotations

import datetime as _dt
import json
import time
import traceback
import uuid
from pathlib import Path
from typing import Sequence

import httpx

from batch_invariance_bench.io import gpu_info, vllm_version
from batch_invariance_bench.perf.configs import DEFAULT_ENGINES, ServerConfig
from batch_invariance_bench.perf.io import (
    PERF_COLUMNS,
    append_perf_row,
    cell_dir,
    perf_csv_path,
    summary_from_llmperf,
)
from batch_invariance_bench.perf.llmperf_runner import run_llmperf
from batch_invariance_bench.perf.server import VLLMServer
from batch_invariance_bench.perf.vram import VRAMSampler


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _warmup(base_url: str, api_key: str, model_id: str, n: int) -> None:
    """Send and discard n short completions so the first measured request
    doesn't eat graph-compile and first-page-fault costs."""
    if n <= 0:
        return
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "warmup"}],
        "max_tokens": 8,
        "temperature": 0,
    }
    with httpx.Client(timeout=60.0) as client:
        for _ in range(n):
            try:
                client.post(f"{base_url}/chat/completions", headers=headers, json=payload)
            except Exception:
                pass


def run(
    engines: Sequence[ServerConfig] = tuple(DEFAULT_ENGINES),
    concurrency: Sequence[int] = (1, 4, 16, 64),
    mean_input_tokens: int = 550,
    stddev_input_tokens: int = 150,
    mean_output_tokens: int = 150,
    stddev_output_tokens: int = 10,
    max_requests: int = 100,
    warmup_requests: int = 5,
    timeout_s: float = 600.0,
    vram_hz: float = 5.0,
    port: int = 8000,
    api_key: str = "token-abc123",
    out_path: str | Path = "data/perf",
) -> Path:
    """Run every (engine, concurrency) cell. One CSV row + one cell directory per cell."""
    out_dir = Path(out_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = uuid.uuid4().hex[:12]
    arch, gpu_name = gpu_info()
    vllm_v = vllm_version()
    csv_path = perf_csv_path(out_dir, gpu_name)

    print(
        f"[perf] out_dir={out_dir} engines={[e.name for e in engines]} "
        f"concurrency={list(concurrency)} run_id={run_id}",
        flush=True,
    )

    for engine in engines:
        for conc in concurrency:
            t_cell = time.perf_counter()
            cell = cell_dir(out_dir, gpu_name, engine.name, conc)
            cell.mkdir(parents=True, exist_ok=True)
            log_path = cell / "vllm_serve.log"

            print(f"[{engine.name} | c={conc}] setup...", flush=True)
            server = VLLMServer(engine, port=port, log_path=log_path)
            error: str | None = None
            llmperf_summary: dict = {}
            peak_vram = float("nan")
            mean_vram = float("nan")
            t_run = 0.0

            try:
                server.start()
                print(
                    f"[{engine.name} | c={conc}] ready "
                    f"({time.perf_counter() - t_cell:.1f}s)",
                    flush=True,
                )
                _warmup(server.base_url, api_key, engine.model_id, warmup_requests)

                t_run_start = time.perf_counter()
                with VRAMSampler(hz=vram_hz, out_csv=cell / "vram_samples.csv") as sampler:
                    llmperf_summary = run_llmperf(
                        model_id=engine.model_id,
                        base_url=server.base_url,
                        api_key=api_key,
                        concurrency=conc,
                        max_requests=max_requests,
                        mean_input_tokens=mean_input_tokens,
                        stddev_input_tokens=stddev_input_tokens,
                        mean_output_tokens=mean_output_tokens,
                        stddev_output_tokens=stddev_output_tokens,
                        timeout_s=timeout_s,
                        results_dir=cell,
                    )
                t_run = time.perf_counter() - t_run_start
                peak_vram = sampler.peak_mb
                mean_vram = sampler.mean_mb
            except Exception:
                error = traceback.format_exc()
                print(f"[{engine.name} | c={conc}] ERROR\n{error}", flush=True)
            finally:
                server.stop()
                print(f"[{engine.name} | c={conc}] teardown", flush=True)

            row = {col: "" for col in PERF_COLUMNS}
            row.update({
                "run_id":              run_id,
                "gpu_arch":            arch,
                "gpu_name":            gpu_name,
                "engine":              engine.name,
                "vllm_version":        vllm_v,
                "model_id":            engine.model_id,
                "concurrency":         conc,
                "mean_input_tokens":   mean_input_tokens,
                "stddev_input_tokens": stddev_input_tokens,
                "mean_output_tokens":  mean_output_tokens,
                "stddev_output_tokens": stddev_output_tokens,
                "peak_vram_mb":        peak_vram,
                "mean_vram_mb":        mean_vram,
                "duration_s":          round(t_run, 3),
                "timestamp":           _now(),
            })
            row.update({
                k: v for k, v in summary_from_llmperf(llmperf_summary).items()
                if v is not None
            })
            append_perf_row(csv_path, row)

            (cell / "metadata.json").write_text(json.dumps({
                "run_id": run_id,
                "gpu_arch": arch,
                "gpu_name": gpu_name,
                "engine": engine.name,
                "model_id": engine.model_id,
                "extra_env": engine.extra_env,
                "extra_cli": engine.extra_cli,
                "server_command": server.command(),
                "concurrency": conc,
                "mean_input_tokens": mean_input_tokens,
                "stddev_input_tokens": stddev_input_tokens,
                "mean_output_tokens": mean_output_tokens,
                "stddev_output_tokens": stddev_output_tokens,
                "max_requests": max_requests,
                "warmup_requests": warmup_requests,
                "vllm_version": vllm_v,
                "duration_s": round(t_run, 3),
                "cell_wallclock_s": round(time.perf_counter() - t_cell, 3),
                "timestamp": _now(),
                "error": error,
            }, indent=2))

            print(
                f"[{engine.name} | c={conc}] done "
                f"({time.perf_counter() - t_cell:.1f}s cell, {t_run:.1f}s measured)",
                flush=True,
            )

    return out_dir
