from __future__ import annotations

import concurrent.futures as cf
import dataclasses
import random
import subprocess
import time
import traceback
import uuid
from pathlib import Path
from typing import Sequence

import httpx

from batch_invariance_bench.common.csvio import now, write_json
from batch_invariance_bench.common.gpu import gpu_info, vllm_version
from batch_invariance_bench.engines.base import ServerSpec, VLLMBase
from batch_invariance_bench.engines.default import VLLMDefault
from batch_invariance_bench.engines.tm_batch_invariant import VLLMTMBatchInvariant
from batch_invariance_bench.perf.batch_sampler import BatchSampler
from batch_invariance_bench.perf.load_test import run_load_test
from batch_invariance_bench.perf.log_parse import memory_from_vllm_log
from batch_invariance_bench.perf.schema import (
    PERF_COLUMNS,
    append_perf_row,
    perf_csv_path,
    serve_log_path,
)
from batch_invariance_bench.perf.server import VLLMServer
from batch_invariance_bench.perf.vram import VRAMSampler


# Pinned so every engine decodes the same work: ignore_eos forces exactly
# max_tokens tokens, temperature 0 keeps decoding deterministic.
DEFAULT_SAMPLING_PARAMS: dict = {"ignore_eos": True, "temperature": 0}


def default_engines() -> list[VLLMBase]:
    """The Default-vs-TM comparison PERF.md is built around.

    VLLMFxpr() also works as a server engine now and can be passed to run()
    explicitly.
    """
    return [VLLMDefault(), VLLMTMBatchInvariant()]


def warmup(
    base_url: str,
    api_key: str,
    model_id: str,
    n: int,
    concurrency: int,
) -> None:
    """Send warmup requests at the target concurrency so the measured path is
    hot before measurement. Raises if every warmup request fails.
    """
    n = max(n, 2 * concurrency)
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

        def _one(_: int) -> bool:
            try:
                client.post(
                    f"{base_url}/chat/completions", headers=headers, json=payload
                )
                return True
            except Exception:
                return False

        with cf.ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
            results = list(ex.map(_one, range(n)))
    if results and not any(results):
        raise RuntimeError("all warmup requests failed; server appears broken")


def _lock_gpu_clocks(mhz: int | None) -> tuple[bool, int | None]:
    """Lock the GPU graphics clock for a steady clock across the sweep.

    Best effort. Returns (locked, effective_mhz) and never raises.
    """
    try:
        if mhz is None:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=clocks.max.graphics",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            mhz = int(out.stdout.strip().splitlines()[0])
        subprocess.run(["nvidia-smi", "-pm", "1"], capture_output=True, check=True)
        subprocess.run(
            ["nvidia-smi", f"--lock-gpu-clocks={mhz}"], capture_output=True, check=True
        )
        return True, mhz
    except Exception as e:
        print(f"[perf] GPU clock lock failed ({e}); continuing unlocked", flush=True)
        return False, None


def _reset_gpu_clocks() -> None:
    try:
        subprocess.run(
            ["nvidia-smi", "--reset-gpu-clocks"], capture_output=True, check=True
        )
    except Exception:
        pass


def run(
    engines: Sequence[VLLMBase] | None = None,
    concurrency: Sequence[int] = (1, 4, 16, 64),
    mean_input_tokens: int = 550,
    stddev_input_tokens: int = 150,
    mean_output_tokens: int = 150,
    stddev_output_tokens: int = 0,
    max_requests: int = 500,
    requests_per_concurrency: int = 20,
    repeats: int = 3,
    warmup_requests: int = 5,
    timeout_s: float = 600.0,
    server_timeout_s: float = 600.0,
    enforce_eager: bool = False,
    vram_hz: float = 5.0,
    batch_hz: float = 10.0,
    sampling_params: dict | None = None,
    lock_gpu_clocks: bool = True,
    gpu_clock_mhz: int | None = None,
    seed: int = 0,
    port: int = 8000,
    api_key: str = "token-abc123",
    out_path: str | Path | None = None,
) -> Path:
    """Run every (engine, concurrency, repeat) cell.

    Each cell appends one row to its engine's CSV. A run produces one .perf.csv
    and one .serve.log per engine, plus <run_id>.run_manifest.json.

    `engines` defaults to default_engines(). Each engine is turned into a
    `vllm serve` spec via server_spec(), so the served run matches it exactly.

    Cells run in a fixed-seed shuffled order, so Default and TM see the same
    average thermal and clock conditions instead of one running fully first.

    enforce_eager defaults to False, keeping CUDA graphs and torch.compile on.
    Set it True only for an eager comparison run.
    """
    out_dir = Path(out_path) if out_path else Path("data/perf")
    out_dir.mkdir(parents=True, exist_ok=True)

    if engines is None:
        engines = default_engines()
    if sampling_params is None:
        sampling_params = dict(DEFAULT_SAMPLING_PARAMS)

    # Turn each engine into its `vllm serve` launch spec.
    specs: list[ServerSpec] = [e.server_spec() for e in engines]

    # enforce_eager=False keeps CUDA graphs and torch.compile on.
    if enforce_eager:
        specs = [
            s
            if "--enforce-eager" in s.cli
            else dataclasses.replace(s, cli=(*s.cli, "--enforce-eager"))
            for s in specs
        ]

    run_id = uuid.uuid4().hex[:12]
    arch, gpu_name = gpu_info()
    vllm_v = vllm_version()

    # Shuffle the cells so Default and TM interleave.
    cells = [
        (spec, conc, r)
        for spec in specs
        for conc in concurrency
        for r in range(repeats)
    ]
    random.Random(seed).shuffle(cells)

    clocks_locked, effective_mhz = False, None
    if lock_gpu_clocks:
        clocks_locked, effective_mhz = _lock_gpu_clocks(gpu_clock_mhz)

    write_json(
        out_dir / f"{run_id}.run_manifest.json",
        {
            "run_id": run_id,
            "gpu_arch": arch,
            "gpu_name": gpu_name,
            "vllm_version": vllm_v,
            "seed": seed,
            "engines": [s.key for s in specs],
            "concurrency": list(concurrency),
            "repeats": repeats,
            "sampling_params": sampling_params,
            "enforce_eager": enforce_eager,
            "gpu_clocks_locked": clocks_locked,
            "gpu_clock_mhz": effective_mhz,
            "cell_order": [
                {"engine": s.key, "concurrency": c, "repeat_idx": r}
                for (s, c, r) in cells
            ],
            "outputs": {
                s.key: {
                    "csv": str(perf_csv_path(out_dir, gpu_name, run_id, s.key)),
                    "log": str(serve_log_path(out_dir, gpu_name, run_id, s.key)),
                }
                for s in specs
            },
            "timestamp": now(),
        },
    )

    print(
        f"[perf] out_dir={out_dir} run_id={run_id} cells={len(cells)} "
        f"engines={[s.key for s in specs]} concurrency={list(concurrency)} "
        f"repeats={repeats} clocks_locked={clocks_locked}",
        flush=True,
    )

    try:
        for spec, conc, repeat_idx in cells:
            t_cell = time.perf_counter()
            log_path = serve_log_path(out_dir, gpu_name, run_id, spec.key)
            csv_path = perf_csv_path(out_dir, gpu_name, run_id, spec.key)
            effective_max_requests = max(max_requests, requests_per_concurrency * conc)

            tag = f"{spec.key} | c={conc} | r={repeat_idx}"
            print(f"[{tag}] setup...", flush=True)
            server = VLLMServer(spec, port=port, log_path=log_path)
            error: str | None = None
            test_summary: dict = {}
            vram_summary: dict = {}
            batch_summary: dict = {}
            t_run = 0.0

            try:
                server.start(timeout_s=server_timeout_s)
                print(
                    f"[{tag}] ready ({time.perf_counter() - t_cell:.1f}s)", flush=True
                )
                warmup(server.base_url, api_key, spec.model_id, warmup_requests, conc)

                t_run_start = time.perf_counter()
                with (
                    VRAMSampler(server_pgid=server.pgid, hz=vram_hz) as vram,
                    BatchSampler(base_url=server.base_url, hz=batch_hz) as batch,
                ):
                    test_summary = run_load_test(
                        model_id=spec.model_id,
                        base_url=server.base_url,
                        api_key=api_key,
                        concurrency=conc,
                        max_requests=effective_max_requests,
                        mean_input_tokens=mean_input_tokens,
                        stddev_input_tokens=stddev_input_tokens,
                        mean_output_tokens=mean_output_tokens,
                        stddev_output_tokens=stddev_output_tokens,
                        timeout_s=timeout_s,
                        sampling_params=sampling_params,
                        seed=seed,
                    )
                t_run = time.perf_counter() - t_run_start
                vram_summary = {
                    "proc_peak_vram_mb": vram.proc_peak_mb,
                    "proc_mean_vram_mb": vram.proc_mean_mb,
                    "device_peak_vram_mb": vram.device_peak_mb,
                    "vram_source": vram.vram_source,
                }
                batch_summary = {
                    "batch_running_mean": batch.running_mean,
                    "batch_running_p50": batch.running_p50,
                    "batch_running_p90": batch.running_p90,
                    "batch_running_max": batch.running_max,
                    "batch_waiting_mean": batch.waiting_mean,
                }
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                print(f"[{tag}] ERROR\n{traceback.format_exc()}", flush=True)
            finally:
                server.stop()
                print(f"[{tag}] teardown", flush=True)

            mem = memory_from_vllm_log(log_path)
            if error is None and not mem:
                print(
                    f"[{tag}] WARNING: no memory fields parsed from {log_path}; "
                    f"the vllm-log patterns may be stale for this vLLM version",
                    flush=True,
                )
            mem_row: dict = {}
            if "kv_cache_gib" in mem:
                mem_row["kv_cache_mb"] = mem["kv_cache_gib"] * 1024
            if "peak_activation_gib" in mem:
                mem_row["peak_activation_mb"] = mem["peak_activation_gib"] * 1024
            if "gpu_blocks" in mem:
                mem_row["gpu_blocks"] = mem["gpu_blocks"]

            row = {col: "" for col in PERF_COLUMNS}
            row.update(
                {
                    "run_id": run_id,
                    "gpu_arch": arch,
                    "gpu_name": gpu_name,
                    "engine": spec.key,
                    "vllm_version": vllm_v,
                    "model_id": spec.model_id,
                    "concurrency": conc,
                    "repeat_idx": repeat_idx,
                    "mean_input_tokens": mean_input_tokens,
                    "stddev_input_tokens": stddev_input_tokens,
                    "mean_output_tokens": mean_output_tokens,
                    "stddev_output_tokens": stddev_output_tokens,
                    "effective_max_requests": effective_max_requests,
                    "duration_s": round(t_run, 3),
                    "timestamp": now(),
                    "error": error or "",
                }
            )
            row.update(vram_summary)
            row.update(batch_summary)
            row.update(mem_row)
            row.update({k: v for k, v in test_summary.items() if v is not None})
            append_perf_row(csv_path, row)

            print(
                f"[{tag}] done ({time.perf_counter() - t_cell:.1f}s cell, "
                f"{t_run:.1f}s measured)",
                flush=True,
            )
    finally:
        if clocks_locked:
            _reset_gpu_clocks()
            print("[perf] GPU clocks reset", flush=True)

    return out_dir
