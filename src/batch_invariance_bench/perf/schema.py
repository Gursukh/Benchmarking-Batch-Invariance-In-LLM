"""CSV schema and output-path helpers for the perf harness."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

from batch_invariance_bench.common.csvio import append_csv_rows, slug


PERF_COLUMNS = [
    "run_id",
    "gpu_arch",
    "gpu_name",
    "engine",
    "vllm_version",
    "model_id",
    "concurrency",
    "repeat_idx",
    "mean_input_tokens",
    "stddev_input_tokens",
    "mean_output_tokens",
    "stddev_output_tokens",
    "effective_max_requests",
    "num_completed",
    "num_errors",
    "error_rate",
    "ttft_mean_s",
    "ttft_p50_s",
    "ttft_p90_s",
    "ttft_p95_s",
    # itl_* are pooled across every decoded token of every request,
    # not per-request means. See perf/load_test.py.
    "itl_mean_s",
    "itl_p50_s",
    "itl_p90_s",
    "itl_p95_s",
    "e2e_mean_s",
    "e2e_p50_s",
    "e2e_p90_s",
    "e2e_p95_s",
    "req_output_throughput_mean",
    "overall_output_throughput",
    "batch_running_mean",
    "batch_running_p50",
    "batch_running_p90",
    "batch_running_max",
    "batch_waiting_mean",
    "proc_peak_vram_mb",
    "proc_mean_vram_mb",
    "device_peak_vram_mb",
    "vram_source",
    "kv_cache_mb",
    "peak_activation_mb",
    "gpu_blocks",
    "duration_s",
    "timestamp",
    "error",
]


def perf_csv_path(
    out_path: str | os.PathLike,
    gpu_name: str,
    run_id: str,
    engine_name: str,
) -> Path:
    """Path of the per-engine perf CSV. The run id keeps runs in separate files."""
    return Path(out_path) / (f"{slug(gpu_name)}.{run_id}.{slug(engine_name)}.perf.csv")


def serve_log_path(
    out_path: str | os.PathLike,
    gpu_name: str,
    run_id: str,
    engine_name: str,
) -> Path:
    """Path of the per-engine `vllm serve` log.

    Each cell of an engine overwrites it; memory is parsed from it before the
    next cell of that engine runs.
    """
    return Path(out_path) / (f"{slug(gpu_name)}.{run_id}.{slug(engine_name)}.serve.log")


def append_perf_row(path: str | os.PathLike, row: Mapping[str, object]) -> None:
    """Append one row to the perf CSV (schema: PERF_COLUMNS)."""
    append_csv_rows(path, [row], PERF_COLUMNS)
