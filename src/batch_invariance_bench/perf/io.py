from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Mapping

from batch_invariance_bench.io import _slug


PERF_COLUMNS = [
    "run_id",
    "gpu_arch",
    "gpu_name",
    "engine",
    "vllm_version",
    "model_id",
    "concurrency",
    "mean_input_tokens",
    "stddev_input_tokens",
    "mean_output_tokens",
    "stddev_output_tokens",
    "num_completed",
    "num_errors",
    "error_rate",
    "ttft_mean_s",
    "ttft_p50_s",
    "ttft_p90_s",
    "ttft_p95_s",
    "ttft_p99_s",
    "itl_mean_s",
    "itl_p50_s",
    "itl_p90_s",
    "itl_p95_s",
    "itl_p99_s",
    "e2e_mean_s",
    "e2e_p50_s",
    "e2e_p90_s",
    "e2e_p95_s",
    "e2e_p99_s",
    "req_output_throughput_mean",
    "overall_output_throughput",
    "peak_vram_mb",
    "mean_vram_mb",
    "duration_s",
    "timestamp",
]


def cell_dir(out_path: str | os.PathLike, gpu_name: str, engine_name: str, concurrency: int) -> Path:
    return Path(out_path) / f"{_slug(gpu_name)}.{_slug(engine_name)}.c{concurrency}"


def perf_csv_path(out_path: str | os.PathLike, gpu_name: str) -> Path:
    return Path(out_path) / f"{_slug(gpu_name)}.perf.csv"


def append_perf_row(path: str | os.PathLike, row: Mapping[str, object]) -> None:
    """Append one row to the perf CSV. Writes a header if the file is new."""
    p = Path(path)
    is_new = not p.exists() or p.stat().st_size == 0
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PERF_COLUMNS, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def summary_from_llmperf(d: Mapping[str, object]) -> dict:
    """LLMPerf summary -> our flat row schema. Missing keys come back as None
    so callers can filter them before writing the CSV."""
    num_completed = d.get("num_completed_requests") or 0
    num_errors    = d.get("number_errors") or 0
    total = num_completed + num_errors
    error_rate = (num_errors / total) if total else 0.0

    return {
        "num_completed": num_completed,
        "num_errors":    num_errors,
        "error_rate":    error_rate,
        "ttft_mean_s":   d.get("ttft_s_mean"),
        "ttft_p50_s":    d.get("ttft_s_quantiles_p50"),
        "ttft_p90_s":    d.get("ttft_s_quantiles_p90"),
        "ttft_p95_s":    d.get("ttft_s_quantiles_p95"),
        "ttft_p99_s":    d.get("ttft_s_quantiles_p99"),
        "itl_mean_s":    d.get("inter_token_latency_s_mean"),
        "itl_p50_s":     d.get("inter_token_latency_s_quantiles_p50"),
        "itl_p90_s":     d.get("inter_token_latency_s_quantiles_p90"),
        "itl_p95_s":     d.get("inter_token_latency_s_quantiles_p95"),
        "itl_p99_s":     d.get("inter_token_latency_s_quantiles_p99"),
        "e2e_mean_s":    d.get("end_to_end_latency_s_mean"),
        "e2e_p50_s":     d.get("end_to_end_latency_s_quantiles_p50"),
        "e2e_p90_s":     d.get("end_to_end_latency_s_quantiles_p90"),
        "e2e_p95_s":     d.get("end_to_end_latency_s_quantiles_p95"),
        "e2e_p99_s":     d.get("end_to_end_latency_s_quantiles_p99"),
        "req_output_throughput_mean": d.get("request_output_throughput_token_per_s_mean"),
        "overall_output_throughput":  d.get("mean_output_throughput_token_per_s"),
    }


