"""CSV schema and output paths for the perf harness."""

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
    "engine_label",
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
    # ttft: prefill only, the first batch of arrivals with no queue wait.
    "ttft_mean_s",
    "ttft_p50_s",
    "ttft_p90_s",
    "ttft_p95_s",
    # submit to first token: every request, including queued ones.
    "submit_to_first_token_mean_s",
    "submit_to_first_token_p50_s",
    "submit_to_first_token_p90_s",
    "submit_to_first_token_p95_s",
    # itl: pooled inter token gaps, measured at scheduler steps.
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
    return Path(out_path) / f"{slug(gpu_name)}.{run_id}.{slug(engine_name)}.perf.csv"


def append_perf_row(path: str | os.PathLike, row: Mapping[str, object]) -> None:
    append_csv_rows(path, [row], PERF_COLUMNS)
