from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def run_llmperf(
    *,
    model_id: str,
    base_url: str,
    api_key: str,
    concurrency: int,
    max_requests: int,
    mean_input_tokens: int,
    stddev_input_tokens: int,
    mean_output_tokens: int,
    stddev_output_tokens: int,
    timeout_s: float,
    results_dir: Path,
) -> dict:
    """Shell out to llmperf, return the parsed summary dict it writes to disk."""
    results_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "llmperf.token_benchmark_ray",
        "--model", model_id,
        "--llm-api", "openai",
        "--mean-input-tokens", str(mean_input_tokens),
        "--stddev-input-tokens", str(stddev_input_tokens),
        "--mean-output-tokens", str(mean_output_tokens),
        "--stddev-output-tokens", str(stddev_output_tokens),
        "--num-concurrent-requests", str(concurrency),
        "--max-num-completed-requests", str(max_requests),
        "--timeout", str(int(timeout_s)),
        "--results-dir", str(results_dir),
        "--additional-sampling-params", "{}",
    ]

    env = {
        **os.environ,
        "OPENAI_API_BASE": base_url,
        "OPENAI_API_KEY":  api_key,
    }

    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"llmperf exited {proc.returncode}\n"
            f"stdout:\n{proc.stdout[-2000:]}\n"
            f"stderr:\n{proc.stderr[-2000:]}"
        )

    summaries = sorted(results_dir.glob("*_summary.json"))
    if not summaries:
        raise RuntimeError(
            f"llmperf produced no *_summary.json in {results_dir}; "
            f"stdout tail:\n{proc.stdout[-1000:]}"
        )
    return json.loads(summaries[-1].read_text())
