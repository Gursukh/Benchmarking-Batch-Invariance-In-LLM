"""CSV schema for the correctness (batch-invariance) harness."""

from __future__ import annotations


OUTPUT_COLUMNS = [
    "run_id",
    "gpu_arch",
    "gpu_name",
    "engine",
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
