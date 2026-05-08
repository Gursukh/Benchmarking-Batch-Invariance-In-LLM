from __future__ import annotations

from batch_invariance_bench.engines.vllm_base import VLLMBase


class VLLMDefault(VLLMBase):
    """Stock vLLM, no batch-invariance patches."""


# class Llama3VLLMDefault(VLLMDefault):
#     hf_id = "meta-llama/Meta-Llama-3-8B-Instruct"
#     max_model_len = 8192
