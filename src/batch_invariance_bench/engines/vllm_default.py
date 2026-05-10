from __future__ import annotations

from batch_invariance_bench.engines.vllm_base import VLLMBase


class VLLMDefault(VLLMBase):
    """Stock vLLM with no batch-invariance patches applied."""


# Example of swapping in a different model:
# class Llama3VLLMDefault(VLLMDefault):
#     hf_id = "meta-llama/Meta-Llama-3-8B-Instruct"
#     max_model_len = 8192
