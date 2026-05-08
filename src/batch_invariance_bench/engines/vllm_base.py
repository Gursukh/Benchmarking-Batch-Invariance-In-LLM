from __future__ import annotations

import gc

import torch
from vllm import LLM, SamplingParams

from batch_invariance_bench.engine import Engine


DEFAULT_SAMPLING: dict = {
    "temperature": 0,
    "top_p": 0.95,
    "max_tokens": 2048,
}


class VLLMBase(Engine):
    """Shared vLLM plumbing. Subclass to toggle batch-invariance modes."""

    hf_id: str = "Qwen/Qwen3-0.6B"
    dtype: str = "bfloat16"
    max_model_len: int = 4096
    vllm_kwargs: dict = {}

    default_sampling: dict = DEFAULT_SAMPLING

    def __init__(self, name: str | None = None) -> None:
        self.name = name or f"{type(self).__name__}::{self.hf_id}"
        self._llm: LLM | None = None

    def setup(self) -> None:
        self._llm = LLM(
            model=self.hf_id,
            dtype=self.dtype,
            max_model_len=self.max_model_len,
            **self.vllm_kwargs,
        )

    def generate(
        self,
        prompts: list[str],
        n: int,
        sampling: dict | None = None,
    ) -> list[list[str]]:
        assert self._llm is not None, "call setup() first"
        params = SamplingParams(n=n, **{**self.default_sampling, **(sampling or {})})
        outputs = self._llm.generate(prompts, params)
        return [[o.text for o in req.outputs] for req in outputs]

    def teardown(self) -> None:
        del self._llm
        self._llm = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
