from __future__ import annotations

import os

from batch_invariance_bench.engines.vllm_base import VLLMBase


class VLLMTMBatchInvariant(VLLMBase):
    """vLLM with Thinking Machines' batch-invariant ops turned on.

    Activated via the VLLM_BATCH_INVARIANT=1 env var. See
    https://github.com/thinking-machines-lab/batch_invariant_ops.
    """

    def __init__(self, name: str | None = None) -> None:
        super().__init__(name=name)
        self._prev_env: str | None = None

    def setup(self) -> None:
        # Save the existing value so teardown can restore it.
        self._prev_env = os.environ.get("VLLM_BATCH_INVARIANT")
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
        super().setup()

    def teardown(self) -> None:
        try:
            super().teardown()
        finally:
            if self._prev_env is None:
                os.environ.pop("VLLM_BATCH_INVARIANT", None)
            else:
                os.environ["VLLM_BATCH_INVARIANT"] = self._prev_env
            self._prev_env = None


# class Llama3VLLMTMBatchInvariant(VLLMTMBatchInvariant):
#     hf_id = "meta-llama/Meta-Llama-3-8B-Instruct"
#     max_model_len = 8192
