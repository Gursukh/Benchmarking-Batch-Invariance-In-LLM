from __future__ import annotations

from batch_invariance_bench.common.env import apply_env, restore_env
from batch_invariance_bench.engines.base import VLLMBase


# The Thinking Machines batch-invariant switch, used by both the in-process
# engine (setup) and the server path (_server_env).
VLLM_BATCH_INVARIANT_ENV: dict[str, str] = {"VLLM_BATCH_INVARIANT": "1"}


class VLLMTMBatchInvariant(VLLMBase):
    """vLLM with Thinking Machines' batch-invariant ops turned on.

    https://github.com/thinking-machines-lab/batch_invariant_ops
    """

    label = "TM"

    def __init__(
        self,
        name: str | None = None,
        vllm_kwargs: dict | None = None,
        sampling: dict | None = None,
    ) -> None:
        super().__init__(name=name, vllm_kwargs=vllm_kwargs, sampling=sampling)
        self._prev_env: dict[str, str | None] = {}

    def setup(self) -> None:
        # vLLM reads VLLM_BATCH_INVARIANT when the engine is built; set it first.
        self._prev_env = apply_env(VLLM_BATCH_INVARIANT_ENV)
        super().setup()

    def teardown(self) -> None:
        try:
            super().teardown()
        finally:
            restore_env(self._prev_env)
            self._prev_env = {}

    def _server_env(self) -> dict[str, str]:
        """Env the server needs to enable batch-invariant ops."""
        return dict(VLLM_BATCH_INVARIANT_ENV)


# class Llama3VLLMTMBatchInvariant(VLLMTMBatchInvariant):
#     hf_id = "meta-llama/Meta-Llama-3-8B-Instruct"
#     max_model_len = 8192
