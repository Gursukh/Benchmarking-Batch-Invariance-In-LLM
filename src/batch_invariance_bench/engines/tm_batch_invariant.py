from __future__ import annotations

from batch_invariance_bench.common.env import apply_env, restore_env
from batch_invariance_bench.engines.base import VLLMBase


# vLLM reads VLLM_BATCH_INVARIANT when the engine is built.
VLLM_BATCH_INVARIANT_ENV: dict[str, str] = {"VLLM_BATCH_INVARIANT": "1"}


class VLLMTMBatchInvariant(VLLMBase):
    """vLLM with Thinking Machines' batch-invariant ops enabled.

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
        self._prev_env = apply_env(VLLM_BATCH_INVARIANT_ENV)
        super().setup()

    def teardown(self) -> None:
        try:
            super().teardown()
        finally:
            restore_env(self._prev_env)
            self._prev_env = {}
