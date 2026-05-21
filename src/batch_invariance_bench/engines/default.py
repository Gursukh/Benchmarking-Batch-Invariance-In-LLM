from __future__ import annotations

from batch_invariance_bench.engines.base import VLLMBase


class VLLMDefault(VLLMBase):
    """Stock vLLM, no batch-invariance patches."""

    label = "Default"
