from __future__ import annotations

from typing import Any

from batch_invariance_bench.common.env import apply_env, restore_env
from batch_invariance_bench.engines.base import VLLMBase


# Distinguishes "argument omitted" from an explicit None (None means "disable").
_UNSET: Any = object()


class VLLMFxpr(VLLMBase):
    """vLLM with fxpr_vllm fixed-point reduction kernels."""

    label = "FXPR"

    quantization: str | None = "fixedpoint"
    attention_backend: str | None = "CUSTOM"
    fxp_int_bits: int = 32
    fxp_frac_bits: int = 16

    def __init__(
        self,
        name: str | None = None,
        *,
        quantization: str | None = _UNSET,
        attention_backend: str | None = _UNSET,
        fxp_int_bits: int | None = None,
        fxp_frac_bits: int | None = None,
        vllm_kwargs: dict | None = None,
        sampling: dict | None = None,
    ) -> None:
        super().__init__(name=name, vllm_kwargs=vllm_kwargs, sampling=sampling)
        if quantization is not _UNSET:
            self.quantization = quantization
        if attention_backend is not _UNSET:
            self.attention_backend = attention_backend
        if fxp_int_bits is not None:
            self.fxp_int_bits = fxp_int_bits
        if fxp_frac_bits is not None:
            self.fxp_frac_bits = fxp_frac_bits
        self._prev_env: dict[str, str | None] = {}

    def _fxp_env(self) -> dict[str, str]:
        """Bit-width env, read by fxpr at kernel registration."""
        return {
            "FXPR_INT_BITS": str(self.fxp_int_bits),
            "FXPR_FRAC_BITS": str(self.fxp_frac_bits),
        }

    def setup(self) -> None:
        # Bit-width env must be set before register() so fxpr picks it up.
        self._prev_env = apply_env(self._fxp_env())
        from fxpr_vllm.register import register

        register()
        super().setup()

    def _extra_vllm_kwargs(self) -> dict:
        extra: dict = {}
        if self.quantization is not None:
            extra["quantization"] = self.quantization
        if self.attention_backend is not None:
            extra["attention_backend"] = self.attention_backend
        return extra

    def teardown(self) -> None:
        try:
            super().teardown()
        finally:
            restore_env(self._prev_env)
            self._prev_env = {}
