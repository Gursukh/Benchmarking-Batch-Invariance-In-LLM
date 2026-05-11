from __future__ import annotations

import os

from batch_invariance_bench.engines.vllm_base import VLLMBase


class VLLMFxpr(VLLMBase):
    """vLLM with fxpr_vllm fixed-point-reduction kernels."""

    quantization: str | None = "fixed_point_det"
    attention_backend: str | None = "CUSTOM"
    fxp_int_bits: int = 16
    fxp_frac_bits: int = 8

    def __init__(
        self,
        name: str | None = None,
        *,
        quantization: str | None = None,
        attention_backend: str | None = None,
        fxp_int_bits: int | None = None,
        fxp_frac_bits: int | None = None,
    ) -> None:
        super().__init__(name=name)
        if quantization is not None:
            self.quantization = quantization
        if attention_backend is not None:
            self.attention_backend = attention_backend
        if fxp_int_bits is not None:
            self.fxp_int_bits = fxp_int_bits
        if fxp_frac_bits is not None:
            self.fxp_frac_bits = fxp_frac_bits
        self._prev_env: dict[str, str | None] = {}

    def setup(self) -> None:
        # Save existing env so teardown can restore it exactly.
        for key in ("VLLM_FXP_INT_BITS", "VLLM_FXP_FRAC_BITS"):
            self._prev_env[key] = os.environ.get(key)
        os.environ["VLLM_FXP_INT_BITS"] = str(self.fxp_int_bits)
        os.environ["VLLM_FXP_FRAC_BITS"] = str(self.fxp_frac_bits)

        from fxpr_vllm.register import register

        register()

        extra: dict = {}
        if self.quantization is not None:
            extra["quantization"] = self.quantization
        if self.attention_backend is not None:
            extra["attention_backend"] = self.attention_backend
        self.vllm_kwargs = {**self.vllm_kwargs, **extra}

        super().setup()

    def teardown(self) -> None:
        try:
            super().teardown()
        finally:
            for key, prev in self._prev_env.items():
                if prev is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = prev
            self._prev_env = {}
