from __future__ import annotations

from batch_invariance_bench.common.env import apply_env, restore_env
from batch_invariance_bench.engines.base import VLLMBase


# vLLM plugin name. The matching entry point in pyproject.toml runs
# fxpr_vllm.register() in any vLLM process whose VLLM_PLUGINS lists it.
FXPR_PLUGIN = "fxpr"


class VLLMFxpr(VLLMBase):
    """vLLM with fxpr_vllm fixed-point-reduction kernels."""

    label = "FXPR"

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
        vllm_kwargs: dict | None = None,
        sampling: dict | None = None,
    ) -> None:
        super().__init__(name=name, vllm_kwargs=vllm_kwargs, sampling=sampling)
        if quantization is not None:
            self.quantization = quantization
        if attention_backend is not None:
            self.attention_backend = attention_backend
        if fxp_int_bits is not None:
            self.fxp_int_bits = fxp_int_bits
        if fxp_frac_bits is not None:
            self.fxp_frac_bits = fxp_frac_bits
        self._prev_env: dict[str, str | None] = {}

    def _fxp_env(self) -> dict[str, str]:
        """fxpr kernel bit-width env, read at registration time."""
        return {
            "VLLM_FXP_INT_BITS": str(self.fxp_int_bits),
            "VLLM_FXP_FRAC_BITS": str(self.fxp_frac_bits),
        }

    def setup(self) -> None:
        # In-process: set the bit-width env, then register fxpr's kernels.
        # The server path does this through the `fxpr` plugin instead.
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

    def _server_env(self) -> dict[str, str]:
        return self._fxp_env()

    def _server_cli(self) -> list[str]:
        cli: list[str] = []
        if self.quantization is not None:
            cli += ["--quantization", self.quantization]
        if self.attention_backend is not None:
            cli += ["--attention-backend", self.attention_backend]
        return cli

    def _server_plugins(self) -> list[str]:
        # Without this plugin the server never runs fxpr_vllm.register() and
        # silently falls back to stock vLLM kernels. See _fxpr_plugin.py.
        return [FXPR_PLUGIN]

    def teardown(self) -> None:
        try:
            super().teardown()
        finally:
            restore_env(self._prev_env)
            self._prev_env = {}
