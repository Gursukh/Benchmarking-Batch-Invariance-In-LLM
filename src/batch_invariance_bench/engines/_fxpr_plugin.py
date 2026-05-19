"""vLLM plugin shim for the fxpr engine.

Registered as a `vllm.general_plugins` entry point named `fxpr` in
pyproject.toml. vLLM calls this in every process whose VLLM_PLUGINS lists
`fxpr`, so the server runs the same kernels as the in-process engine.

fxpr_vllm is optional. register() raises a clear error if it is missing.
"""

from __future__ import annotations


def register() -> None:
    """Entry point: register fxpr_vllm's kernels in this process."""
    try:
        from fxpr_vllm.register import register as _fxpr_register
    except ImportError as e:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "the `fxpr` vLLM plugin is enabled (VLLM_PLUGINS contains 'fxpr') "
            "but fxpr_vllm is not installed in this environment"
        ) from e
    _fxpr_register()
