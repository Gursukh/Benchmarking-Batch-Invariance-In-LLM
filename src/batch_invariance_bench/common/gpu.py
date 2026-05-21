from __future__ import annotations


def gpu_info() -> tuple[str, str]:
    """Return (arch, name), e.g. ('sm_90', 'NVIDIA H100'), or ('cpu', 'cpu')."""
    try:
        import torch
    except ImportError:
        return "cpu", "cpu"
    if not torch.cuda.is_available():
        return "cpu", "cpu"
    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    return f"sm_{major}{minor}", name


def vllm_version() -> str:
    """vLLM's installed version, or 'unknown'."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("vllm")
    except PackageNotFoundError:
        return "unknown"


def assert_single_gpu() -> None:
    """Raise if more than one CUDA device is visible.

    gpu_name reports device 0 only; multi-GPU runs would silently mislabel
    rows. Set CUDA_VISIBLE_DEVICES=0 on multi-GPU hosts.
    """
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    n = torch.cuda.device_count()
    if n > 1:
        raise RuntimeError(
            f"{n} CUDA devices visible; this harness is single-GPU only. "
            f"Set CUDA_VISIBLE_DEVICES=<one index>."
        )
