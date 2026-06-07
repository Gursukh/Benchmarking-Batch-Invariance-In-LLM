from __future__ import annotations


def gpu_info() -> tuple[str, str]:
    """(arch, name), e.g. ('sm_90', 'NVIDIA H100'), else ('cpu', 'cpu')."""
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
    """Installed vLLM version, or 'unknown'."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("vllm")
    except PackageNotFoundError:
        return "unknown"