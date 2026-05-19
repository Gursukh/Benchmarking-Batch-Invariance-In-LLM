"""Best-effort memory-profiling numbers parsed from a `vllm serve` log."""

from __future__ import annotations

import os
import re
from pathlib import Path


def memory_from_vllm_log(log_path: str | os.PathLike) -> dict:
    """Parse memory numbers out of a `vllm serve` log, best effort.

    These log lines change between vLLM versions, so every field is optional
    and just absent if not matched. *_gib values are in GiB, gpu_blocks is a
    count. An empty result on a healthy run means the patterns are stale; the
    perf runner warns when that happens.
    """
    p = Path(log_path)
    if not p.exists():
        return {}
    text = p.read_text(errors="replace")
    out: dict[str, float] = {}

    m = re.search(r"Available KV cache memory:\s*([\d.]+)\s*GiB", text)
    if m:
        out["kv_cache_gib"] = float(m.group(1))

    m = re.search(r"#?\s*GPU blocks:\s*([\d,]+)", text)
    if m:
        out["gpu_blocks"] = float(m.group(1).replace(",", ""))

    m = re.search(
        r"(?:torch peak memory|peak activation memory|activation peak memory)"
        r"[^\d]*([\d.]+)\s*GiB",
        text,
        re.IGNORECASE,
    )
    if m:
        out["peak_activation_gib"] = float(m.group(1))

    return out
