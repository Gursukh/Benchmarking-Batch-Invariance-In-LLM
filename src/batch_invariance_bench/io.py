from __future__ import annotations

import csv
import datetime as _dt
import os
import re
from pathlib import Path
from typing import Iterable, Mapping


CSV_COLUMNS = [
    "run_id",
    "gpu_arch",
    "gpu_name",
    "engine",
    "task",
    "batch_size",
    "problem_id",
    "n_samples",
    "pass_at_1",
    "pass_at_4",
    "mean_resp_len",
    "std_resp_len",
    "completions_json",
    "correct_mask",
    "seed",
    "vllm_version",
    "timestamp",
]


def gpu_info() -> tuple[str, str]:
    """(arch, name) — e.g. ('sm_90', 'NVIDIA H100'). ('cpu', 'cpu') if no GPU."""
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
    try:
        import vllm
    except ImportError:
        return "unknown"
    return getattr(vllm, "__version__", "unknown")


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_") or "x"


def default_output_path(out_dir: str | os.PathLike = "results") -> Path:
    _, name = gpu_info()
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    p = Path(out_dir) / f"{_slug(name)}_{ts}.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def append_rows(path: str | os.PathLike, rows: Iterable[Mapping[str, object]]) -> None:
    """Append rows; write the header iff the file is new/empty."""
    p = Path(path)
    is_new = not p.exists() or p.stat().st_size == 0
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
