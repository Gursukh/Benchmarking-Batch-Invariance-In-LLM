"""CSV output and the per-run provenance columns stamped on every row."""

from __future__ import annotations

import argparse
import csv
import os
import platform
import re
import uuid
from datetime import datetime, timezone
from importlib import metadata as _im
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch

# relative to cwd so it works from an installed wheel; override with NONDET_RESULTS_DIR
RESULTS_ROOT = Path(os.environ.get("NONDET_RESULTS_DIR", "results")).resolve() / "Chapter3"


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _compact_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def gpu_info() -> tuple[str, str]:
    # (arch, name), or ("cpu", "cpu") with no GPU
    if not torch.cuda.is_available():
        return "cpu", "cpu"
    major, minor = torch.cuda.get_device_capability(0)
    return f"sm_{major}{minor}", torch.cuda.get_device_name(0)


def _version(pkg: str) -> str:
    try:
        return _im.version(pkg)
    except Exception:
        return "unknown"


def run_id() -> str:
    arch, _ = gpu_info()
    return f"{_compact_now()}_{arch}_{uuid.uuid4().hex[:6]}"


# one run id per process, shared by every CSV written
_RUN_ID = run_id()


def results_dir() -> Path:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    return RESULTS_ROOT


def _slug(s: str) -> str:
    # filesystem-safe token: keep alnum . + - , collapse the rest to _
    return re.sub(r"[^A-Za-z0-9.+-]+", "_", s).strip("_") or "unknown"


def gpu_model_filename(model: str) -> str:
    """<gpu>_<model>.csv: one file per hardware x model."""
    _, name = gpu_info()
    return f"{_slug(name)}_{_slug(model)}.csv"


def experiment_path(exp_name: str, filename: str) -> Path:
    """results/<exp_name>/<filename>, created on demand."""
    d = RESULTS_ROOT.parent / _slug(exp_name)
    d.mkdir(parents=True, exist_ok=True)
    return d / filename


def base_parser() -> argparse.ArgumentParser:
    # shared args; scripts add their own then call env_from_args
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    return ap


def env_from_args(args) -> dict:
    return env_columns(model=args.model, dtype=args.dtype)


def env_columns(model: str = "", dtype: str = "") -> dict:
    arch, name = gpu_info()
    return {
        "run_id": _RUN_ID,
        "timestamp": now(),
        "gpu_arch": arch,
        "gpu_name": name,
        "vllm_version": _version("vllm"),
        "torch_version": torch.__version__,
        "transformers_version": _version("transformers"),
        "python": platform.python_version(),
        "model": model,
        "dtype": dtype,
    }


# env_columns() keys, for scripts to prepend to their fieldnames
ENV_FIELDS = list(env_columns().keys())


def append_csv_rows(
    path: str | os.PathLike,
    rows: Iterable[Mapping[str, object]],
    fieldnames: Sequence[str],
) -> Path:
    """Append rows to a CSV, writing the header first if the file is new."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def write_rows(
    filename: str,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str],
) -> Path:
    return append_csv_rows(results_dir() / filename, rows, fieldnames)
