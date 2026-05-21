"""Run a command under Nsight Compute (ncu) for kernel profiling."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Sequence


def profile_with_ncu(
    cmd: Sequence[str],
    out_path: str | Path,
    *,
    set_name: str = "full",
    extra_args: Sequence[str] = (),
) -> Path:
    """Profile cmd with ncu and save the report next to out_path.

    set_name picks the metric set: full, basic or roofline.
    """
    if shutil.which("ncu") is None:
        raise FileNotFoundError("ncu (Nsight Compute) not found on PATH")
    report = Path(out_path).with_suffix(".ncu-rep")
    report.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        "ncu",
        "--export",
        str(report),
        "--set",
        set_name,
        "--force-overwrite",
        *extra_args,
        *cmd,
    ]
    subprocess.run(argv, check=True)
    return report
