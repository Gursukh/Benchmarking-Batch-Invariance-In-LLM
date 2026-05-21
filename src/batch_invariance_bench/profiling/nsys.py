"""Run a command under Nsight Systems (nsys) for timeline profiling."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Sequence


def profile_with_nsys(
    cmd: Sequence[str],
    out_path: str | Path,
    *,
    trace: str = "cuda,nvtx,osrt",
    extra_args: Sequence[str] = (),
) -> Path:
    """Profile cmd with nsys and save the report next to out_path.

    trace is the comma separated trace set, e.g. cuda, nvtx, osrt.
    """
    if shutil.which("nsys") is None:
        raise FileNotFoundError("nsys (Nsight Systems) not found on PATH")
    report = Path(out_path).with_suffix(".nsys-rep")
    report.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        "nsys",
        "profile",
        "-o",
        str(report.with_suffix("")),
        f"--trace={trace}",
        "--force-overwrite=true",
        *extra_args,
        *cmd,
    ]
    subprocess.run(argv, check=True)
    return report
