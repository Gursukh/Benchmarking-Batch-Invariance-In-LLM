"""GPU memory sampling for the perf harness."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

from batch_invariance_bench.common.sampling import BackgroundSampler


class VRAMSampler(BackgroundSampler):
    """Polls NVML and records the GPU memory held by the server's process group.

    proc_* come from per-process NVML accounting for the server's pgid (every
    vLLM worker shares it). If that is not available (old driver, MIG, missing
    permissions) the sample falls back to whole-device usage and vram_source
    reports "device_fallback" instead of "per_process". The *_mb properties
    return nan when NVML itself is unusable.
    """

    csv_header = ("proc_used_mb", "device_used_mb", "device_total_mb")

    def __init__(
        self,
        server_pgid: int | None = None,
        device_index: int = 0,
        hz: float = 5.0,
        out_csv: Path | None = None,
    ) -> None:
        super().__init__(hz=hz, out_csv=out_csv)
        self.server_pgid = server_pgid
        self.device_index = device_index
        self._handle = None
        self._nvml = None
        self._per_process_ok = False

    def _open(self) -> bool:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            self._nvml = pynvml
            return True
        except Exception as e:
            warnings.warn(f"VRAM sampling disabled: {e}", stacklevel=2)
            return False

    def _close(self) -> None:
        if self._nvml is not None:
            self._nvml.nvmlShutdown()

    def _proc_used_mb(self) -> float | None:
        """GPU memory held by the server's process group.

        None when per-process accounting is not available this sample.
        """
        if self.server_pgid is None:
            return None
        try:
            procs = self._nvml.nvmlDeviceGetComputeRunningProcesses(self._handle)
        except Exception:
            return None
        total = 0
        matched = False
        for p in procs:
            try:
                if os.getpgid(p.pid) != self.server_pgid:
                    continue
            except (ProcessLookupError, PermissionError):
                continue
            used = getattr(p, "usedGpuMemory", None)
            if used is None:
                continue
            total += used
            matched = True
        if not matched:
            return None
        return total / (1024**2)

    def _sample(self) -> tuple | None:
        info = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
        device_used_mb = info.used / (1024**2)
        device_total_mb = info.total / (1024**2)
        proc_mb = self._proc_used_mb()
        if proc_mb is not None:
            self._per_process_ok = True
        # fall back to device usage when per-process is unavailable
        proc_used_mb = proc_mb if proc_mb is not None else device_used_mb
        return (proc_used_mb, device_used_mb, device_total_mb)

    def _format(self, values: tuple) -> list[str]:
        return [f"{v:.2f}" for v in values]

    # summary properties; each sample is (ts, proc_used, device_used, device_total)
    @property
    def vram_source(self) -> str:
        if not self._samples:
            return ""
        return "per_process" if self._per_process_ok else "device_fallback"

    @property
    def proc_peak_mb(self) -> float:
        col = self._column(1)
        return max(col) if col else float("nan")

    @property
    def proc_mean_mb(self) -> float:
        col = self._column(1)
        return sum(col) / len(col) if col else float("nan")

    @property
    def device_peak_mb(self) -> float:
        col = self._column(2)
        return max(col) if col else float("nan")
