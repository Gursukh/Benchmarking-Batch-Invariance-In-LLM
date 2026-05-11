from __future__ import annotations

import csv
import datetime as _dt
import threading
import time
import warnings
from pathlib import Path


class VRAMSampler:
    """Context-manager that polls NVML at a fixed Hz on a background thread.
    `peak_mb` / `mean_mb` return nan if NVML isn't usable."""

    def __init__(
        self,
        device_index: int = 0,
        hz: float = 5.0,
        out_csv: Path | None = None,
    ) -> None:
        self.device_index = device_index
        self.hz = hz
        self.out_csv = out_csv
        self._samples: list[tuple[float, float]] = []   # (ts, used_mb)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._handle = None
        self._nvml = None
        self._writer = None
        self._fp = None
        self._available = False

    def __enter__(self) -> "VRAMSampler":
        try:
            import pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            self._nvml = pynvml
            self._available = True
        except Exception as e:
            warnings.warn(f"VRAM sampling disabled: {e}", stacklevel=2)
            return self

        if self.out_csv is not None:
            self.out_csv.parent.mkdir(parents=True, exist_ok=True)
            self._fp = self.out_csv.open("w", newline="")
            self._writer = csv.writer(self._fp)
            self._writer.writerow(["timestamp_iso", "used_mb", "free_mb", "total_mb"])

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
        if self._fp is not None:
            self._fp.close()

    def _loop(self) -> None:
        period = 1.0 / max(self.hz, 0.1)
        while not self._stop.is_set():
            try:
                info = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
                used_mb  = info.used  / (1024 ** 2)
                free_mb  = info.free  / (1024 ** 2)
                total_mb = info.total / (1024 ** 2)
                ts = time.time()
                self._samples.append((ts, used_mb))
                if self._writer is not None:
                    iso = _dt.datetime.fromtimestamp(ts, _dt.timezone.utc).isoformat()
                    self._writer.writerow([iso, f"{used_mb:.2f}", f"{free_mb:.2f}", f"{total_mb:.2f}"])
                    self._fp.flush()
            except Exception:
                pass
            self._stop.wait(period)

    @property
    def peak_mb(self) -> float:
        if not self._samples:
            return float("nan")
        return max(s[1] for s in self._samples)

    @property
    def mean_mb(self) -> float:
        if not self._samples:
            return float("nan")
        return sum(s[1] for s in self._samples) / len(self._samples)
