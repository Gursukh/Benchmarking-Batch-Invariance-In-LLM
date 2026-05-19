"""Background polling sampler.

VRAMSampler and BatchSampler both poll something at a fixed rate on a daemon
thread inside a `with` block. BackgroundSampler holds the shared parts so each
one only writes _open(), _sample() and _close().
"""

from __future__ import annotations

import csv
import datetime as _dt
import threading
import time
from pathlib import Path
from typing import Sequence


def nearest_rank(values: Sequence[float], q: float) -> float:
    """Nearest-rank quantile of values, with q in [0, 1].

    NaNs are dropped. Returns NaN if nothing is left.
    """
    vals = [v for v in values if v == v]
    if not vals:
        return float("nan")
    s = sorted(vals)
    idx = min(len(s) - 1, int(q * (len(s) - 1) + 0.5))
    return s[idx]


class BackgroundSampler:
    """Polls _sample() at `hz` on a daemon thread for the life of a `with` block.

    A subclass sets csv_header and implements three hooks:
      _open()   return False to disable sampling, True otherwise.
      _sample() return one poll's values as a tuple, or None to skip it.
      _close()  release resources (optional).
    Each sample is stored with a timestamp, and written to out_csv if given.
    """

    csv_header: tuple[str, ...] = ()

    def __init__(self, hz: float = 5.0, out_csv: Path | None = None) -> None:
        self.hz = hz
        self.out_csv = out_csv
        # each entry is (timestamp, *sample_values)
        self._samples: list[tuple] = []
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._writer = None
        self._fp = None

    # subclass hooks
    def _open(self) -> bool:
        return True

    def _sample(self) -> tuple | None:
        raise NotImplementedError

    def _close(self) -> None:
        pass

    def _format(self, values: tuple) -> list[str]:
        """Format one sample's values for the CSV row. Subclasses may override."""
        return [str(v) for v in values]

    def __enter__(self) -> "BackgroundSampler":
        if not self._open():
            return self
        if self.out_csv is not None:
            self.out_csv.parent.mkdir(parents=True, exist_ok=True)
            self._fp = self.out_csv.open("w", newline="")
            self._writer = csv.writer(self._fp)
            self._writer.writerow(["timestamp_iso", *self.csv_header])
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        try:
            self._close()
        except Exception:
            pass
        if self._fp is not None:
            self._fp.close()
            self._fp = None

    def _loop(self) -> None:
        period = 1.0 / max(self.hz, 0.1)
        while not self._stop.is_set():
            try:
                values = self._sample()
                if values is not None:
                    ts = time.time()
                    self._samples.append((ts, *values))
                    if self._writer is not None:
                        iso = _dt.datetime.fromtimestamp(
                            ts, _dt.timezone.utc
                        ).isoformat()
                        self._writer.writerow([iso, *self._format(values)])
                        self._fp.flush()
            except Exception:
                pass
            self._stop.wait(period)

    def _column(self, idx: int) -> list[float]:
        """Values in column idx across all samples, NaNs dropped (0 is the time)."""
        return [s[idx] for s in self._samples if s[idx] == s[idx]]
