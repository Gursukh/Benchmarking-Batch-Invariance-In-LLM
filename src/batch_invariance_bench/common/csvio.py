"""CSV helpers shared by both runners."""

from __future__ import annotations

import csv
import datetime as _dt
import os
import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_") or "x"


def default_output_dir(out_dir: str | os.PathLike = "results") -> Path:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def append_csv_rows(
    path: str | os.PathLike,
    rows: Iterable[Mapping[str, object]],
    fieldnames: Sequence[str],
) -> None:
    """Append rows, writing the header first if the file is new."""
    p = Path(path)
    fieldnames = list(fieldnames)
    is_new = not p.exists() or p.stat().st_size == 0
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
