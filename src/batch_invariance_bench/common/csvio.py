"""CSV and JSON output helpers, shared by both runners."""

from __future__ import annotations

import csv
import datetime as _dt
import json
import os
import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def now() -> str:
    """Current UTC time as an ISO-8601 string."""
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def slug(s: str) -> str:
    """Make a string safe to use in a filename."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_") or "x"


def default_output_dir(out_dir: str | os.PathLike = "results") -> Path:
    """Return the output directory, creating it if needed."""
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def append_csv_rows(
    path: str | os.PathLike,
    rows: Iterable[Mapping[str, object]],
    fieldnames: Sequence[str],
) -> None:
    """Append rows to a CSV, writing the header first if the file is new.

    Raises if the file exists but its header does not match fieldnames.
    """
    p = Path(path)
    fieldnames = list(fieldnames)
    is_new = not p.exists() or p.stat().st_size == 0
    if not is_new:
        with p.open("r", newline="") as f:
            existing = next(csv.reader(f), [])
        if existing != fieldnames:
            raise ValueError(
                f"CSV schema mismatch for {p}: existing header {existing} "
                f"does not match expected {fieldnames}"
            )
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: str | os.PathLike, obj: object) -> None:
    """Write obj to path as indented JSON, creating parent dirs."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2))
