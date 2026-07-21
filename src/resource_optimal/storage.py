"""Persistent time-series storage for usage samples (SQLite-backed)."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, List, Optional

from .collector import Sample

_COLUMNS = (
    "ts",
    "ram_used_mb",
    "ram_total_mb",
    "ram_percent",
    "cpu_percent",
    "gpu_used_mb",
    "gpu_total_mb",
    "gpu_percent",
    "gpu_util",
    "tpu_used_mb",
    "tpu_total_mb",
    "tpu_percent",
    "tpu_util",
    "npu_used_mb",
    "npu_total_mb",
    "npu_percent",
    "npu_util",
)


class TimeSeriesStore:
    """Append-only store of :class:`Sample` rows backed by SQLite.

    Use ``":memory:"`` (the default) for an ephemeral store, or a file path to
    persist usage history across runs so the forecaster can learn from trends.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        cols = ", ".join(f"{c} REAL" for c in _COLUMNS)
        self._conn.execute(f"CREATE TABLE IF NOT EXISTS samples ({cols})")
        # migrate older databases that predate the TPU/NPU columns
        existing = {
            row[1] for row in self._conn.execute("PRAGMA table_info(samples)")
        }
        for c in _COLUMNS:
            if c not in existing:
                self._conn.execute(f"ALTER TABLE samples ADD COLUMN {c} REAL")
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_samples_ts ON samples(ts)"
        )
        self._conn.commit()

    def add(self, sample: Sample) -> None:
        self.add_many([sample])

    def add_many(self, samples: Iterable[Sample]) -> None:
        rows = [tuple(getattr(s, c) for c in _COLUMNS) for s in samples]
        if not rows:
            return
        placeholders = ", ".join("?" for _ in _COLUMNS)
        self._conn.executemany(
            f"INSERT INTO samples ({', '.join(_COLUMNS)}) VALUES ({placeholders})",
            rows,
        )
        self._conn.commit()

    def all(self, limit: Optional[int] = None) -> List[Sample]:
        """Return stored samples in chronological order (oldest first)."""
        q = "SELECT * FROM samples ORDER BY ts ASC"
        if limit is not None:
            # newest ``limit`` rows, still returned oldest-first
            q = (
                "SELECT * FROM (SELECT * FROM samples ORDER BY ts DESC "
                f"LIMIT {int(limit)}) ORDER BY ts ASC"
            )
        cur = self._conn.execute(q)
        return [Sample(**{k: row[k] for k in _COLUMNS}) for row in cur.fetchall()]

    def series(self, field: str, limit: Optional[int] = None) -> List[float]:
        """Return one metric as a list, skipping ``None`` gaps."""
        if field not in _COLUMNS:
            raise ValueError(f"unknown field {field!r}; expected one of {_COLUMNS}")
        return [
            getattr(s, field)
            for s in self.all(limit=limit)
            if getattr(s, field) is not None
        ]

    def __len__(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0])

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "TimeSeriesStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
