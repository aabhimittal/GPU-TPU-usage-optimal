"""High-level orchestration tying the pieces together."""
from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

from .advisor import OptimalAdvisor, Recommendation
from .collector import ResourceCollector, Sample
from .storage import TimeSeriesStore
from .visualizer import render_dashboard


class ResourceMonitor:
    """Sample usage, persist it, and produce optimal-usage recommendations.

    Parameters
    ----------
    db_path:
        SQLite path for history (``":memory:"`` for ephemeral runs).
    gpu_index, tpu_index, npu_index:
        Device indices to monitor when several of a kind are present.
    collector:
        Optional pre-built collector (e.g. with injected TPU/NPU back-ends);
        overrides the ``*_index`` arguments when supplied.
    """

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        gpu_index: int = 0,
        tpu_index: int = 0,
        npu_index: int = 0,
        advisor: Optional[OptimalAdvisor] = None,
        collector: Optional[ResourceCollector] = None,
    ) -> None:
        self.collector = collector or ResourceCollector(
            gpu_index=gpu_index, tpu_index=tpu_index, npu_index=npu_index
        )
        self.store = TimeSeriesStore(db_path)
        self.advisor = advisor or OptimalAdvisor()

    def tick(self) -> Sample:
        """Collect one sample and persist it."""
        s = self.collector.sample()
        self.store.add(s)
        return s

    def recommendations(self, horizon: int = 1,
                         limit: int = 240) -> List[Recommendation]:
        """Compute recommendations from recent history."""
        ram_usage = self.store.series("ram_used_mb", limit=limit)
        ram_cap = self._last_nonnull("ram_total_mb")
        # RAM is optional; accelerator recommendations stand on their own.
        return self.advisor.recommend_all(
            ram_usage=ram_usage if ram_cap else [],
            ram_capacity=ram_cap or 0.0,
            gpu_usage=self.store.series("gpu_used_mb", limit=limit) or None,
            gpu_capacity=self._last_nonnull("gpu_total_mb"),
            tpu_usage=self.store.series("tpu_used_mb", limit=limit) or None,
            tpu_capacity=self._last_nonnull("tpu_total_mb"),
            npu_usage=self.store.series("npu_used_mb", limit=limit) or None,
            npu_capacity=self._last_nonnull("npu_total_mb"),
            horizon=horizon,
        )

    def _last_nonnull(self, field: str) -> Optional[float]:
        vals = self.store.series(field, limit=10)
        return vals[-1] if vals else None

    def dashboard(self, limit: int = 60) -> str:
        """Render the current terminal dashboard."""
        return render_dashboard(
            ram_series=self.store.series("ram_used_mb", limit=limit),
            ram_pct=self._last_nonnull("ram_percent"),
            gpu_series=self.store.series("gpu_used_mb", limit=limit) or None,
            gpu_pct=self._last_nonnull("gpu_percent"),
            tpu_series=self.store.series("tpu_used_mb", limit=limit) or None,
            tpu_pct=self._last_nonnull("tpu_percent"),
            npu_series=self.store.series("npu_used_mb", limit=limit) or None,
            npu_pct=self._last_nonnull("npu_percent"),
            recs=self.recommendations(),
        )

    def run(self, samples: int, interval: float = 1.0,
            live: bool = False) -> List[Recommendation]:
        """Collect ``samples`` snapshots then return recommendations.

        When ``live`` is set, redraw the dashboard after every sample.
        """
        for i in range(samples):
            self.tick()
            if live:
                print("\n" + self.dashboard())
            if i < samples - 1:
                time.sleep(interval)
        return self.recommendations()

    def close(self) -> None:
        self.collector.close()
        self.store.close()

    def __enter__(self) -> "ResourceMonitor":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
