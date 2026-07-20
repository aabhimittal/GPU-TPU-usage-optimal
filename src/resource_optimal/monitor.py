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
    gpu_index:
        GPU to monitor when several are present.
    """

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        gpu_index: int = 0,
        advisor: Optional[OptimalAdvisor] = None,
    ) -> None:
        self.collector = ResourceCollector(gpu_index=gpu_index)
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
        gpu_usage = self.store.series("gpu_used_mb", limit=limit)
        ram_cap = self._last_nonnull("ram_total_mb")
        gpu_cap = self._last_nonnull("gpu_total_mb")
        if not ram_usage or not ram_cap:
            return []
        return self.advisor.recommend_all(
            ram_usage=ram_usage,
            ram_capacity=ram_cap,
            gpu_usage=gpu_usage or None,
            gpu_capacity=gpu_cap,
            horizon=horizon,
        )

    def _last_nonnull(self, field: str) -> Optional[float]:
        vals = self.store.series(field, limit=10)
        return vals[-1] if vals else None

    def dashboard(self, limit: int = 60) -> str:
        """Render the current terminal dashboard."""
        ram_series = self.store.series("ram_used_mb", limit=limit)
        gpu_series = self.store.series("gpu_used_mb", limit=limit)
        ram_pct = self._last_nonnull("ram_percent")
        gpu_pct = self._last_nonnull("gpu_percent")
        recs = self.recommendations()
        return render_dashboard(ram_series, ram_pct, gpu_series or None,
                                gpu_pct, recs)

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
