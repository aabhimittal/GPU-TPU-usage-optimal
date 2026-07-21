"""Turn forecasts into concrete optimal-usage recommendations.

The advisor consumes the forecaster's *demand band* and recommends a resource
budget for an upcoming intensive task: reserve the predicted upper-band demand
plus a safety headroom, but never exceed the device's physical capacity. It
also flags when a task is likely to be starved (demand approaching capacity) or
when resources are over-provisioned (large idle headroom).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from .forecaster import Forecast, UsageForecaster


@dataclass
class Recommendation:
    """A sizing recommendation for one resource (RAM or GPU memory)."""

    resource: str
    capacity_mb: float
    predicted_mb: float
    recommended_reserve_mb: float
    headroom_mb: float
    utilisation_pct: float
    status: str  # "ok" | "tight" | "over_capacity" | "over_provisioned"
    message: str


class OptimalAdvisor:
    """Recommend optimal resource budgets from usage history.

    Parameters
    ----------
    headroom:
        Fractional safety buffer added on top of the forecast band's upper
        bound (0.15 = 15%).
    tight_pct / idle_pct:
        Utilisation thresholds (percent of capacity) above which a task is
        flagged "tight" and below which it is flagged "over_provisioned".
    """

    def __init__(
        self,
        headroom: float = 0.15,
        tight_pct: float = 90.0,
        idle_pct: float = 40.0,
        forecaster: Optional[UsageForecaster] = None,
    ) -> None:
        if headroom < 0:
            raise ValueError("headroom must be non-negative")
        self.headroom = headroom
        self.tight_pct = tight_pct
        self.idle_pct = idle_pct
        self.forecaster = forecaster or UsageForecaster()

    def recommend(
        self,
        resource: str,
        usage_mb: Sequence[float],
        capacity_mb: float,
        horizon: int = 1,
    ) -> Recommendation:
        """Recommend a reserve for ``resource`` given recent ``usage_mb``."""
        if capacity_mb <= 0:
            raise ValueError("capacity_mb must be positive")
        fc: Forecast = self.forecaster.forecast(usage_mb, horizon=horizon)

        # size against the upper band, add headroom, clamp to capacity
        target = max(fc.predicted, fc.upper) * (1 + self.headroom)
        reserve = min(max(target, 0.0), capacity_mb)
        util = 100.0 * reserve / capacity_mb
        headroom_mb = capacity_mb - reserve

        raw_target = max(fc.predicted, fc.upper) * (1 + self.headroom)
        if raw_target > capacity_mb:
            status = "over_capacity"
            msg = (
                f"Projected demand ~{raw_target:.0f} MB exceeds {capacity_mb:.0f} MB "
                f"capacity; reduce batch size or shard the task."
            )
        elif util >= self.tight_pct:
            status = "tight"
            msg = (
                f"Reserve {reserve:.0f} MB ({util:.0f}% of capacity) — little "
                f"headroom; watch for spikes."
            )
        elif util <= self.idle_pct:
            status = "over_provisioned"
            msg = (
                f"Only ~{util:.0f}% of capacity needed; {headroom_mb:.0f} MB idle — "
                f"safe to co-locate another task or use a smaller device."
            )
        else:
            status = "ok"
            msg = f"Reserve {reserve:.0f} MB ({util:.0f}% of capacity); {headroom_mb:.0f} MB headroom."

        return Recommendation(
            resource=resource,
            capacity_mb=capacity_mb,
            predicted_mb=fc.predicted,
            recommended_reserve_mb=reserve,
            headroom_mb=headroom_mb,
            utilisation_pct=util,
            status=status,
            message=msg,
        )

    def recommend_all(
        self,
        ram_usage: Sequence[float],
        ram_capacity: float,
        gpu_usage: Optional[Sequence[float]] = None,
        gpu_capacity: Optional[float] = None,
        tpu_usage: Optional[Sequence[float]] = None,
        tpu_capacity: Optional[float] = None,
        npu_usage: Optional[Sequence[float]] = None,
        npu_capacity: Optional[float] = None,
        horizon: int = 1,
    ) -> List[Recommendation]:
        """Recommend for RAM and any accelerator with usage + capacity data."""
        recs: List[Recommendation] = []
        if ram_usage:
            recs.append(self.recommend("RAM", ram_usage, ram_capacity, horizon))
        for label, usage, cap in (
            ("GPU", gpu_usage, gpu_capacity),
            ("TPU", tpu_usage, tpu_capacity),
            ("NPU", npu_usage, npu_capacity),
        ):
            if usage and cap:
                recs.append(self.recommend(label, usage, cap, horizon))
        return recs
