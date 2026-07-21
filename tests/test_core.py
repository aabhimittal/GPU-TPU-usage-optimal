"""Tests for the dependency-free core (forecaster, advisor, storage, viz)."""
import math

import pytest

from resource_optimal.advisor import OptimalAdvisor
from resource_optimal.collector import Sample
from resource_optimal.forecaster import UsageForecaster
from resource_optimal.storage import TimeSeriesStore
from resource_optimal.visualizer import render_dashboard, sparkline


# ---- forecaster ---------------------------------------------------------
def test_forecast_captures_upward_trend():
    fc = UsageForecaster().forecast(list(range(20)), horizon=1)
    assert fc.trend > 0.5
    assert fc.predicted > 18  # should project beyond the last observed value
    assert fc.lower <= fc.predicted <= fc.upper


def test_forecast_flat_series_has_no_trend():
    fc = UsageForecaster().forecast([5.0] * 15)
    assert math.isclose(fc.trend, 0.0, abs_tol=1e-6)
    assert math.isclose(fc.predicted, 5.0, abs_tol=1e-6)
    assert math.isclose(fc.volatility, 0.0, abs_tol=1e-6)


def test_forecast_band_widens_with_horizon():
    y = [10, 12, 11, 13, 12, 14, 13, 15, 14, 16]
    f = UsageForecaster()
    near = f.forecast(y, horizon=1)
    far = f.forecast(y, horizon=8)
    assert (far.upper - far.lower) >= (near.upper - near.lower)


def test_forecast_empty_raises():
    with pytest.raises(ValueError):
        UsageForecaster().forecast([])


def test_forecast_single_point_is_flat():
    fc = UsageForecaster().forecast([7.0], horizon=3)
    assert fc.predicted == 7.0 and fc.lower == 7.0 and fc.upper == 7.0


# ---- advisor ------------------------------------------------------------
def test_advisor_over_capacity_when_demand_exceeds():
    rec = OptimalAdvisor().recommend("GPU", list(range(80, 100)), capacity_mb=90)
    assert rec.status == "over_capacity"
    assert rec.recommended_reserve_mb <= rec.capacity_mb


def test_advisor_over_provisioned_when_idle():
    rec = OptimalAdvisor().recommend("RAM", [100.0] * 12, capacity_mb=10000)
    assert rec.status == "over_provisioned"
    assert rec.headroom_mb > 0


def test_advisor_ok_band():
    rec = OptimalAdvisor(headroom=0.1).recommend("RAM", [600.0] * 12, capacity_mb=1000)
    assert rec.status in {"ok", "tight"}
    assert 0 <= rec.utilisation_pct <= 100


def test_advisor_rejects_bad_capacity():
    with pytest.raises(ValueError):
        OptimalAdvisor().recommend("RAM", [1, 2, 3], capacity_mb=0)


# ---- storage ------------------------------------------------------------
def test_store_roundtrip_and_series():
    with TimeSeriesStore(":memory:") as store:
        store.add_many([Sample(ts=float(i), ram_used_mb=float(i)) for i in range(5)])
        assert len(store) == 5
        assert store.series("ram_used_mb") == [0, 1, 2, 3, 4]


def test_store_limit_returns_newest_chronological():
    with TimeSeriesStore(":memory:") as store:
        store.add_many([Sample(ts=float(i), ram_used_mb=float(i)) for i in range(10)])
        assert store.series("ram_used_mb", limit=3) == [7, 8, 9]


def test_store_series_skips_none():
    with TimeSeriesStore(":memory:") as store:
        store.add(Sample(ts=1.0, gpu_used_mb=None))
        store.add(Sample(ts=2.0, gpu_used_mb=42.0))
        assert store.series("gpu_used_mb") == [42.0]


def test_store_unknown_field_raises():
    with TimeSeriesStore(":memory:") as store:
        with pytest.raises(ValueError):
            store.series("nope")


# ---- visualiser ---------------------------------------------------------
def test_sparkline_length_and_extremes():
    line = sparkline([0, 1, 2, 3, 4, 5, 6, 7, 8])
    assert len(line) == 9
    assert line[0] == " " or line[0] == "▁"
    assert line[-1] == "█"


def test_sparkline_empty():
    assert sparkline([]) == ""


def test_dashboard_contains_sections():
    out = render_dashboard([1, 2, 3], ram_pct=50.0, gpu_series=[4, 5, 6], gpu_pct=25.0)
    assert "RAM" in out and "GPU" in out
