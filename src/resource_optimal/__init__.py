"""resource_optimal: track, visualize and optimize GPU/RAM usage.

A lightweight resource manager that samples GPU + system-memory usage,
stores the time series, forecasts near-future demand with a small online
model and recommends optimal resource allocation for intensive tasks.
"""
from .backends import (
    AcceleratorBackend,
    AcceleratorReading,
    NpuBackend,
    TpuBackend,
)
from .collector import Sample, ResourceCollector
from .storage import TimeSeriesStore
from .forecaster import UsageForecaster, Forecast
from .advisor import OptimalAdvisor, Recommendation
from .monitor import ResourceMonitor

__all__ = [
    "AcceleratorBackend",
    "AcceleratorReading",
    "NpuBackend",
    "TpuBackend",
    "Sample",
    "ResourceCollector",
    "TimeSeriesStore",
    "UsageForecaster",
    "Forecast",
    "OptimalAdvisor",
    "Recommendation",
    "ResourceMonitor",
]

__version__ = "0.1.0"
