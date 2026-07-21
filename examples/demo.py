"""Offline demo: synthesise a usage trend, forecast it, and recommend a budget.

Runs without a GPU or psutil — it feeds a synthetic series through the same
forecaster/advisor/visualiser used in production.

    python examples/demo.py
"""
import math

from resource_optimal.advisor import OptimalAdvisor
from resource_optimal.forecaster import UsageForecaster
from resource_optimal.visualizer import sparkline


def synthetic_usage(n: int = 60, capacity: float = 16000.0):
    """A rising sawtooth with noise, emulating a memory-hungry training loop."""
    out = []
    for t in range(n):
        base = 4000 + 90 * t                     # steady upward trend
        wobble = 700 * math.sin(t / 4.0)         # periodic allocation churn
        out.append(min(capacity, max(0.0, base + wobble)))
    return out


def main() -> None:
    capacity = 16000.0
    usage = synthetic_usage(capacity=capacity)

    print("Recent GPU memory (MB):")
    print("  " + sparkline(usage))

    fc = UsageForecaster().forecast(usage, horizon=8)
    print(f"\nForecast (+8 steps): {fc.predicted:.0f} MB "
          f"[{fc.lower:.0f}, {fc.upper:.0f}]  trend={fc.trend:+.1f} MB/step")

    rec = OptimalAdvisor(headroom=0.15).recommend("GPU", usage, capacity, horizon=8)
    print(f"\nRecommendation [{rec.status}]: {rec.message}")


if __name__ == "__main__":
    main()
