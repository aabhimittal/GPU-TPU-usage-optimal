"""Efficient usage visualisation.

Two rendering paths:

* :func:`sparkline` / :func:`render_dashboard` — zero-dependency Unicode output
  suitable for live terminal display; cheap enough to redraw every sample.
* :func:`save_report` — a matplotlib PNG report (optional dependency) overlaying
  history, forecast band and recommended reserve for offline analysis.
"""
from __future__ import annotations

from typing import Optional, Sequence

from .advisor import Recommendation
from .forecaster import Forecast

_BLOCKS = " ▁▂▃▄▅▆▇█"


def sparkline(values: Sequence[float], lo: Optional[float] = None,
              hi: Optional[float] = None) -> str:
    """Render ``values`` as a Unicode sparkline scaled to ``[lo, hi]``."""
    vals = [float(v) for v in values]
    if not vals:
        return ""
    lo = min(vals) if lo is None else lo
    hi = max(vals) if hi is None else hi
    span = hi - lo
    if span <= 0:
        return _BLOCKS[1] * len(vals)
    out = []
    n = len(_BLOCKS) - 1
    for v in vals:
        frac = (v - lo) / span
        idx = max(0, min(n, round(frac * n)))
        out.append(_BLOCKS[idx])
    return "".join(out)


def _bar(pct: float, width: int = 24) -> str:
    pct = max(0.0, min(100.0, pct))
    filled = round(width * pct / 100.0)
    return "[" + "█" * filled + "·" * (width - filled) + f"] {pct:5.1f}%"


def render_dashboard(
    ram_series: Sequence[float],
    ram_pct: Optional[float] = None,
    gpu_series: Optional[Sequence[float]] = None,
    gpu_pct: Optional[float] = None,
    recs: Optional[Sequence[Recommendation]] = None,
) -> str:
    """Return a compact multi-line dashboard string for the terminal."""
    lines = ["── resource-optimal ──────────────────"]
    if ram_series:
        lines.append(f"RAM  {sparkline(ram_series)}")
    if ram_pct is not None:
        lines.append(f"     {_bar(ram_pct)}")
    if gpu_series:
        lines.append(f"GPU  {sparkline(gpu_series)}")
    if gpu_pct is not None:
        lines.append(f"     {_bar(gpu_pct)}")
    for r in recs or []:
        lines.append(f"→ {r.resource}: {r.message}")
    return "\n".join(lines)


def save_report(
    path: str,
    history: Sequence[float],
    forecast: Forecast,
    recommendation: Optional[Recommendation] = None,
    title: str = "Usage forecast",
) -> str:
    """Write a PNG report; requires matplotlib. Returns the output path."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dep
        raise RuntimeError("save_report requires matplotlib") from exc

    hist = [float(v) for v in history]
    x = list(range(len(hist)))
    fx = len(hist) + forecast.horizon - 1

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(x, hist, label="history", color="#2b8cbe")
    ax.scatter([fx], [forecast.predicted], color="#e34a33", zorder=5,
               label="forecast")
    ax.fill_between([len(hist) - 1, fx],
                    [hist[-1], forecast.lower],
                    [hist[-1], forecast.upper],
                    color="#fdbb84", alpha=0.4, label="demand band")
    if recommendation is not None:
        ax.axhline(recommendation.recommended_reserve_mb, color="#31a354",
                   ls="--", label="recommended reserve")
        ax.axhline(recommendation.capacity_mb, color="#636363", ls=":",
                   label="capacity")
    ax.set_title(title)
    ax.set_xlabel("sample")
    ax.set_ylabel("usage")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
