"""Near-future usage forecasting.

The forecaster is deliberately dependency-free: it implements Holt's linear
(double exponential smoothing) model in pure Python so it runs anywhere and
learns online from a usage series. When ``scikit-learn`` is installed a ridge
auto-regressive model is used instead for a modest accuracy gain; the pure
Python model is the fallback and the reference behaviour.

The "novel" part is not the smoother itself but how its output is consumed
(see :mod:`resource_optimal.advisor`): instead of predicting a single point we
predict a *demand band* — expected level plus an empirical headroom derived
from recent volatility — and size resource limits against that band.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean, pstdev
from typing import List, Optional, Sequence

try:
    from sklearn.linear_model import Ridge  # type: ignore

    _SKLEARN_OK = True
except Exception:  # pragma: no cover - optional
    Ridge = None  # type: ignore
    _SKLEARN_OK = False


@dataclass
class Forecast:
    """Result of a forecast.

    Attributes
    ----------
    predicted:
        Point forecast for the next step (same unit as the input series).
    lower / upper:
        Demand band around ``predicted`` at the configured confidence.
    trend:
        Per-step slope; positive means usage is rising.
    volatility:
        Standard deviation of recent residuals, the band's half-width source.
    horizon:
        Number of steps ahead the point forecast projects.
    """

    predicted: float
    lower: float
    upper: float
    trend: float
    volatility: float
    horizon: int = 1


class UsageForecaster:
    """Online forecaster for a single usage metric.

    Parameters
    ----------
    alpha, beta:
        Level and trend smoothing factors for Holt's method (0-1).
    z:
        Band half-width in residual standard deviations (1.28 ~ 90%).
    """

    def __init__(self, alpha: float = 0.4, beta: float = 0.2, z: float = 1.28) -> None:
        if not 0.0 < alpha <= 1.0 or not 0.0 <= beta <= 1.0:
            raise ValueError("alpha must be in (0,1] and beta in [0,1]")
        self.alpha = alpha
        self.beta = beta
        self.z = z

    # -- Holt's linear trend (pure python) --------------------------------
    def _holt(self, y: Sequence[float]) -> tuple[float, float, List[float]]:
        level = float(y[0])
        trend = float(y[1] - y[0]) if len(y) > 1 else 0.0
        fitted: List[float] = [level]
        for t in range(1, len(y)):
            prev_level = level
            level = self.alpha * y[t] + (1 - self.alpha) * (prev_level + trend)
            trend = self.beta * (level - prev_level) + (1 - self.beta) * trend
            fitted.append(prev_level + trend)  # one-step-ahead fit
        return level, trend, fitted

    def _residual_std(self, y: Sequence[float], fitted: Sequence[float]) -> float:
        res = [a - b for a, b in zip(y[1:], fitted[1:])]
        return pstdev(res) if len(res) > 1 else 0.0

    # -- optional sklearn AR model ----------------------------------------
    def _ridge_point(self, y: Sequence[float], horizon: int) -> Optional[float]:
        if not _SKLEARN_OK or len(y) < 8:
            return None
        lag = min(5, len(y) // 2)
        X = [y[i - lag : i] for i in range(lag, len(y))]
        target = list(y[lag:])
        try:
            model = Ridge(alpha=1.0).fit(X, target)
            window = list(y[-lag:])
            pred = window[-1]
            for _ in range(horizon):
                pred = float(model.predict([window[-lag:]])[0])
                window.append(pred)
            return pred
        except Exception:  # pragma: no cover
            return None

    # -- public API -------------------------------------------------------
    def forecast(self, series: Sequence[float], horizon: int = 1) -> Forecast:
        """Forecast ``horizon`` steps ahead from ``series``.

        Raises ``ValueError`` on an empty series. A single-point series yields a
        flat forecast (no trend, no band).
        """
        y = [float(v) for v in series]
        if not y:
            raise ValueError("cannot forecast from an empty series")
        if len(y) == 1:
            return Forecast(y[0], y[0], y[0], 0.0, 0.0, horizon)

        level, trend, fitted = self._holt(y)
        vol = self._residual_std(y, fitted)
        point = level + trend * horizon

        ridge = self._ridge_point(y, horizon)
        if ridge is not None:
            point = 0.5 * point + 0.5 * ridge  # blend smoother with AR model

        # band widens with the horizon (uncertainty compounds)
        half = self.z * vol * math.sqrt(max(1, horizon))
        return Forecast(
            predicted=point,
            lower=point - half,
            upper=point + half,
            trend=trend,
            volatility=vol,
            horizon=horizon,
        )

    def moving_average(self, series: Sequence[float], window: int = 5) -> float:
        """Simple trailing mean, handy as a baseline comparison."""
        if not series:
            raise ValueError("empty series")
        w = list(series)[-window:]
        return fmean(w)
