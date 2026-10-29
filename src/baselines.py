"""Simple, fast baseline forecasters that work on every series at once.

Each function takes Yf (float matrix, shape n_series x n_days, NaN before launch),
a cutoff (the first day to forecast, 0-based) and a horizon, and returns an
(n_series x horizon) array. Only data before the cutoff is used.
"""
import warnings

import numpy as np

from src import config


def _tile(v, horizon):
    return np.repeat(v[:, None], horizon, axis=1)


def naive(Yf, cutoff, horizon):
    """Repeat the last observed day."""
    return _tile(Yf[:, cutoff - 1], horizon)


def seasonal_naive(Yf, cutoff, horizon, season=7):
    """Same weekday last week."""
    idx = cutoff - season + (np.arange(horizon) % season)
    return Yf[:, idx]


def seasonal_avg(Yf, cutoff, horizon, weeks=4, season=7):
    """Average of the same weekday over the last `weeks` weeks."""
    base = cutoff - season * weeks
    block = Yf[:, base:cutoff].reshape(Yf.shape[0], weeks, season)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = np.nanmean(block, axis=1)
    return m[:, np.arange(horizon) % season]


def moving_average(Yf, cutoff, horizon, window=28):
    """Flat forecast = mean of the last `window` active days."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = np.nanmean(Yf[:, cutoff - window:cutoff], axis=1)
    return _tile(m, horizon)


def croston_sba(Yf, cutoff, horizon, alpha=None):
    """Croston's method with the Syntetos-Boylan bias correction, for intermittent demand.

    Separately smooths the size of non-zero demands (z) and the interval between them (p).
    Forecast (flat) = (1 - alpha/2) * z / p. Vectorised over series, loops over days.
    """
    alpha = config.CROSTON_ALPHA if alpha is None else alpha
    n = Yf.shape[0]
    z = np.full(n, np.nan)
    p = np.full(n, np.nan)
    q = np.zeros(n)
    with np.errstate(invalid="ignore"):
        for t in range(cutoff):
            y = Yf[:, t]
            active = ~np.isnan(y)
            q = np.where(active, q + 1, q)
            demand = active & (y > 0)
            first = demand & np.isnan(z)
            later = demand & ~np.isnan(z)
            z = np.where(first, y, z)
            p = np.where(first, q, p)
            z = np.where(later, alpha * y + (1 - alpha) * z, z)
            p = np.where(later, alpha * q + (1 - alpha) * p, p)
            q = np.where(demand, 0.0, q)
        f = (1 - alpha / 2) * z / p
    f = np.where(np.isnan(f), 0.0, f)   # never had demand -> forecast 0
    return _tile(f, horizon)


BASELINES = {
    "naive": naive,
    "seasonal_naive": seasonal_naive,
    "seasonal_avg_4w": seasonal_avg,
    "moving_avg_28d": moving_average,
    "croston_sba": croston_sba,
}
