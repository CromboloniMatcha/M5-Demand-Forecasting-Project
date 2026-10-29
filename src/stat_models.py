"""Classical statistical models via statsmodels: ETS (Holt-Winters) and SARIMA.

These are fitted ONE SERIES AT A TIME, so they are much slower than the baselines in
baselines.py. They are therefore used on (a) a stratified sample of item-store series and
(b) aggregated series (store x category, total), where demand is smooth enough for them.

Quick check that statsmodels works on your machine:
    python -m src.stat_models
"""
import time
import warnings

import numpy as np

from src import config


def statsmodels_available():
    try:
        import statsmodels  # noqa: F401
        return True
    except ImportError:
        return False


def ets_forecast(y, horizon):
    """Additive Holt-Winters with weekly seasonality, no trend. Forecasts are clipped at 0."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    y = np.asarray(y, dtype=float)
    if len(y) == 0:
        return np.zeros(horizon)
    if len(y) < 28 or np.all(y == y[0]):
        return np.full(horizon, y[-5:].mean() if len(y) else 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ExponentialSmoothing(y, trend=None, seasonal="add", seasonal_periods=7,
                                     initialization_method="estimated")
        fc = np.asarray(model.fit().forecast(horizon), dtype=float)
    if not np.all(np.isfinite(fc)):
        raise ValueError("non-finite ETS forecast")
    return np.clip(fc, 0, None)


def sarima_forecast(y, horizon):
    """SARIMA(1,0,1)(1,1,1,7) with fixed orders (not auto-selected). Clipped at 0."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    y = np.asarray(y, dtype=float)
    if len(y) < 56 or np.all(y == y[0]):
        return np.full(horizon, y[-5:].mean() if len(y) else 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(y, order=(1, 0, 1), seasonal_order=(1, 1, 1, 7),
                        enforce_stationarity=False, enforce_invertibility=False)
        fc = np.asarray(model.fit(disp=False, maxiter=50).forecast(horizon), dtype=float)
    if not np.all(np.isfinite(fc)):
        raise ValueError("non-finite SARIMA forecast")
    return np.clip(fc, 0, None)


def forecast_many(Yf, rows, cutoff, horizon, fn, max_train=None, log=None):
    """Fit `fn` on each selected row. Returns (forecasts[len(rows) x horizon], n_failed).

    Rows that fail are returned as NaN; the caller decides on a fallback.
    The training window starts at each series' launch and is capped at `max_train` days.
    """
    max_train = max_train or config.STATS_MAX_TRAIN_DAYS
    out = np.full((len(rows), horizon), np.nan)
    failed = 0
    t0 = time.time()
    for k, i in enumerate(rows):
        y = Yf[i, max(0, cutoff - max_train):cutoff]
        y = y[~np.isnan(y)]
        try:
            out[k] = fn(y, horizon)
        except Exception:
            failed += 1
        if log and (k + 1) % 50 == 0:
            log(f"    {k + 1}/{len(rows)} series fitted ({time.time() - t0:,.0f}s)")
    return out, failed


def _selftest():
    if not statsmodels_available():
        print("statsmodels is NOT installed. Install it with: pip install statsmodels")
        return
    import statsmodels
    print("statsmodels", statsmodels.__version__)
    rng = np.random.default_rng(0)
    t = np.arange(400)
    y = rng.poisson(5 + 2 * np.sin(2 * np.pi * t / 7)).astype(float)
    for name, fn in [("ETS", ets_forecast), ("SARIMA", sarima_forecast)]:
        t0 = time.time()
        fc = fn(y, 14)
        print(f"{name}: ok in {time.time() - t0:.1f}s, first 7 forecasts = {np.round(fc[:7], 2).tolist()}")


if __name__ == "__main__":
    _selftest()
