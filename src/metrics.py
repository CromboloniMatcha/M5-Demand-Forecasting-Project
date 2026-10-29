"""Forecast accuracy metrics.

WAPE    = sum |forecast - actual| / sum actual        (volume-weighted, easy to explain)
Bias    = sum (forecast - actual) / sum actual         (positive = over-forecast)
RMSSE   = sqrt(mean squared error / mean squared one-step naive error on the training data)
          This is the scaled error used in the M5 competition. 1.0 = as good as the naive
          one-step forecast; below 1.0 is better.
wRMSSE  = RMSSE averaged with revenue weights. NOTE: the official M5 WRMSSE also averages over
          12 aggregation levels. This project reports the bottom (item-store) level only.
"""
import warnings

import numpy as np
import pandas as pd


def train_scale(Yf, cutoff):
    """Mean squared one-step naive error over each series' active training days."""
    d = np.diff(Yf[:, :cutoff], axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        scale = np.nanmean(d * d, axis=1).astype(np.float64)
    scale[~(scale > 0)] = np.nan   # constant or empty history: RMSSE undefined
    return scale


def per_series_errors(actual, forecast, scale):
    """One row of error totals per series. actual and forecast have shape (n_series, horizon)."""
    a = np.asarray(actual, dtype=np.float64)
    e = np.asarray(forecast, dtype=np.float64) - a
    return pd.DataFrame({
        "abs_err": np.abs(e).sum(axis=1),
        "err": e.sum(axis=1),
        "actual": a.sum(axis=1),
        "rmsse": np.sqrt((e ** 2).mean(axis=1) / scale),
    })


def summarize(df):
    """Metrics for a group of series. df needs abs_err, err, actual, rmsse and weight columns."""
    act = df["actual"].sum()
    ok = df["rmsse"].notna() & (df["weight"] > 0)
    return {
        "n_series": len(df),
        "WAPE": df["abs_err"].sum() / act if act > 0 else np.nan,
        "Bias": df["err"].sum() / act if act > 0 else np.nan,
        "RMSSE": df["rmsse"].mean(),
        "wRMSSE": ((df.loc[ok, "weight"] * df.loc[ok, "rmsse"]).sum() / df.loc[ok, "weight"].sum())
                  if ok.any() else np.nan,
    }


def group_summary(df, by):
    rows = []
    for key, g in df.groupby(by, sort=True):
        rows.append({"group": key, **summarize(g)})
    return pd.DataFrame(rows)
