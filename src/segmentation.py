"""Segment every series by business value (ABC), demand variability (XYZ) and
statistical demand pattern (smooth / erratic / intermittent / lumpy).

All three use ONLY data before `ref_end` (the first backtest cutoff), so the segments
never see any test window. Series with too little history are labelled NEW.

ABC  : by revenue in the window. A = top 80% of revenue, B = next 15%, C = last 5%.
XYZ  : by coefficient of variation (CV) of weekly demand. X < 0.5, Y < 1.0, Z >= 1.0.
SBC  : Syntetos-Boylan-Croston classification from ADI (average days between demands)
       and CV^2 of the non-zero demand sizes.
"""
import warnings

import numpy as np
import pandas as pd

from src import config


def segment_series(data, Yf, ref_end):
    n = data.n_series
    weeks = min(config.SEG_WINDOW_DAYS // 7, ref_end // 7)
    start = ref_end - 7 * weeks
    block = Yf[:, start:ref_end]

    active = ~np.isnan(block)
    n_active = active.sum(axis=1)
    units = np.nansum(block, axis=1)
    revenue = units * data.series["avg_price"].to_numpy()   # revenue proxy: units x average price

    # --- weekly CV (only fully active weeks) ---
    wk = block.reshape(n, weeks, 7)
    full_week = ~np.isnan(wk).any(axis=2)
    wk_sum = np.where(full_week, np.nan_to_num(wk).sum(axis=2), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        wk_mean = np.nanmean(wk_sum, axis=1)
        wk_std = np.nanstd(wk_sum, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = np.where(wk_mean > 0, wk_std / wk_mean, np.inf)

    # --- demand pattern (ADI and CV^2 of non-zero sizes, daily data) ---
    with np.errstate(divide="ignore", invalid="ignore"):
        nz = block > 0
        n_nz = nz.sum(axis=1)
        adi = np.where(n_nz > 0, n_active / n_nz, np.inf)
        mean_nz = np.where(n_nz > 0, np.where(nz, block, 0).sum(axis=1) / n_nz, np.nan)
        var_nz = np.where(n_nz > 0,
                          np.where(nz, (block - mean_nz[:, None]) ** 2, 0).sum(axis=1) / n_nz, np.nan)
        cv2 = var_nz / mean_nz ** 2
    pattern = np.select(
        [n_nz == 0,
         (adi < config.SBC_ADI_CUT) & (cv2 < config.SBC_CV2_CUT),
         (adi < config.SBC_ADI_CUT) & (cv2 >= config.SBC_CV2_CUT),
         (adi >= config.SBC_ADI_CUT) & (cv2 < config.SBC_CV2_CUT)],
        ["no_demand", "smooth", "erratic", "intermittent"], default="lumpy")

    seg = data.series[["series_id", "item_id", "store_id", "dept_id", "cat_id"]].copy()
    seg["n_active_window"] = n_active
    seg["revenue_window"] = revenue
    seg["cv_weekly"] = cv
    seg["adi"] = adi
    seg["cv2_size"] = cv2
    seg["demand_pattern"] = pattern

    eligible = n_active >= config.MIN_SEG_DAYS

    # --- ABC among eligible series ---
    abc = np.full(n, "NEW", dtype=object)
    order = np.argsort(-revenue[eligible], kind="stable")
    idx = np.flatnonzero(eligible)[order]
    total = revenue[idx].sum()
    if total > 0:
        cum = np.cumsum(revenue[idx]) / total
        abc[idx] = np.where(cum <= config.ABC_CUTS[0], "A", np.where(cum <= config.ABC_CUTS[1], "B", "C"))
    else:
        abc[idx] = "C"

    # --- XYZ among eligible series ---
    xyz = np.full(n, "NEW", dtype=object)
    xyz[eligible] = np.where(cv[eligible] < config.XYZ_CUTS[0], "X",
                             np.where(cv[eligible] < config.XYZ_CUTS[1], "Y", "Z"))

    seg["abc"] = abc
    seg["xyz"] = xyz
    seg["abc_xyz"] = np.where(eligible, np.char.add(abc.astype(str), xyz.astype(str)), "NEW")
    return seg


def segment_summary(seg):
    """Counts and revenue share per ABC-XYZ cell, and counts per demand pattern."""
    base = seg[seg["abc_xyz"] != "NEW"]
    total_rev = base["revenue_window"].sum()
    cells = (base.groupby(["abc", "xyz"])
             .agg(n_series=("series_id", "count"), revenue=("revenue_window", "sum")).reset_index())
    cells["pct_series"] = 100 * cells["n_series"] / cells["n_series"].sum()
    cells["pct_revenue"] = 100 * cells["revenue"] / total_rev if total_rev > 0 else np.nan
    patterns = seg.groupby("demand_pattern").agg(n_series=("series_id", "count")).reset_index()
    patterns["pct_series"] = 100 * patterns["n_series"] / patterns["n_series"].sum()
    return cells, patterns
