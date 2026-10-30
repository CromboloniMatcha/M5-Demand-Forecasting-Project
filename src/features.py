"""Feature engineering for the LightGBM demand model (direct multi-horizon forecasting).

Setup (same as the backtest in notebook 02):
    A "forecast origin" o means: days 0..o-1 are known, we forecast days o..o+27.
    One row = one series x one forecast day. Every feature is known at the origin,
    except calendar and price of the TARGET day, which are known in advance
    (the calendar is fixed and Walmart's weekly prices are published in the data).
    The horizon h (1..28) is a feature, so ONE model serves all 28 days with no recursion.

No feature looks past the origin, so there is no leakage.
"""
import sqlite3
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

H = config.HORIZON
LAGS = (1, 2, 3, 7, 14, 21, 28)
WINDOWS = (7, 14, 28, 56, 112)

FEATURES = (
    ["h", "wday", "month", "dom", "event_type", "event_name", "snap",
     "price_t", "price_ratio_28", "price_ratio_hist", "price_changed"]
    + [f"lag{k}" for k in LAGS]
    + [f"rm{w}" for w in WINDOWS]
    + ["rstd28", "zero_rate28", "days_since_sale", "rm7_over_rm28", "sw4", "sw8",
       "age_days", "store", "dept", "cat"]
)
CATEGORICAL = ["store", "dept", "cat", "event_type", "event_name"]


@dataclass
class Exog:
    """Information that is known in advance: calendar, prices, static item attributes."""
    P: np.ndarray            # (n_series, n_days) float32 daily price, NaN when not on sale
    wday: np.ndarray         # (n_days,)
    month: np.ndarray
    dom: np.ndarray
    event_type: np.ndarray   # 0 = no event
    event_name: np.ndarray   # 0 = no event
    snap: np.ndarray         # (3, n_days): CA, TX, WI
    state_idx: np.ndarray    # (n_series,) 0/1/2
    store: np.ndarray        # (n_series,) integer codes
    dept: np.ndarray
    cat: np.ndarray


def load_exog(db_path, data):
    db_path = Path(db_path or config.DB_PATH)
    con = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)
    n_days, n_series = data.n_days, data.n_series
    cal = pd.read_sql_query(
        "SELECT d_num, date, wm_yr_wk, wday, month, event_name_1, event_type_1, snap_ca, snap_tx, snap_wi "
        "FROM dim_calendar WHERE d_num <= ? ORDER BY d_num", con, params=(n_days,))

    # weekly prices -> daily price matrix
    weeks = cal["wm_yr_wk"].to_numpy()
    uniq, week_of_day = np.unique(weeks, return_inverse=True)
    Pw = np.full((n_series, len(uniq)), np.nan, dtype=np.float32)
    cur = con.execute("SELECT series_id, wm_yr_wk, sell_price FROM fact_prices")
    while True:
        rows = cur.fetchmany(1_000_000)
        if not rows:
            break
        a = np.asarray(rows, dtype=np.float64)
        wi = np.searchsorted(uniq, a[:, 1].astype(np.int64))
        wi_c = np.minimum(wi, len(uniq) - 1)
        ok = (wi < len(uniq)) & (uniq[wi_c] == a[:, 1])
        Pw[a[ok, 0].astype(np.int64) - 1, wi_c[ok]] = a[ok, 2]
    con.close()
    P = Pw[:, week_of_day]

    s = data.series
    state = s["store_id"].str[:2].map({"CA": 0, "TX": 1, "WI": 2}).fillna(0).astype(int).to_numpy()
    return Exog(
        P=P,
        wday=cal["wday"].to_numpy(dtype=np.float32),
        month=cal["month"].to_numpy(dtype=np.float32),
        dom=cal["date"].str[8:10].astype(int).to_numpy(dtype=np.float32),
        event_type=pd.factorize(cal["event_type_1"])[0].astype(np.float32) + 1,
        event_name=pd.factorize(cal["event_name_1"])[0].astype(np.float32) + 1,
        snap=cal[["snap_ca", "snap_tx", "snap_wi"]].to_numpy(dtype=np.float32).T,
        state_idx=state,
        store=pd.Categorical(s["store_id"]).codes.astype(np.float32),
        dept=pd.Categorical(s["dept_id"]).codes.astype(np.float32),
        cat=pd.Categorical(s["cat_id"]).codes.astype(np.float32),
    )


def _weekday_means(Yf, o, weeks):
    """(n_series, 7): mean of the last `weeks` occurrences of each weekday offset before o."""
    block = Yf[:, o - 7 * weeks:o].reshape(Yf.shape[0], weeks, 7)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(block, axis=1)


def origin_features(Yf, P, first_idx, o):
    """Features that depend only on data before the origin o. Every value has shape (n_series,)."""
    f = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for k in LAGS:
            f[f"lag{k}"] = Yf[:, o - k]
        for w in WINDOWS:
            f[f"rm{w}"] = np.nanmean(Yf[:, o - w:o], axis=1)
        w28 = Yf[:, o - 28:o]
        f["rstd28"] = np.nanstd(w28, axis=1)
        act = ~np.isnan(w28)
        with np.errstate(invalid="ignore"):
            f["zero_rate28"] = ((w28 == 0) & act).sum(axis=1) / np.maximum(act.sum(axis=1), 1)
            pos = Yf[:, o - 112:o] > 0
        f["days_since_sale"] = np.where(pos.any(axis=1), np.argmax(pos[:, ::-1], axis=1), 112)
        f["rm7_over_rm28"] = f["rm7"] / (f["rm28"] + 0.01)
        f["age_days"] = np.clip(o - first_idx, 0, None)
        f["p_last"] = P[:, o - 1]
        f["p_mean28"] = np.nanmean(P[:, o - 28:o], axis=1)
        f["p_hist"] = np.nanmean(P[:, max(0, o - 364):o], axis=1)
        f["sw4"] = _weekday_means(Yf, o, 4)
        f["sw8"] = _weekday_means(Yf, o, 8)
    return f


def build_rows(Yf, exog, first_idx, o, h_per_series=None, rng=None, min_history=None):
    """Rows for one forecast origin.

    h_per_series=None -> all 28 horizons for every eligible series (validation and test).
    h_per_series=k    -> k random horizons per series (training, keeps the data small).
    Returns X (float32), y (float32, NaN where unknown), series index, horizon index (0-based).
    """
    min_history = config.MIN_HISTORY_DAYS if min_history is None else min_history
    elig = np.flatnonzero(first_idx <= o - min_history)
    if h_per_series is None:
        hh = np.tile(np.arange(H), (len(elig), 1))
    else:
        rng = rng or np.random.default_rng(0)
        hh = np.argsort(rng.random((len(elig), H)), axis=1)[:, :h_per_series]
    s = np.repeat(elig, hh.shape[1])
    h = hh.ravel()
    t = o + h

    base = origin_features(Yf, exog.P, first_idx, o)
    price_t = exog.P[s, t]
    with np.errstate(invalid="ignore", divide="ignore"):
        cols = {
            "h": h + 1,
            "wday": exog.wday[t], "month": exog.month[t], "dom": exog.dom[t],
            "event_type": exog.event_type[t], "event_name": exog.event_name[t],
            "snap": exog.snap[exog.state_idx[s], t],
            "price_t": price_t,
            "price_ratio_28": price_t / base["p_mean28"][s],
            "price_ratio_hist": price_t / base["p_hist"][s],
            "price_changed": np.where(np.isnan(price_t) | np.isnan(base["p_last"][s]), np.nan,
                                      (price_t != base["p_last"][s]).astype(np.float32)),
            "sw4": base["sw4"][s, h % 7],
            "sw8": base["sw8"][s, h % 7],
            "store": exog.store[s], "dept": exog.dept[s], "cat": exog.cat[s],
        }
    for name, arr in base.items():
        if name in ("sw4", "sw8", "p_last", "p_mean28", "p_hist"):
            continue
        cols[name] = arr[s]
    X = np.column_stack([cols[name] for name in FEATURES]).astype(np.float32)
    y = Yf[s, t].astype(np.float32)
    return X, y, s, h
