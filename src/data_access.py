"""Load the sales data as a (series x days) matrix.

Reading 24 million rows from SQLite takes a minute or two, so the matrix is cached
next to the database (m5_matrix_cache.pkl). The cache is rebuilt automatically when
the database file changes.
"""
import pickle
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src import config


@dataclass
class SalesData:
    Y: np.ndarray            # int16, shape (n_series, n_days): actual units sold
    series: pd.DataFrame     # one row per series, same order as the rows of Y
    dates: np.ndarray        # ISO date string for every day index
    first_idx: np.ndarray    # first day index (0-based) with a price; n_days if never on sale

    @property
    def n_series(self):
        return self.Y.shape[0]

    @property
    def n_days(self):
        return self.Y.shape[1]


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _connect_readonly(db_path):
    return sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)


def load_sales_data(db_path=None, use_cache=True, verbose=True):
    db_path = Path(db_path or config.DB_PATH)
    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found: {db_path}\nBuild it first: python -m src.build_database")

    cache_path = db_path.with_name(db_path.stem + "_matrix_cache.pkl")
    stat = db_path.stat()
    stamp = (stat.st_mtime_ns, stat.st_size)

    if use_cache and cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                blob = pickle.load(f)
            if blob.get("stamp") == stamp:
                if verbose:
                    _log("sales matrix loaded from cache")
                return SalesData(blob["Y"], blob["series"], blob["dates"], blob["first_idx"])
        except Exception:
            pass  # a broken cache is simply rebuilt

    if verbose:
        _log("reading sales from the database (first time only, a few minutes for California)")
    con = _connect_readonly(db_path)
    required = {"fact_sales", "dim_series", "series_profile", "fact_prices"}
    found = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    if required - found:
        raise RuntimeError(f"Database is incomplete (missing {sorted(required - found)}). "
                           "Re-run: python -m src.build_database")

    n_days = con.execute("SELECT MAX(d_num) FROM fact_sales").fetchone()[0]
    series = pd.read_sql_query(
        """
        SELECT sr.series_id, sr.item_id, sr.store_id, i.dept_id, i.cat_id,
               p.first_price_d, COALESCE(pr.avg_price, 0.0) AS avg_price
        FROM dim_series sr
        JOIN dim_item i ON i.item_id = sr.item_id
        JOIN series_profile p ON p.series_id = sr.series_id
        LEFT JOIN (SELECT series_id, AVG(sell_price) AS avg_price
                   FROM fact_prices GROUP BY series_id) pr ON pr.series_id = sr.series_id
        ORDER BY sr.series_id
        """, con)
    n_series = len(series)
    if not np.array_equal(series["series_id"].to_numpy(), np.arange(1, n_series + 1)):
        raise RuntimeError("series_id values are not 1..N. Rebuild the database.")

    dates = np.array([r[0] for r in con.execute(
        "SELECT date FROM dim_calendar WHERE d_num <= ? ORDER BY d_num", (n_days,))])

    Y = np.zeros((n_series, n_days), dtype=np.int16)
    cur = con.execute("SELECT series_id, d_num, sales FROM fact_sales")
    n_read = 0
    while True:
        rows = cur.fetchmany(2_000_000)
        if not rows:
            break
        a = np.asarray(rows, dtype=np.int64)
        Y[a[:, 0] - 1, a[:, 1] - 1] = a[:, 2]
        n_read += len(rows)
        if verbose:
            _log(f"  rows read: {n_read:,}")
    con.close()

    first_price_d = series["first_price_d"].fillna(n_days + 1).to_numpy()
    first_idx = np.minimum(first_price_d.astype(np.int64) - 1, n_days)

    try:
        with open(cache_path, "wb") as f:
            pickle.dump({"stamp": stamp, "Y": Y, "series": series, "dates": dates,
                         "first_idx": first_idx}, f, protocol=pickle.HIGHEST_PROTOCOL)
    except OSError:
        pass  # caching is optional

    return SalesData(Y, series, dates, first_idx)


def masked_float(data):
    """Sales as float32 with NaN before each series' launch (days with no price).

    Those days are not real zero demand, so models and metrics must not see them.
    """
    Yf = data.Y.astype(np.float32)
    before_launch = np.arange(data.n_days)[None, :] < data.first_idx[:, None]
    Yf[before_launch] = np.nan
    return Yf
