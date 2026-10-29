"""Build the SQLite database from the raw M5 CSV files.

Usage (from the project root):
    python -m src.build_database
    python -m src.build_database --states CA TX
    python -m src.build_database --raw-dir data/sample_raw --db data/processed/m5_sample.db

What it does:
    1. Creates the star schema (sql/01_schema.sql)
    2. Loads calendar, then sales (wide -> long, chunk by chunk), then prices
    3. Builds indexes, a view and aggregate tables (sql/02_indexes_views.sql)

The old database file is deleted first, so the script is safe to re-run.
"""
import argparse
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import config


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_sql_file(con, path):
    con.executescript(Path(path).read_text(encoding="utf-8"))


def find_sales_file(raw_dir):
    for name in config.SALES_FILES:
        path = raw_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No sales file found in {raw_dir}. Expected one of: {', '.join(config.SALES_FILES)}.\n"
        "Download the M5 data from Kaggle (see README) and put the CSVs in data/raw/."
    )


def need(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return path


def load_calendar(con, raw_dir):
    cal = pd.read_csv(need(raw_dir / config.CALENDAR_FILE))
    cal["d_num"] = cal["d"].str.replace("d_", "", regex=False).astype(int)
    cal = cal.rename(columns={"snap_CA": "snap_ca", "snap_TX": "snap_tx", "snap_WI": "snap_wi"})
    cols = ["d_num", "date", "wm_yr_wk", "weekday", "wday", "month", "year",
            "event_name_1", "event_type_1", "event_name_2", "event_type_2",
            "snap_ca", "snap_tx", "snap_wi"]
    cal = cal[cols]
    cal = cal.astype(object).where(cal.notna(), None)  # NaN -> NULL
    con.executemany(
        f"INSERT INTO dim_calendar VALUES ({','.join('?' * len(cols))})",
        cal.itertuples(index=False, name=None),
    )
    con.commit()
    log(f"calendar loaded: {len(cal):,} days")


def load_sales(con, raw_dir, states):
    sales_path = find_sales_file(raw_dir)
    header = pd.read_csv(sales_path, nrows=0)
    day_cols = [c for c in header.columns if c.startswith("d_")]
    d_nums = np.array([int(c[2:]) for c in day_cols], dtype=np.int64)
    dtypes = {c: "int32" for c in day_cols}
    log(f"sales file: {sales_path.name} ({len(day_cols)} days), loading states: {', '.join(states)}")

    next_id = 1
    n_rows = 0
    for chunk in pd.read_csv(sales_path, chunksize=config.CHUNK_ROWS, dtype=dtypes):
        chunk = chunk[chunk["state_id"].isin(states)].reset_index(drop=True)
        if chunk.empty:
            continue
        n = len(chunk)
        series_ids = np.arange(next_id, next_id + n, dtype=np.int64)
        next_id += n

        # Dimensions
        con.executemany("INSERT OR IGNORE INTO dim_store VALUES (?, ?)",
                        chunk[["store_id", "state_id"]].drop_duplicates().itertuples(index=False, name=None))
        con.executemany("INSERT OR IGNORE INTO dim_item VALUES (?, ?, ?)",
                        chunk[["item_id", "dept_id", "cat_id"]].drop_duplicates().itertuples(index=False, name=None))
        con.executemany("INSERT INTO dim_series VALUES (?, ?, ?)",
                        zip(series_ids.tolist(), chunk["item_id"].tolist(), chunk["store_id"].tolist()))

        # Wide -> long (row-major ravel keeps each series' days together and in order)
        sid_long = np.repeat(series_ids, len(day_cols))
        d_long = np.tile(d_nums, n)
        sales_long = chunk[day_cols].to_numpy().ravel()
        con.executemany("INSERT INTO fact_sales VALUES (?, ?, ?)",
                        zip(sid_long.tolist(), d_long.tolist(), sales_long.tolist()))
        con.commit()
        n_rows += len(sales_long)
        log(f"  series loaded: {next_id - 1:,}  |  sales rows: {n_rows:,}")
    if next_id == 1:
        raise ValueError(f"No rows found for states {states}. Check config.STATES.")


def load_prices(con, raw_dir):
    prices_path = need(raw_dir / config.PRICES_FILE)
    series_map = pd.read_sql_query("SELECT series_id, item_id, store_id FROM dim_series", con)
    stores = set(series_map["store_id"])
    n_rows = 0
    for chunk in pd.read_csv(prices_path, chunksize=1_000_000):
        chunk = chunk[chunk["store_id"].isin(stores)]
        if chunk.empty:
            continue
        chunk = chunk.merge(series_map, on=["store_id", "item_id"], how="inner")
        con.executemany("INSERT INTO fact_prices VALUES (?, ?, ?)",
                        zip(chunk["series_id"].tolist(), chunk["wm_yr_wk"].tolist(), chunk["sell_price"].tolist()))
        con.commit()
        n_rows += len(chunk)
        log(f"  price rows loaded: {n_rows:,}")


def main():
    parser = argparse.ArgumentParser(description="Build the M5 SQLite database")
    parser.add_argument("--raw-dir", type=Path, default=config.RAW_DIR)
    parser.add_argument("--db", type=Path, default=config.DB_PATH)
    parser.add_argument("--states", nargs="+", default=config.STATES)
    args = parser.parse_args()

    args.db.parent.mkdir(parents=True, exist_ok=True)
    if args.db.exists():
        try:
            args.db.unlink()
        except PermissionError:
            raise SystemExit(
                f"Cannot replace {args.db}: another program is still using it.\n"
                "Fix: close the Jupyter notebook (Kernel > Shut Down Kernel) and any VS Code window that opened it,\n"
                "pause OneDrive sync (it can lock big files), then run this command again.")

    start = time.time()
    con = sqlite3.connect(args.db)
    # Speed settings for a one-off bulk build. The DB is rebuildable, so durability is not needed.
    con.executescript("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; "
                      "PRAGMA temp_store=MEMORY; PRAGMA cache_size=-200000;")

    log("creating schema")
    run_sql_file(con, config.SQL_DIR / "01_schema.sql")
    load_calendar(con, args.raw_dir)
    load_sales(con, args.raw_dir, args.states)
    log("loading prices")
    load_prices(con, args.raw_dir)
    log("building indexes, view and aggregate tables (this is the slowest step)")
    run_sql_file(con, config.SQL_DIR / "02_indexes_views.sql")
    con.close()

    size_mb = args.db.stat().st_size / 1e6
    log(f"done in {time.time() - start:,.0f}s  |  database: {args.db}  ({size_mb:,.0f} MB)")
    log("next: python -m src.run_sql sql/03_data_quality.sql")


if __name__ == "__main__":
    main()