"""Generate a small SYNTHETIC dataset in the exact M5 file format.

Use it to test the whole pipeline (build database -> SQL -> notebook) in a minute,
before you download the real 400+ MB of M5 data. The numbers are fake, so never
report results from this data.

Usage (from the project root):
    python -m src.make_sample_data
    python -m src.build_database --raw-dir data/sample_raw --db data/processed/m5_sample.db --states CA TX
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

N_DAYS = 1941
START = pd.Timestamp("2011-01-29")  # d_1 in the real M5 calendar (a Saturday)

DEPTS = {"FOODS": ["FOODS_1", "FOODS_2", "FOODS_3"],
         "HOBBIES": ["HOBBIES_1", "HOBBIES_2"],
         "HOUSEHOLD": ["HOUSEHOLD_1", "HOUSEHOLD_2"]}
STORES = {"CA_1": "CA", "CA_2": "CA", "TX_1": "TX", "WI_1": "WI"}
BASE_LEVEL = {"FOODS": 3.0, "HOBBIES": 0.8, "HOUSEHOLD": 1.2}
EVENTS = {(1, 1): ("NewYear", "National"), (2, 7): ("SuperBowl", "Sporting"),
          (7, 4): ("IndependenceDay", "National"), (10, 31): ("Halloween", "Cultural"),
          (12, 25): ("Christmas", "National")}


def make_calendar():
    dates = pd.date_range(START, periods=N_DAYS)
    week_idx = np.arange(N_DAYS) // 7
    wm_yr_wk = 10000 + (11 + week_idx // 52) * 100 + (week_idx % 52 + 1)
    cal = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "wm_yr_wk": wm_yr_wk,
        "weekday": dates.day_name(),
        "wday": ((dates.dayofweek - 5) % 7) + 1,  # 1 = Saturday, like M5
        "month": dates.month,
        "year": dates.year,
        "d": [f"d_{i}" for i in range(1, N_DAYS + 1)],
    })
    ev = [EVENTS.get((m, d), (None, None)) for m, d in zip(dates.month, dates.day)]
    cal["event_name_1"] = [e[0] for e in ev]
    cal["event_type_1"] = [e[1] for e in ev]
    cal["event_name_2"] = None
    cal["event_type_2"] = None
    dom = dates.day.to_numpy()
    cal["snap_CA"] = (dom <= 10).astype(int)
    cal["snap_TX"] = np.isin(dom, [1, 3, 5, 6, 7, 9, 11, 12, 13, 15]).astype(int)
    cal["snap_WI"] = np.isin(dom, [2, 3, 5, 6, 8, 9, 11, 12, 14, 15]).astype(int)
    return cal, dates, week_idx, wm_yr_wk


def main():
    parser = argparse.ArgumentParser(description="Create synthetic M5-format data")
    parser.add_argument("--out", type=Path, default=config.ROOT / "data" / "sample_raw")
    parser.add_argument("--items-per-dept", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    cal, dates, week_idx, wm_yr_wk = make_calendar()
    cal.to_csv(args.out / config.CALENDAR_FILE, index=False)

    dow_mult = np.array([1.25, 1.2, 0.95, 0.9, 0.9, 0.95, 1.05])  # Sat..Fri
    dow = ((dates.dayofweek - 5) % 7).to_numpy()
    yearly = 1 + 0.12 * np.sin(2 * np.pi * np.arange(N_DAYS) / 365.25)
    is_event = cal["event_name_1"].notna().to_numpy()
    snap_cols = {"CA": cal["snap_CA"].to_numpy(), "TX": cal["snap_TX"].to_numpy(), "WI": cal["snap_WI"].to_numpy()}
    n_weeks = week_idx.max() + 1

    sales_rows, price_rows = [], []
    for cat, depts in DEPTS.items():
        for dept in depts:
            for k in range(1, args.items_per_dept + 1):
                item_id = f"{dept}_{k:03d}"
                item_scale = rng.lognormal(0, 0.8)
                base_price = float(np.round(rng.lognormal(1.2, 0.6), 2)) + 0.5
                for store_id, state in STORES.items():
                    launch = 1 if rng.random() < 0.7 else int(rng.integers(30, 900))
                    level = BASE_LEVEL[cat] * item_scale * rng.lognormal(0, 0.3)

                    # weekly price path: mostly flat, occasional promo or permanent change
                    wk_price = np.full(n_weeks, base_price)
                    for _ in range(int(rng.integers(0, 5))):
                        start_wk = int(rng.integers(0, n_weeks - 4))
                        length = int(rng.integers(1, 5))
                        wk_price[start_wk:start_wk + length] *= rng.choice([0.8, 0.9, 1.1])
                    wk_price = np.round(wk_price, 2)
                    day_price = wk_price[week_idx]

                    lam = (level * dow_mult[dow] * yearly
                           * (1 + 0.0001 * np.arange(N_DAYS))
                           * (1 + 0.25 * is_event)
                           * (1 + (0.12 * snap_cols[state] if cat == "FOODS" else 0))
                           * (day_price / base_price) ** -1.5)
                    sales = rng.poisson(np.clip(lam, 0, None))
                    launch_idx = launch - 1
                    sales[:launch_idx] = 0  # not yet on shelf

                    row = {"id": f"{item_id}_{store_id}_evaluation", "item_id": item_id, "dept_id": dept,
                           "cat_id": cat, "store_id": store_id, "state_id": state}
                    row.update({f"d_{i + 1}": int(v) for i, v in enumerate(sales)})
                    sales_rows.append(row)

                    first_week = week_idx[launch_idx]
                    for w in range(first_week, n_weeks):
                        price_rows.append((store_id, item_id, int(wm_yr_wk[w * 7]), float(wk_price[w])))

    pd.DataFrame(sales_rows).to_csv(args.out / "sales_train_evaluation.csv", index=False)
    pd.DataFrame(price_rows, columns=["store_id", "item_id", "wm_yr_wk", "sell_price"]).to_csv(
        args.out / config.PRICES_FILE, index=False)
    print(f"Synthetic M5-format data written to {args.out} ({len(sales_rows)} series, {N_DAYS} days)")


if __name__ == "__main__":
    main()
