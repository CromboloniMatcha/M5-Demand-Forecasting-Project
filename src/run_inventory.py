"""Notebook 04: inventory simulation. How much does a better forecast save?

Usage (from the project root, after python -m src.run_week2 and python -m src.run_lgbm):
    python -m src.run_inventory
    python -m src.run_inventory --lead-time 14 --service-level 0.98

WHAT IS SIMULATED
    Every series is run through a periodic-review "order-up-to" policy: every R days the planner
    orders enough to bring the inventory position (on hand + on order) up to S, where S is the
    alpha-quantile of demand over the next (L + R) days, L = lead time. Orders arrive L days later.
    Unmet demand is a lost sale (no backorders).
    Demand over (L + R) days is modelled as negative binomial (Poisson when there is no extra noise):
    its mean is the sum of the forecast, its extra variance (dispersion phi) is measured from the
    method's OWN past forecast errors, pooled by demand pattern and volume. A better forecast has
    smaller errors, so it needs less safety stock for the same service level.

HONESTY NOTES
    * M5 has no stock, cost or lead-time data. Lead time, review period and the cost ratios are
      assumptions (src/config.py). Costs are computed in the notebook so they can be changed.
    * Forecasts are made once per 28-day origin and used for the whole window (monthly forecast run).
    * Windows simulated: backtest origins 2..N. Dispersion uses only EARLIER origins (no leakage).
    * An "oracle" that knows the future demand shows the best any forecast could ever do.

Outputs (reports/tables/): inventory_results.csv, inventory_assumptions.csv,
    inventory_calibration.csv, replenishment_plan.csv
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

from src import config
from src.baselines import BASELINES
from src.data_access import load_sales_data, masked_float
from src.run_week2 import LEVELS, get_cutoffs

H = config.HORIZON


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def nb_quantile(alpha, mu, phi):
    """alpha-quantile of a negative binomial with mean mu and variance mu + phi * mu^2."""
    mu = np.asarray(mu, dtype=np.float64)
    phi = np.broadcast_to(np.asarray(phi, dtype=np.float64), mu.shape)
    out = np.zeros_like(mu)
    pos = mu > 1e-9
    pois = pos & (phi < 1e-3)
    nb = pos & ~pois
    if pois.any():
        out[pois] = poisson.ppf(alpha, mu[pois])
    if nb.any():
        n_ = 1.0 / phi[nb]
        out[nb] = nbinom.ppf(alpha, n_, n_ / (n_ + mu[nb]))
    return out


def make_groups(a_cal, pattern):
    """Group = demand pattern x volume tercile (volume measured on calibration demand, same for every method)."""
    mean_d = a_cal.mean(axis=1)
    bucket = np.digitize(mean_d, np.quantile(mean_d, [1 / 3, 2 / 3]))
    return np.char.add(np.char.add(pattern.astype(str), "|v"), bucket.astype(str))


def estimate_dispersion(a_cal, f_cal, group, block):
    """phi per series: pooled method of moments, Var(error over a block) = mu + phi * mu^2."""
    n, d = a_cal.shape
    nb = d // block
    a_b = a_cal[:, :nb * block].reshape(n, nb, block).sum(axis=2)
    f_b = f_cal[:, :nb * block].reshape(n, nb, block).sum(axis=2)
    num = ((a_b - f_b) ** 2 - f_b).sum(axis=1)
    den = (f_b ** 2).sum(axis=1)
    phi = np.zeros(n)
    table = []
    for g in np.unique(group):
        m = group == g
        d_g = den[m].sum()
        val = float(np.clip(num[m].sum() / d_g, 0.0, 10.0)) if d_g > 0 else 0.0
        phi[m] = val
        table.append((g, int(m.sum()), val))
    return phi, table


def simulate(A, order_up_to, lead, review):
    """Periodic-review order-up-to simulation, vectorised over series.

    A: (n, T) actual demand. order_up_to(d): (n,) level computed on review day d using only
    information available on day d. Starts with on-hand = order_up_to(0). Lost sales, no backorders.
    Returns sold, lost and the sum over days of end-of-day on-hand units, each shape (n,).
    """
    n, T = A.shape
    if lead + review > T:
        raise ValueError("lead time + review period must not exceed the simulated window")
    on_hand = order_up_to(0).astype(np.float64)
    arrivals = np.zeros((n, T + lead + 1))
    sold = np.zeros(n)
    lost = np.zeros(n)
    inv_days = np.zeros(n)
    for d in range(T):
        on_hand += arrivals[:, d]
        if d % review == 0 and d + lead + review <= T:
            on_order = arrivals[:, d + 1:].sum(axis=1)
            q = np.maximum(0.0, order_up_to(d) - (on_hand + on_order))
            arrivals[:, d + lead] += q
        s = np.minimum(on_hand, A[:, d])
        sold += s
        lost += A[:, d] - s
        on_hand -= s
        inv_days += on_hand
    return sold, lost, inv_days


def main():
    p = argparse.ArgumentParser(description="Inventory simulation")
    p.add_argument("--db", type=Path, default=config.DB_PATH)
    p.add_argument("--lead-time", type=int, default=config.INV_LEAD_TIME)
    p.add_argument("--review-period", type=int, default=config.INV_REVIEW_PERIOD)
    p.add_argument("--service-level", type=float, default=config.INV_SERVICE_LEVEL)
    args = p.parse_args()
    L, R = args.lead_time, args.review_period
    target = args.service_level
    alphas = sorted(set(config.INV_ALPHA_GRID) | {target})

    npz_path = config.PROCESSED_DIR / "lgbm_forecasts.npz"
    seg_path = config.TABLES_DIR / "series_segments.csv"
    if not npz_path.exists():
        raise SystemExit("lgbm_forecasts.npz not found. Run (again): python -m src.run_lgbm")
    if not seg_path.exists():
        raise SystemExit("series_segments.csv not found. Run first: python -m src.run_week2")

    t_start = time.time()
    data = load_sales_data(args.db)
    Yf = masked_float(data)
    Y = data.Y
    cutoffs = get_cutoffs(data.n_days)
    if len(cutoffs) < 2:
        raise SystemExit("Need at least 2 backtest origins (N_ORIGINS in config.py).")
    z = np.load(npz_path)
    if list(z["cutoffs"]) != list(cutoffs):
        raise SystemExit("lgbm_forecasts.npz was made with other cutoffs. Re-run: python -m src.run_lgbm")

    seg = pd.read_csv(seg_path)
    if not np.array_equal(seg["series_id"].to_numpy(), data.series["series_id"].to_numpy()):
        raise SystemExit("series_segments.csv does not match this database. Re-run: python -m src.run_week2")
    seg["all"] = "all"

    price_all = data.series["avg_price"].to_numpy()
    elig = (data.first_idx <= cutoffs[0] - config.MIN_HISTORY_DAYS) & (price_all > 0)
    idx = np.flatnonzero(elig)
    price = price_all[idx]
    pattern = seg["demand_pattern"].to_numpy()[idx]
    labels = seg.iloc[idx][LEVELS].reset_index(drop=True)
    log(f"{len(idx):,} series simulated | lead time {L}d, review {R}d, target service level {target:.0%}")

    # forecasts and actuals of every origin, restricted to the simulated series
    methods = [m for m in config.INV_METHODS]
    FC = {m: {} for m in methods}
    AC = {}
    for oi, c in enumerate(cutoffs, 1):
        AC[oi] = Y[idx, c:c + H].astype(np.float64)
        for m in methods:
            if m == "lightgbm":
                FC[m][oi] = z[f"F{oi}"][idx].astype(np.float64)
            else:
                FC[m][oi] = np.nan_to_num(BASELINES[m](Yf, c, H), nan=0.0)[idx].astype(np.float64)

    sim_origins = list(range(2, len(cutoffs) + 1))
    sim_days = len(sim_origins) * H
    keys = [(m, a) for m in methods for a in alphas] + [("oracle", np.nan)]
    cols = ["demand_units", "sold_units", "lost_units", "demand_rev", "lost_rev", "inv_rev_days"]
    acc = {k: {c: np.zeros(len(idx)) for c in cols} for k in keys}
    calib_rows, plan = [], None

    for oi in sim_origins:
        a_cal = np.hstack([AC[j] for j in range(1, oi)])
        grp = make_groups(a_cal, pattern)
        A = AC[oi]
        log(f"simulating origin {oi} (window starts {data.dates[cutoffs[oi - 1]]}), "
            f"calibration on {oi - 1} earlier window(s)")
        for m in methods:
            f_cal = np.hstack([FC[m][j] for j in range(1, oi)])
            phi, table = estimate_dispersion(a_cal, f_cal, grp, L + R)
            for g, n_g, val in table:
                calib_rows.append({"origin": oi, "method": m, "group": g, "n_series": n_g, "dispersion_phi": round(val, 4)})
            F = FC[m][oi]
            for a in alphas:
                def S(d, F=F, phi=phi, a=a):
                    return nb_quantile(a, F[:, d:d + L + R].sum(axis=1), phi)
                sold, lost, inv = simulate(A, S, L, R)
                r = acc[(m, a)]
                r["demand_units"] += A.sum(axis=1); r["sold_units"] += sold; r["lost_units"] += lost
                r["demand_rev"] += A.sum(axis=1) * price; r["lost_rev"] += lost * price; r["inv_rev_days"] += inv * price
            if oi == sim_origins[-1] and m == "lightgbm":
                mu_cover = F[:, :L + R].sum(axis=1)
                mu_lead = F[:, :L].sum(axis=1)
                s_level = nb_quantile(target, mu_cover, phi)
                plan = pd.DataFrame({
                    "series_id": seg["series_id"].to_numpy()[idx], "item_id": seg["item_id"].to_numpy()[idx],
                    "store_id": seg["store_id"].to_numpy()[idx], "cat_id": seg["cat_id"].to_numpy()[idx],
                    "abc_xyz": seg["abc_xyz"].to_numpy()[idx], "demand_pattern": pattern, "price": np.round(price, 2),
                    "forecast_cover_demand": np.round(mu_cover, 2),
                    "safety_stock": np.maximum(s_level - mu_cover, 0).round(0),
                    "order_up_to": s_level,
                    "reorder_point": nb_quantile(target, mu_lead, phi),
                    "dispersion_phi": np.round(phi, 3),
                    "actual_cover_demand": A[:, :L + R].sum(axis=1)})
        # oracle: knows the future, so order-up-to = true demand over the cover period
        sold, lost, inv = simulate(A, lambda d: A[:, d:d + L + R].sum(axis=1), L, R)
        r = acc[("oracle", np.nan)]
        r["demand_units"] += A.sum(axis=1); r["sold_units"] += sold; r["lost_units"] += lost
        r["demand_rev"] += A.sum(axis=1) * price; r["lost_rev"] += lost * price; r["inv_rev_days"] += inv * price

    rows = []
    for (m, a), r in acc.items():
        df = pd.concat([pd.DataFrame(r), labels], axis=1)
        for level in LEVELS:
            g = df.groupby(level)
            agg = g[cols].sum()
            agg["n_series"] = g.size()
            for grp_name, rec in agg.iterrows():
                rows.append({"method": m, "alpha": a, "level": level, "group": grp_name,
                             "n_series": int(rec["n_series"]), "sim_days": sim_days,
                             **{c: float(rec[c]) for c in cols}})
    res = pd.DataFrame(rows)
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(config.TABLES_DIR / "inventory_results.csv", index=False)
    pd.DataFrame(calib_rows).to_csv(config.TABLES_DIR / "inventory_calibration.csv", index=False)
    plan.to_csv(config.TABLES_DIR / "replenishment_plan.csv", index=False)
    assumptions = [
        ("policy", "periodic review, order-up-to (R, S), lost sales, no backorders"),
        ("lead_time_days", L), ("review_period_days", R), ("target_service_level", target),
        ("service_levels_tried", ", ".join(f"{a:.2f}" for a in alphas)),
        ("demand_model", "negative binomial over lead time + review period; dispersion from the method's own past errors"),
        ("simulated_windows", ", ".join(f"{data.dates[cutoffs[o - 1]]} (+{H}d)" for o in sim_origins)),
        ("n_series", len(idx)), ("forecast_refresh", "once per 28-day origin"),
        ("oracle", "knows true future demand; lower bound for cost")]
    pd.DataFrame(assumptions, columns=["assumption", "value"]).to_csv(
        config.TABLES_DIR / "inventory_assumptions.csv", index=False)

    # ---- console summary at the target service level (default cost assumptions) ----
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
    s = res[(res["level"] == "all") & ((res["alpha"] == target) | res["alpha"].isna())].set_index("method")
    out = pd.DataFrame({
        "fill_rate_units": s["sold_units"] / s["demand_units"],
        "avg_inventory_$": s["inv_rev_days"] * config.INV_COST_RATIO / s["sim_days"],
        "lost_revenue_$": s["lost_rev"]})
    out["holding_cost_$"] = s["inv_rev_days"] * config.INV_COST_RATIO * config.INV_HOLDING_RATE / 365
    out["stockout_cost_$"] = s["lost_rev"] * ((1 - config.INV_COST_RATIO) + config.INV_STOCKOUT_PENALTY)
    out["total_cost_$"] = out["holding_cost_$"] + out["stockout_cost_$"]
    print(f"\nAt target service level {target:.0%} (default cost assumptions):")
    print(out.sort_values("total_cost_$"))
    log(f"done in {time.time() - t_start:,.0f}s. Tables saved to {config.TABLES_DIR}")
    log("next: open notebooks/04_inventory_simulation.ipynb")


if __name__ == "__main__":
    main()
