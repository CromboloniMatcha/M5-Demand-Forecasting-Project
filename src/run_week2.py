"""Week 2: segmentation + baseline forecasts + rolling-origin backtest.

Usage (from the project root):
    python -m src.run_week2                    # fast: 5 baselines on every series
    python -m src.run_week2 --with-stats       # also ETS on a sample + ETS/SARIMA on aggregates (slower)
    python -m src.run_week2 --db data/processed/m5_sample.db --ets-sample 30

Outputs (reports/tables/):
    series_segments.csv       ABC, XYZ and demand pattern for every series
    baseline_metrics.csv      WAPE, Bias, RMSSE, wRMSSE per origin x method x segment
    aggregate_metrics.csv     accuracy at store x category and total level
    aggregate_forecasts.csv   forecasts vs actuals at those levels (for plots and the dashboard)
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import config, stat_models
from src.baselines import BASELINES
from src.data_access import load_sales_data, masked_float
from src.metrics import group_summary, per_series_errors, train_scale
from src.segmentation import segment_series, segment_summary

LEVELS = ["all", "abc", "xyz", "abc_xyz", "demand_pattern", "cat_id", "store_id"]
AGG_BASELINES = ["naive", "seasonal_naive", "seasonal_avg_4w", "moving_avg_28d"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_cutoffs(n_days):
    h, n = config.HORIZON, config.N_ORIGINS
    cutoffs = [n_days - h * (n - k) for k in range(n)]
    if cutoffs[0] < 120:
        raise ValueError("Not enough history for the requested number of backtest origins.")
    return cutoffs


def stratified_sample(seg, mask, n, seed=42):
    """Sample about n series, spread across the ABC-XYZ cells in proportion to their size."""
    rng = np.random.default_rng(seed)
    cand = seg[mask & (seg["abc_xyz"] != "NEW")]
    if len(cand) <= n:
        return np.sort(cand.index.to_numpy())
    picks = []
    for _, g in cand.groupby("abc_xyz"):
        k = max(1, int(round(n * len(g) / len(cand))))
        picks.append(rng.choice(g.index.to_numpy(), size=min(k, len(g)), replace=False))
    return np.sort(np.concatenate(picks))


def add_rows(rows, meta, df, labels, scope, n_fallback):
    """Group per-series errors by every segment level and append summary rows."""
    full = pd.concat([df.reset_index(drop=True), labels.reset_index(drop=True)], axis=1)
    for level in LEVELS:
        gs = group_summary(full, level)
        for rec in gs.to_dict("records"):
            rows.append({**meta, "scope": scope, "level": level, "n_fallback": n_fallback, **rec})


def evaluate_series_level(data, Yf, seg, cutoffs, with_stats, ets_sample):
    n, H = data.n_series, config.HORIZON
    Y, price = data.Y, data.series["avg_price"].to_numpy()
    labels_all = seg[LEVELS]
    elig_all = data.first_idx <= cutoffs[0] - config.MIN_HISTORY_DAYS
    sample_idx = stratified_sample(seg, elig_all, ets_sample) if with_stats else None
    if with_stats:
        log(f"ETS sample: {len(sample_idx)} series")

    rows = []
    for oi, c in enumerate(cutoffs, 1):
        t0 = time.time()
        meta = {"origin": oi, "cutoff_date": str(data.dates[c])}
        A = Y[:, c:c + H].astype(np.float64)
        scale = train_scale(Yf, c)
        eligible = data.first_idx <= c - config.MIN_HISTORY_DAYS
        weight = np.nansum(Yf[:, c - 28:c], axis=1) * price     # revenue proxy, last 28 days
        log(f"origin {oi}/{len(cutoffs)}: forecasting {H} days from {meta['cutoff_date']}, "
            f"{int(eligible.sum()):,} series evaluated")

        forecasts = {name: np.nan_to_num(fn(Yf, c, H), nan=0.0) for name, fn in BASELINES.items()}
        fallbacks = {}
        sample_mask = np.zeros(n, dtype=bool)
        if with_stats:
            sidx = sample_idx[eligible[sample_idx]]
            sample_mask[sidx] = True
            fc, _ = stat_models.forecast_many(Yf, sidx, c, H, stat_models.ets_forecast, log=log)
            bad = np.isnan(fc).any(axis=1)
            fc[bad] = forecasts["seasonal_avg_4w"][sidx][bad]    # fallback so every series has a forecast
            F = np.zeros((n, H))
            F[sidx] = fc
            forecasts["ets"] = F
            fallbacks["ets"] = int(bad.sum())

        for name, F in forecasts.items():
            scopes = [("ets_sample", eligible & sample_mask)] if with_stats else []
            if name != "ets":
                scopes.append(("all", eligible))
            for scope, m in scopes:
                if not m.any():
                    continue
                df = per_series_errors(A[m], F[m], scale[m])
                df["weight"] = weight[m]
                add_rows(rows, {**meta, "method": name}, df, labels_all[m], scope, fallbacks.get(name, 0))
        log(f"  origin {oi} done in {time.time() - t0:,.0f}s")
    return pd.DataFrame(rows)


def evaluate_aggregates(data, with_stats, cutoffs):
    """Accuracy at total and store x category level, where demand is smooth."""
    H = config.HORIZON
    groups = {"TOTAL": np.arange(data.n_series)}
    for (store, cat), g in data.series.groupby(["store_id", "cat_id"]):
        groups[f"{store}|{cat}"] = g.index.to_numpy()
    names = list(groups)
    Yagg = np.vstack([data.Y[idx].sum(axis=0, dtype=np.float64) for idx in groups.values()]).astype(np.float32)

    metric_rows, fc_rows = [], []
    for oi, c in enumerate(cutoffs, 1):
        log(f"aggregates, origin {oi}/{len(cutoffs)}")
        A = Yagg[:, c:c + H].astype(np.float64)
        scale = train_scale(Yagg, c)
        methods = {m: np.nan_to_num(BASELINES[m](Yagg, c, H), nan=0.0) for m in AGG_BASELINES}
        if with_stats:
            rows = np.arange(len(names))
            for label, fn in [("ets", stat_models.ets_forecast), ("sarima", stat_models.sarima_forecast)]:
                fc, failed = stat_models.forecast_many(Yagg, rows, c, H, fn)
                bad = np.isnan(fc).any(axis=1)
                fc[bad] = methods["seasonal_avg_4w"][bad]
                methods[label] = fc
                if failed:
                    log(f"  {label}: {failed} aggregate series fell back to seasonal_avg_4w")
        for method, F in methods.items():
            e = per_series_errors(A, F, scale)
            for gi, g in enumerate(names):
                act = e.loc[gi, "actual"]
                metric_rows.append({
                    "origin": oi, "cutoff_date": str(data.dates[c]), "method": method, "group": g,
                    "WAPE": e.loc[gi, "abs_err"] / act if act > 0 else np.nan,
                    "Bias": e.loc[gi, "err"] / act if act > 0 else np.nan,
                    "RMSSE": e.loc[gi, "rmsse"]})
                fc_rows.append(pd.DataFrame({
                    "origin": oi, "group": g, "method": method,
                    "date": data.dates[c:c + H], "forecast": F[gi], "actual": A[gi]}))
    return pd.DataFrame(metric_rows), pd.concat(fc_rows, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Week 2: segmentation and baseline backtest")
    parser.add_argument("--db", type=Path, default=config.DB_PATH)
    parser.add_argument("--with-stats", action="store_true", help="also run ETS and SARIMA (needs statsmodels)")
    parser.add_argument("--ets-sample", type=int, default=config.ETS_SAMPLE_SIZE)
    args = parser.parse_args()

    with_stats = args.with_stats
    if with_stats and not stat_models.statsmodels_available():
        log("statsmodels is not installed, so ETS and SARIMA are skipped. Install: pip install statsmodels")
        with_stats = False

    pd.set_option("display.width", 170)
    pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    data = load_sales_data(args.db)
    Yf = masked_float(data)
    cutoffs = get_cutoffs(data.n_days)
    log(f"{data.n_series:,} series x {data.n_days} days | backtest cutoffs: "
        + ", ".join(str(data.dates[c]) for c in cutoffs))

    log("segmenting series (ABC, XYZ, demand pattern)")
    seg = segment_series(data, Yf, cutoffs[0])
    seg["all"] = "all"
    seg.drop(columns=["all"]).to_csv(config.TABLES_DIR / "series_segments.csv", index=False)
    cells, patterns = segment_summary(seg)
    print("\nABC-XYZ cells:\n", cells.round(1).to_string(index=False))
    print("\nDemand patterns:\n", patterns.round(1).to_string(index=False), "\n")

    metrics = evaluate_series_level(data, Yf, seg, cutoffs, with_stats, args.ets_sample)
    metrics.to_csv(config.TABLES_DIR / "baseline_metrics.csv", index=False)

    agg_metrics, agg_fc = evaluate_aggregates(data, with_stats, cutoffs)
    agg_metrics.to_csv(config.TABLES_DIR / "aggregate_metrics.csv", index=False)
    agg_fc.to_csv(config.TABLES_DIR / "aggregate_forecasts.csv", index=False)

    overall = metrics[(metrics["scope"] == "all") & (metrics["level"] == "all")]
    print("\nAll evaluated series, mean over origins (lower is better):")
    print(overall.groupby("method")[["WAPE", "Bias", "RMSSE", "wRMSSE"]].mean().sort_values("RMSSE"))
    bp = metrics[(metrics["scope"] == "all") & (metrics["level"] == "demand_pattern")]
    print("\nRMSSE by demand pattern (mean over origins):")
    print(bp.groupby(["group", "method"])["RMSSE"].mean().unstack("method"))
    print("\nTotal-level WAPE (mean over origins):")
    print(agg_metrics[agg_metrics["group"] == "TOTAL"].groupby("method")["WAPE"].mean().sort_values())

    log(f"done in {time.time() - t_start:,.0f}s. Tables saved to {config.TABLES_DIR}")
    log("next: open notebooks/02_baselines.ipynb")


if __name__ == "__main__":
    main()
