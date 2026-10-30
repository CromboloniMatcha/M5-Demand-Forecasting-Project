"""Notebook 03 pipeline: LightGBM global model, evaluated with the SAME backtest as notebook 02.

Usage (from the project root, after python -m src.run_week2):
    python -m src.run_lgbm
    python -m src.run_lgbm --train-origins 26          # faster: about 6 months of training origins
    python -m src.run_lgbm --db data/processed/m5_sample.db --allow-sklearn   # pipeline test only

For every backtest cutoff c (same cutoffs as notebook 02):
    train rows   : weekly origins o <= c - 56  (their targets end before c - 28)
    validation   : origin c - 28, all 28 horizons (early stopping)
    test         : origin c, all 28 horizons (never seen during training)
So no model ever sees data from its own test window.

Outputs (reports/tables/):
    lgbm_metrics.csv             same layout as baseline_metrics.csv, method = lightgbm
    lgbm_horizon_metrics.csv     WAPE and Bias per day ahead (1..28), vs two baselines
    lgbm_aggregate_metrics.csv   bottom-up accuracy at total and store x category level
    lgbm_aggregate_forecasts.csv forecasts vs actuals at those levels
    lgbm_forecasts_last_origin.csv.gz   item-store forecasts for the last 28 days
    lgbm_feature_importance.csv, lgbm_training_log.csv
and data/processed/lgbm_shap_sample.npz, models/lgbm_origin<k>.txt
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import config, lgbm_model
from src.baselines import BASELINES
from src.data_access import load_sales_data, masked_float
from src.features import FEATURES, build_rows, load_exog
from src.metrics import per_series_errors, train_scale
from src.run_week2 import LEVELS, add_rows, get_cutoffs

H = config.HORIZON


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    p = argparse.ArgumentParser(description="Train and backtest the LightGBM model")
    p.add_argument("--db", type=Path, default=config.DB_PATH)
    p.add_argument("--train-origins", type=int, default=config.LGB_TRAIN_ORIGINS)
    p.add_argument("--h-per-origin", type=int, default=config.LGB_H_PER_ORIGIN)
    p.add_argument("--allow-sklearn", action="store_true", help="test the pipeline without LightGBM")
    p.add_argument("--no-shap", action="store_true")
    args = p.parse_args()

    if not lgbm_model.lightgbm_available() and not args.allow_sklearn:
        raise SystemExit("LightGBM is not installed. Install it with: pip install lightgbm")
    seg_path = config.TABLES_DIR / "series_segments.csv"
    if not seg_path.exists():
        raise SystemExit("series_segments.csv not found. Run first: python -m src.run_week2")

    t_start = time.time()
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    data = load_sales_data(args.db)
    Y, first_idx = data.Y, data.first_idx
    Yf = masked_float(data)
    exog = load_exog(args.db, data)
    cutoffs = get_cutoffs(data.n_days)
    price = data.series["avg_price"].to_numpy()
    n = data.n_series

    seg = pd.read_csv(seg_path)
    if not np.array_equal(seg["series_id"].to_numpy(), data.series["series_id"].to_numpy()):
        raise SystemExit("series_segments.csv does not match this database. Re-run: python -m src.run_week2")
    seg["all"] = "all"
    labels_all = seg[LEVELS]

    groups = {"TOTAL": np.arange(n)}
    for (store, cat), g in data.series.groupby(["store_id", "cat_id"]):
        groups[f"{store}|{cat}"] = g.index.to_numpy()
    names = list(groups)
    Yagg = np.vstack([Y[idx].sum(axis=0, dtype=np.float64) for idx in groups.values()]).astype(np.float32)

    rng = np.random.default_rng(42)
    metric_rows, horizon_rows, agg_rows, agg_fc_rows, train_log = [], [], [], [], []
    last = {}

    for oi, c in enumerate(cutoffs, 1):
        t0 = time.time()
        date_c = str(data.dates[c])
        log(f"origin {oi}/{len(cutoffs)} (forecast from {date_c}): building features")
        train_origins = [c - 2 * H - config.LGB_ORIGIN_STEP * k for k in range(args.train_origins)]
        train_origins = [o for o in train_origins if o - 112 >= 0]
        parts = [build_rows(Yf, exog, first_idx, o, args.h_per_origin, rng) for o in train_origins]
        Xtr = np.concatenate([q[0] for q in parts]); ytr = np.concatenate([q[1] for q in parts])
        keep = ~np.isnan(ytr)
        Xtr, ytr = Xtr[keep], ytr[keep]
        Xva, yva, _, _ = build_rows(Yf, exog, first_idx, c - H, None)
        keep = ~np.isnan(yva)
        Xva, yva = Xva[keep], yva[keep]
        log(f"  train rows: {len(ytr):,} | validation rows: {len(yva):,} | features: {len(FEATURES)}")

        trained = lgbm_model.train_model(Xtr, ytr, Xva, yva, allow_sklearn=args.allow_sklearn)
        log(f"  trained ({trained.backend}) in {trained.seconds:,.0f}s, best iteration {trained.best_iteration}")
        train_log.append({"origin": oi, "cutoff_date": date_c, "backend": trained.backend,
                          "n_train_rows": len(ytr), "n_val_rows": len(yva),
                          "best_iteration": trained.best_iteration, "valid_loss": trained.valid_loss,
                          "train_seconds": round(trained.seconds, 1)})
        lgbm_model.save_model(trained, config.MODELS_DIR / f"lgbm_origin{oi}.txt")

        # ---- test origin: all 28 horizons ----
        Xte, _, s_te, h_te = build_rows(Yf, exog, first_idx, c, None)
        pred = lgbm_model.predict(trained, Xte)
        elig_idx = np.flatnonzero(first_idx <= c - config.MIN_HISTORY_DAYS)
        F = np.zeros((n, H))
        F[elig_idx] = pred.reshape(len(elig_idx), H)
        eligible = np.zeros(n, dtype=bool)
        eligible[elig_idx] = True

        A = Y[:, c:c + H].astype(np.float64)
        scale = train_scale(Yf, c)
        weight = np.nansum(Yf[:, c - 28:c], axis=1) * price
        df = per_series_errors(A[eligible], F[eligible], scale[eligible])
        df["weight"] = weight[eligible]
        meta = {"origin": oi, "cutoff_date": date_c, "method": "lightgbm"}
        add_rows(metric_rows, meta, df, labels_all[eligible], "all", 0)

        ma = np.nan_to_num(BASELINES["moving_avg_28d"](Yf, c, H), nan=0.0)
        sa = np.nan_to_num(BASELINES["seasonal_avg_4w"](Yf, c, H), nan=0.0)
        for name, Fm in [("lightgbm", F), ("moving_avg_28d", ma), ("seasonal_avg_4w", sa)]:
            err = Fm[eligible] - A[eligible]
            act = A[eligible].sum(axis=0)
            for h in range(H):
                horizon_rows.append({"origin": oi, "cutoff_date": date_c, "method": name, "h": h + 1,
                                     "WAPE": np.abs(err[:, h]).sum() / act[h] if act[h] > 0 else np.nan,
                                     "Bias": err[:, h].sum() / act[h] if act[h] > 0 else np.nan})

        # ---- bottom-up aggregates (series too new for the model use the moving average) ----
        F_full = F.copy()
        F_full[~eligible] = ma[~eligible]
        Fagg = np.vstack([F_full[idx].sum(axis=0) for idx in groups.values()])
        Aagg = Yagg[:, c:c + H].astype(np.float64)
        e = per_series_errors(Aagg, Fagg, train_scale(Yagg, c))
        for gi, g in enumerate(names):
            act = e.loc[gi, "actual"]
            agg_rows.append({"origin": oi, "cutoff_date": date_c, "method": "lightgbm", "group": g,
                             "WAPE": e.loc[gi, "abs_err"] / act if act > 0 else np.nan,
                             "Bias": e.loc[gi, "err"] / act if act > 0 else np.nan,
                             "RMSSE": e.loc[gi, "rmsse"]})
            agg_fc_rows.append(pd.DataFrame({"origin": oi, "group": g, "method": "lightgbm",
                                             "date": data.dates[c:c + H], "forecast": Fagg[gi],
                                             "actual": Aagg[gi]}))
        o_wape = df["abs_err"].sum() / df["actual"].sum()
        log(f"  origin {oi} done in {time.time() - t0:,.0f}s | WAPE (all series): {o_wape:.3f}")
        last = dict(trained=trained, Xte=Xte, F=F, eligible=eligible, A=A, c=c, elig_idx=elig_idx)

    # ---- extras from the last origin: item-store forecasts, importance, SHAP ----
    c, F, eligible, A, elig_idx = last["c"], last["F"], last["eligible"], last["A"], last["elig_idx"]
    sd = data.series.iloc[elig_idx]
    fc = pd.DataFrame({
        "series_id": np.repeat(sd["series_id"].to_numpy(), H),
        "item_id": np.repeat(sd["item_id"].to_numpy(), H),
        "store_id": np.repeat(sd["store_id"].to_numpy(), H),
        "cat_id": np.repeat(sd["cat_id"].to_numpy(), H),
        "date": np.tile(data.dates[c:c + H], len(elig_idx)),
        "h": np.tile(np.arange(1, H + 1), len(elig_idx)),
        "forecast": np.round(F[elig_idx].ravel(), 4),
        "actual": A[elig_idx].ravel().astype(int)})
    fc.to_csv(config.TABLES_DIR / "lgbm_forecasts_last_origin.csv.gz", index=False, compression="gzip")

    imp = lgbm_model.importance(last["trained"])
    if imp is not None and not args.no_shap:
        Xte = last["Xte"]
        take = rng.choice(len(Xte), size=min(config.LGB_SHAP_ROWS, len(Xte)), replace=False)
        sv = lgbm_model.shap_values(last["trained"], Xte[take])
        np.savez_compressed(config.PROCESSED_DIR / "lgbm_shap_sample.npz",
                            X=Xte[take], shap=sv, features=np.array(FEATURES))
        imp["mean_abs_shap"] = imp["feature"].map(dict(zip(FEATURES, np.abs(sv[:, :-1]).mean(axis=0))))
        log("SHAP values computed")
    if imp is not None:
        imp.sort_values("gain", ascending=False).to_csv(config.TABLES_DIR / "lgbm_feature_importance.csv", index=False)

    pd.DataFrame(metric_rows).to_csv(config.TABLES_DIR / "lgbm_metrics.csv", index=False)
    pd.DataFrame(horizon_rows).to_csv(config.TABLES_DIR / "lgbm_horizon_metrics.csv", index=False)
    pd.DataFrame(agg_rows).to_csv(config.TABLES_DIR / "lgbm_aggregate_metrics.csv", index=False)
    pd.concat(agg_fc_rows, ignore_index=True).to_csv(config.TABLES_DIR / "lgbm_aggregate_forecasts.csv", index=False)
    pd.DataFrame(train_log).to_csv(config.TABLES_DIR / "lgbm_training_log.csv", index=False)

    # ---- console summary vs the notebook 02 baselines ----
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
    m = pd.DataFrame(metric_rows)
    mine = m[(m["level"] == "all")][["WAPE", "Bias", "RMSSE", "wRMSSE"]].mean()
    print("\nLightGBM, mean over origins (all evaluated series):")
    print(mine.to_frame("lightgbm").T)
    base_path = config.TABLES_DIR / "baseline_metrics.csv"
    if base_path.exists():
        b = pd.read_csv(base_path)
        b = b[(b["scope"] == "all") & (b["level"] == "all")]
        print("\nBaselines from notebook 02, same series and origins:")
        print(b.groupby("method")[["WAPE", "Bias", "RMSSE", "wRMSSE"]].mean().sort_values("RMSSE"))
    if last["trained"].backend != "lightgbm":
        print("\nWARNING: these numbers come from the scikit-learn test fallback, not LightGBM.")
    log(f"done in {time.time() - t_start:,.0f}s. Tables saved to {config.TABLES_DIR}")
    log("next: open notebooks/03_lightgbm_model.ipynb")


if __name__ == "__main__":
    main()
