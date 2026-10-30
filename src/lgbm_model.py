"""Thin wrapper around LightGBM.

LightGBM is the real model. An explicit, opt-in scikit-learn fallback exists only so the
pipeline can be tested on machines without LightGBM (pass --allow-sklearn). Its results are
NOT the project results.
"""
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config
from src.features import CATEGORICAL, FEATURES


def lightgbm_available():
    try:
        import lightgbm  # noqa: F401
        return True
    except ImportError:
        return False


@dataclass
class Trained:
    backend: str
    model: object
    best_iteration: int
    valid_loss: float
    seconds: float


def train_model(Xtr, ytr, Xva, yva, allow_sklearn=False, params=None):
    params = dict(config.LGB_PARAMS if params is None else params)
    t0 = time.time()
    if lightgbm_available():
        import lightgbm as lgb
        dtr = lgb.Dataset(Xtr, label=ytr, feature_name=list(FEATURES),
                          categorical_feature=list(CATEGORICAL), free_raw_data=False)
        dva = lgb.Dataset(Xva, label=yva, reference=dtr)
        booster = lgb.train(
            params, dtr, num_boost_round=config.LGB_MAX_ROUNDS,
            valid_sets=[dva], valid_names=["valid"],
            callbacks=[lgb.early_stopping(config.LGB_EARLY_STOP, verbose=False), lgb.log_evaluation(0)])
        score = booster.best_score.get("valid", {})
        loss = float(next(iter(score.values()))) if score else float("nan")
        best = int(booster.best_iteration or booster.current_iteration())
        return Trained("lightgbm", booster, best, loss, time.time() - t0)

    if not allow_sklearn:
        raise RuntimeError("LightGBM is not installed. Install it with: pip install lightgbm")
    from sklearn.ensemble import HistGradientBoostingRegressor
    cat_idx = [FEATURES.index(c) for c in CATEGORICAL]
    model = HistGradientBoostingRegressor(
        loss="poisson", learning_rate=0.1, max_iter=150, max_leaf_nodes=31, min_samples_leaf=50,
        l2_regularization=1.0, categorical_features=cat_idx, random_state=42)
    model.fit(Xtr, ytr)
    rmse = float(np.sqrt(np.mean((model.predict(Xva) - yva) ** 2)))
    return Trained("sklearn_fallback", model, int(model.n_iter_), rmse, time.time() - t0)


def predict(trained, X):
    if trained.backend == "lightgbm":
        p = trained.model.predict(X, num_iteration=trained.best_iteration)
    else:
        p = trained.model.predict(X)
    return np.clip(np.asarray(p, dtype=np.float64), 0, None)


def shap_values(trained, X):
    """TreeSHAP values from LightGBM itself (no extra library). Shape (n, n_features + 1).
    The last column is the base value. Values are in log space (the Tweedie link)."""
    if trained.backend != "lightgbm":
        return None
    return np.asarray(trained.model.predict(X, num_iteration=trained.best_iteration, pred_contrib=True))


def importance(trained):
    if trained.backend != "lightgbm":
        return None
    m = trained.model
    return pd.DataFrame({"feature": m.feature_name(),
                         "gain": m.feature_importance(importance_type="gain"),
                         "split": m.feature_importance(importance_type="split")})


def save_model(trained, path):
    if trained.backend == "lightgbm":
        trained.model.save_model(str(path), num_iteration=trained.best_iteration)
