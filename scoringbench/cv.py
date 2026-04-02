"""Cross-validation loop.

Public API
----------
run_fold(X_train, X_test, y_train, y_test, model_factories, seed) -> dict
    Fit and evaluate every model on one pre-split fold.
    Returns {model_name: {metric: value, ...}, ...}

run_cv(X, y, model_factories, n_folds, seed) -> list[dict]
    Run K-fold CV and return one result dict per fold.
"""

import time
from typing import Callable

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

from .metrics import compute_metrics, compute_point_metrics, ENERGY_BETAS
from .wrappers import ProbabilisticWrapper


# ---------------------------------------------------------------------------
# Single-fold evaluation
# ---------------------------------------------------------------------------

def run_fold(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_factories: dict[str, Callable],
    seed: int,
) -> dict:
    """Fit and evaluate every model on one fold.

    Each factory produces a fresh ProbabilisticWrapper.
    If predict_distribution() is not implemented, falls back to point metrics
    only (distributional metrics are set to None).

    Returns {model_name: {mae, rmse, r2, crps, log_score, sharpness,
                          coverage_90, interval_score_90,
                          coverage_95, interval_score_95,
                          train_time}}
    """
    y_test_np = np.asarray(y_test, dtype=float)
    fold_results: dict[str, dict] = {}

    for name, factory in model_factories.items():
        print(f"    [{name}] fitting …", flush=True)
        model: ProbabilisticWrapper = factory()

        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0

        try:
            dist = model.predict_distribution(X_test)
            metrics = compute_metrics(dist, y_test_np)
        except NotImplementedError:
            y_pred = model.predict(X_test)
            metrics = compute_point_metrics(y_test_np, y_pred)
            for key in (
                "crps", "log_score", "sharpness",
                "coverage_90", "interval_score_90",
                "coverage_95", "interval_score_95",
                "crls",
                "wcrps_left", "wcrps_right", "wcrps_center",
                *[f"energy_score_beta_{b}" for b in ENERGY_BETAS],
            ):
                metrics[key] = None

        metrics["train_time"] = elapsed

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        crps_str = f"  CRPS={metrics['crps']:.6f}" if metrics.get("crps") is not None else ""
        print(
            f"    [{name}] MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}"
            f"{crps_str}",
            flush=True,
        )
        fold_results[name] = metrics

    return fold_results


# ---------------------------------------------------------------------------
# K-fold CV
# ---------------------------------------------------------------------------

def run_cv(
    X: pd.DataFrame,
    y: pd.Series,
    model_factories: dict[str, Callable],
    n_folds: int,
    seed: int,
) -> list[dict]:
    """K-fold cross-validation. Returns list of fold result dicts."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    results = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\n  Fold {fold_idx + 1}/{n_folds}", flush=True)
        fold_data = run_fold(
            X.iloc[train_idx], X.iloc[test_idx],
            y.iloc[train_idx], y.iloc[test_idx],
            model_factories, seed,
        )
        fold_data["fold"] = fold_idx
        results.append(fold_data)

    return results
