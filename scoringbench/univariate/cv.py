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
import traceback
import gc
from typing import Callable
import json

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer

from .metrics import compute_metrics, compute_point_metrics, ENERGY_BETAS, DPD_BETAS, CRTS_ALPHAS
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
    
    Imputation is performed within this fold:
    - fit_transform on X_train only (learn statistics from train data)
    - transform on X_test (apply train statistics)
    This ensures no data leakage from test to train via imputation statistics.

    Returns {model_name: {mae, rmse, r2, crps, sharpness,
                          coverage_90, interval_score_90,
                          coverage_95, interval_score_95,
                          crts_alpha_{1.01,...,2.0}, train_time}}
    """
    # Impute missing values (learn from train, apply to both train and test)
    if X_train.isna().sum().sum() > 0 or X_test.isna().sum().sum() > 0:
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        categorical_cols = X_train.select_dtypes(exclude=[np.number]).columns

        if len(numeric_cols) > 0:
            numeric_imputer = SimpleImputer(strategy='median')
            numeric_imputer.fit(X_train[numeric_cols])
            X_train[numeric_cols] = numeric_imputer.transform(X_train[numeric_cols])
            X_test[numeric_cols] = numeric_imputer.transform(X_test[numeric_cols])

        if len(categorical_cols) > 0:
            categorical_imputer = SimpleImputer(strategy='most_frequent')
            categorical_imputer.fit(X_train[categorical_cols])
            X_train[categorical_cols] = categorical_imputer.transform(X_train[categorical_cols])
            X_test[categorical_cols] = categorical_imputer.transform(X_test[categorical_cols])

    y_test_np = np.asarray(y_test, dtype=float)
    fold_results: dict[str, dict] = {}

    for name, factory in model_factories.items():
        print(f"    [{name}] fitting …", flush=True)
        try:
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
                    "crps", "sharpness",
                    "coverage_90", "interval_score_90",
                    "coverage_95", "interval_score_95",
                    "cde_loss",
                    "wcrps_left", "wcrps_right", "wcrps_center",
                    *[f"energy_score_beta_{b}" for b in ENERGY_BETAS],
                    *[f"dpd_beta_{b}" for b in DPD_BETAS],
                    *[f"crts_alpha_{a}" for a in CRTS_ALPHAS],
                ):
                    metrics[key] = None

            metrics["train_time"] = elapsed

            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Convert numpy types to native Python types for JSON serialization
            metrics_display = {k: float(v) if v is not None else None 
                              for k, v in metrics.items()}
            print(f"    [{name}] {json.dumps(metrics_display, indent=2)}")
            fold_results[name] = metrics
            
        except Exception as e:
            print(f"    [{name}] FAILED with error: {type(e).__name__}: {str(e)}")
            traceback.print_exc()
            # Simple error record. Downstream scripts (autorank, aggregate) will see
            # missing metric columns as NaN (pandas fills them automatically in parquet).
            fold_results[name] = {
                "error": str(e),
                "error_type": type(e).__name__,
                "train_time": None,
            }
            
            # Still try to clear memory even on error
            try:
                gc.collect()
            except:
                pass
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass

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
