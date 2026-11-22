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

from .metrics import (
    compute_metrics, compute_point_metrics,
    ENERGY_BETAS, DPD_BETAS, CRTS_ALPHAS, DENSITY_BASED_KEYS, PIT_KEYS,
)
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
    score_density: bool = True,
) -> dict:
    """Fit and evaluate every model on one fold.

    ``score_density`` (a dataset-level property computed once on the full target;
    see :mod:`scoringbench.univariate.discreteness`) gates the density-based
    scoring rules.  When ``False`` the target is discrete / lattice-valued, the
    DENSITY_BASED_KEYS are not proper and the PIT KS diagnostic (PIT_KEYS) is
    invalid (its continuity null false-rejects even a perfect forecast), so both
    families are emitted as ``None`` for every model; downstream ranking drops
    the dataset from those leaderboards only (``autorank`` ``dropna(how='any')``
    per metric) while keeping it in the grid-robust ones.  Each model's metrics
    also carry ``discrete_y`` for transparency in the raw parquet and the
    flattened result rows.

    Each factory produces a fresh ProbabilisticWrapper.
    If predict_distribution() is not implemented, falls back to point metrics
    only (distributional metrics are set to None).
    
    Imputation is performed within this fold:
    - fit_transform on X_train only (learn statistics from train data)
    - transform on X_test (apply train statistics)
    This ensures no data leakage from test to train via imputation statistics.

    Timing keys
    -----------
    ``train_time`` is the END-TO-END cost, ``fit_time + predict_time``. It is not
    a fit-only stopwatch, because in-context learners (e.g. Mitra-2) do their
    training inside ``predict``: a fit-only number reports them as effectively
    free and makes them incomparable to eager models. ``fit_time`` and
    ``predict_time`` are reported alongside so the split stays recoverable.

    Returns {model_name: {mae, rmse, r2, crps, sharpness,
                          coverage_90, interval_score_90,
                          coverage_95, interval_score_95,
                          crts_alpha_{1.01,...,2.0},
                          fit_time, predict_time, train_time}}
    """
    X_train, X_test = X_train.copy(), X_test.copy()
    categorical_cols = X_train.select_dtypes(exclude=[np.number]).columns
    # load_dataset already yields ``category`` (levels from the full dataset);
    # this only guards direct callers passing object columns, giving train and
    # test one shared level set so their codes agree.
    for col in categorical_cols:
        if not isinstance(X_train[col].dtype, pd.CategoricalDtype):
            levels = pd.concat([X_train[col], X_test[col]]).dropna().unique()
            dtype = pd.CategoricalDtype(sorted(levels, key=str))
            X_train[col] = X_train[col].astype(dtype)
            X_test[col] = X_test[col].astype(dtype)

    # Impute missing values (learn from train, apply to both train and test)
    if X_train.isna().sum().sum() > 0 or X_test.isna().sum().sum() > 0:
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) > 0:
            numeric_imputer = SimpleImputer(strategy='median')
            numeric_imputer.fit(X_train[numeric_cols])
            X_train[numeric_cols] = numeric_imputer.transform(X_train[numeric_cols])
            X_test[numeric_cols] = numeric_imputer.transform(X_test[numeric_cols])

        # Mode imputation done in pandas so the ``category`` dtype survives.
        for col in categorical_cols:
            mode = X_train[col].mode(dropna=True)
            if len(mode) > 0:
                X_train[col] = X_train[col].fillna(mode.iloc[0])
                X_test[col] = X_test[col].fillna(mode.iloc[0])

    # Integer-coded view for wrappers without native categorical support.
    # Categories were fixed on the full dataset, so codes agree across folds.
    if len(categorical_cols) > 0:
        X_train_codes, X_test_codes = X_train.copy(), X_test.copy()
        for col in categorical_cols:
            X_train_codes[col] = X_train[col].cat.codes
            X_test_codes[col] = X_test[col].cat.codes
    else:
        X_train_codes, X_test_codes = X_train, X_test

    y_test_np = np.asarray(y_test, dtype=float)
    fold_results: dict[str, dict] = {}

    for name, factory in model_factories.items():
        print(f"    [{name}] fitting …", flush=True)
        try:
            model: ProbabilisticWrapper = factory()
            native_cat = getattr(model, "supports_categorical", False)
            Xtr = X_train if native_cat else X_train_codes
            Xte = X_test if native_cat else X_test_codes


            # Timing is split by PHASE and reported as both parts plus a total
            #
            # perf_counter is monotonic: it measures elapsed wall time and is
            # immune to system-clock adjustments, unlike time.time(). The
            # factory() construction above is deliberately outside the window.
            t0 = time.perf_counter()
            model.fit(Xtr, y_train)
            fit_time = time.perf_counter() - t0

            t1 = time.perf_counter()
            try:
                dist = model.predict_distribution(Xte)
                predict_time = time.perf_counter() - t1
                metrics = compute_metrics(dist, y_test_np, score_density=score_density)
            except NotImplementedError:
                # predict_distribution bailed out; the fallback point predict is
                # the real inference work, so time THAT (the aborted call above
                # raised before doing anything, so nothing is double-counted).
                t1 = time.perf_counter()
                y_pred = model.predict(Xte)
                predict_time = time.perf_counter() - t1
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

            # Discrete / lattice target: density rules are not proper -> emit
            # None uniformly so the parquet column exists and downstream ranking
            # drops this dataset from the density leaderboards only.
            if not score_density:
                for key in (*DENSITY_BASED_KEYS, *PIT_KEYS):
                    metrics[key] = None
            metrics["discrete_y"] = not score_density

            metrics["fit_time"] = fit_time
            metrics["predict_time"] = predict_time
            metrics["train_time"] = fit_time + predict_time

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
                "discrete_y": not score_density,
                "train_time": None,
                "fit_time": None,
                "predict_time": None,
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
