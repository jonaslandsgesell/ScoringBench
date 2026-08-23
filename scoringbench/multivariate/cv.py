"""Cross-validation loop for the multivariate benchmark.

Public API
----------
run_fold(X_train, X_test, Y_train, Y_test, model_factories, seed) -> dict
    Fit and evaluate every model on one pre-split fold.
    Returns {model_name: {metric: value, ...}, ...}
"""

import gc
import json
import time
import traceback
from typing import Callable

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer

from .metrics import SCORING_RULE_KEYS, compute_metrics, compute_point_metrics
from .wrappers import MultivariateWrapper


def _impute_inplace(X_train: pd.DataFrame, X_test: pd.DataFrame) -> None:
    """Impute missing feature values (fit on train, apply to both)."""
    if X_train.isna().sum().sum() == 0 and X_test.isna().sum().sum() == 0:
        return
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    categorical_cols = X_train.select_dtypes(exclude=[np.number]).columns

    if len(numeric_cols) > 0:
        imp = SimpleImputer(strategy="median")
        imp.fit(X_train[numeric_cols])
        X_train[numeric_cols] = imp.transform(X_train[numeric_cols])
        X_test[numeric_cols] = imp.transform(X_test[numeric_cols])

    if len(categorical_cols) > 0:
        imp = SimpleImputer(strategy="most_frequent")
        imp.fit(X_train[categorical_cols])
        X_train[categorical_cols] = imp.transform(X_train[categorical_cols])
        X_test[categorical_cols] = imp.transform(X_test[categorical_cols])


def run_fold(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    Y_train: pd.DataFrame,
    Y_test: pd.DataFrame,
    model_factories: dict[str, Callable],
    seed: int,
) -> dict:
    """Fit and evaluate every model on one fold.

    Each factory returns a fresh :class:`MultivariateWrapper`. Models emit a
    ``MultivariateSamplePrediction`` via ``predict_ensemble``; scoring is done
    on those draws directly. If ``predict_ensemble`` is not implemented, we fall
    back to point metrics only (scoring-rule columns are set to None).
    """
    _impute_inplace(X_train, X_test)

    y_test_np = np.asarray(Y_test, dtype=np.float64)
    if y_test_np.ndim == 1:
        y_test_np = y_test_np[:, None]

    fold_results: dict[str, dict] = {}

    for name, factory in model_factories.items():
        print(f"    [{name}] fitting …", flush=True)
        try:
            model: MultivariateWrapper = factory()

            # perf_counter is monotonic: it measures elapsed wall time for the
            # fit only (factory() construction above is excluded) and is immune
            # to system-clock adjustments, unlike time.time().
            t0 = time.perf_counter()
            model.fit(X_train, Y_train)
            elapsed = time.perf_counter() - t0

            try:
                pred = model.predict_ensemble(X_test)
                metrics = compute_metrics(pred, y_test_np)
            except NotImplementedError:
                y_pred = np.asarray(model.predict(X_test), dtype=np.float64)
                metrics = compute_point_metrics(y_test_np, y_pred)
                for key in SCORING_RULE_KEYS:
                    metrics[key] = None

            metrics["train_time"] = elapsed

            del model
            gc.collect()
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            metrics_display = {
                k: (float(v) if v is not None else None) for k, v in metrics.items()
            }
            print(f"    [{name}] {json.dumps(metrics_display, indent=2)}")
            fold_results[name] = metrics

        except Exception as e:
            print(f"    [{name}] FAILED with error: {type(e).__name__}: {str(e)}")
            traceback.print_exc()
            fold_results[name] = {
                "error": str(e),
                "error_type": type(e).__name__,
                "train_time": None,
            }
            try:
                gc.collect()
            except Exception:
                pass
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    return fold_results
