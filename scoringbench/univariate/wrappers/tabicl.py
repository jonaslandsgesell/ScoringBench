"""TabICL wrapper for ScoringBench."""

from __future__ import annotations

import numpy as np

from .base import DistributionPrediction, ProbabilisticWrapper
from .quantile_based import quantiles_to_distribution

import sys
from pathlib import Path

# Prefer local checkout of `tabicl` when present in the workspace.
# Compute repository root relative to this file and insert the local
# `tabicl/src` directory at the front of `sys.path` if it exists.
repo_root = Path(__file__).resolve().parents[2]
local_tabicl = repo_root / "tabicl" / "src"
if local_tabicl.exists():
    sys.path.insert(0, str(local_tabicl))


class TabICLWrapper(ProbabilisticWrapper):
    """Wraps TabICLRegressor (v2).

    predict() works out of the box (uses output_type='mean').
    predict_distribution() is TODO — the plan is to call
        predict(X, output_type='quantiles', alphas=...)
    and convert the per-sample quantile values into a piecewise-uniform
    histogram (DistributionPrediction with 2-D bin_edges).
    Until that conversion is implemented this raises NotImplementedError,
    and cv.py will run point metrics only.
    """

    # Quantile levels and output grid resolution
    
    

    def __init__(self, **kwargs):
        from tabicl import TabICLRegressor
        self._model = TabICLRegressor(**kwargs)
        self._ALPHAS = np.linspace(0.005, 0.995, 200).tolist()   # 200 quantiles

    def fit(self, X, y) -> "TabICLWrapper":
        self._model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return np.asarray(self._model.predict(X, output_type="mean"))

    def predict_distribution(self, X) -> DistributionPrediction:
        X_arr = np.asarray(X.values if hasattr(X, "values") else X)
        raw_q = self._model.predict(X_arr, output_type="quantiles", alphas=self._ALPHAS)

        # Robustly convert to (n_samples, n_alphas) numpy array
        if isinstance(raw_q, dict):
            q_arr = list(raw_q.values())[0]
        else:
            q_arr = raw_q

        if isinstance(q_arr, list):
            q = np.vstack([np.asarray(r).ravel() for r in q_arr])
        else:
            q = np.asarray(q_arr, dtype=float)

        if q.ndim == 1:
            q = q[np.newaxis, :]
        if q.shape[1] != len(self._ALPHAS) and q.shape[0] == len(self._ALPHAS):
            q = q.T

        # 1. Enforce monotonicity by sorting
        q = np.sort(q, axis=1)

        # 2. Convert to a DistributionPrediction using the *shared* quantile ->
        #    distribution mapping (regular per-sample z-grid, CDF interpolated
        #    at the grid edges), exactly like the other quantile-based wrappers
        #    (CatBoost / XGB-quantile / NGBoost), so every quantile model is
        #    discretized identically.
        alphas = np.asarray(self._ALPHAS, dtype=float)
        return quantiles_to_distribution(q, alphas)

class FinetuneTabICLWrapper(TabICLWrapper):
    """Wraps a finetuned TabICL regressor.
    
    Inherits from TabICLWrapper to reuse the distribution prediction logic.
    """

    def __init__(
        self,
        *,
        epochs: int = 80,
        learning_rate: float = 1e-5,
        n_estimators_finetune: int = 2,
        n_estimators_validation: int = 2,
        n_estimators_inference: int = 8,
        early_stopping: bool = True,
        patience: int = 8,
        eval_metric: str | None = None,
        random_state: int = 0,
        verbose: bool = False,
        max_data_size: int = 10_000,
        **kwargs,
    ):
        # Set the distribution-grid attributes directly instead of calling
        # super().__init__(), which would build a throwaway TabICLRegressor and
        # reject finetune-only kwargs such as ``max_data_size``.
        self._ALPHAS = np.linspace(0.005, 0.995, 200).tolist()   # 200 quantiles

        from tabicl import FinetunedTabICLRegressor

        # Replace self._model with the finetuned variant
        self._model = FinetunedTabICLRegressor(
            epochs=epochs,
            learning_rate=learning_rate,
            n_estimators_finetune=n_estimators_finetune,
            n_estimators_validation=n_estimators_validation,
            n_estimators_inference=n_estimators_inference,
            early_stopping=early_stopping,
            patience=patience,
            eval_metric=eval_metric,
            random_state=random_state,
            verbose=verbose,
            max_data_size=max_data_size,
            **kwargs,
        )