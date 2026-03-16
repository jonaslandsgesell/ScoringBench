"""TabICL wrapper for ScoringBench."""

from __future__ import annotations

import numpy as np

from .base import DistributionPrediction, ProbabilisticWrapper


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

    # Intended quantile grid for the histogram reconstruction
    num_quantiles =200
    _ALPHAS = np.linspace(0.01, 0.99, num_quantiles).tolist()

    def __init__(self, **kwargs):
        from tabicl import TabICLRegressor
        self._model = TabICLRegressor(**kwargs)

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

        n_samples = q.shape[0]
        alphas = np.array(self._ALPHAS, dtype=float)

        # Build per-sample bin edges: left tail + n_alphas quantiles + right tail
        # Left / right tail width = neighboring inter-quantile distance
        left_w  = np.maximum(q[:, 1]  - q[:, 0],  1e-6)   # (n_samples,)
        right_w = np.maximum(q[:, -1] - q[:, -2], 1e-6)
        bin_edges = np.concatenate(
            [(q[:, 0] - left_w)[:, None], q, (q[:, -1] + right_w)[:, None]],
            axis=1,
        )  # (n_samples, n_alphas + 1) = (n_samples, n_bins + 1)

        # Mass per bin: left_tail mass = alphas[0], inter-quantile = diff(alphas),
        # right_tail mass = 1 - alphas[-1]
        masses = np.concatenate([[alphas[0]], np.diff(alphas), [1.0 - alphas[-1]]])
        probas = np.broadcast_to(masses[None, :], (n_samples, len(masses))).copy()

        bin_midpoints = (bin_edges[:, :-1] + bin_edges[:, 1:]) / 2   # (n_samples, n_bins)
        mean = np.sum(probas * bin_midpoints, axis=-1)                  # (n_samples,)

        return DistributionPrediction(
            probas=probas,
            bin_edges=bin_edges,
            bin_midpoints=bin_midpoints,
            mean=mean,
        )
