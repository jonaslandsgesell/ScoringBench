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

    # Quantile levels and output grid resolution
    _ALPHAS = np.linspace(0.005, 0.995, 200).tolist()   # 200 quantiles
    _N_GRID = len(_ALPHAS)                                         # regular z-grid bins per sample

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

        # 1. Enforce monotonicity by sorting
        q = np.sort(q, axis=1)

        n_samples = q.shape[0]
        alphas = np.array(self._ALPHAS, dtype=float)
        # Extend with boundary CDF values (0 at left tail, 1 at right tail)
        alphas_ext = np.concatenate([[0.0], alphas, [1.0]])

        n_grid = self._N_GRID
        all_bin_edges = np.empty((n_samples, n_grid + 1), dtype=np.float32)
        all_probas    = np.empty((n_samples, n_grid),     dtype=np.float32)

        for i in range(n_samples):
            qi = q[i]

            # Tail extension: use neighbouring inter-quantile gap
            left_w  = max(qi[1]  - qi[0],  1e-6)
            right_w = max(qi[-1] - qi[-2], 1e-6)
            z_min = qi[0]  - left_w
            z_max = qi[-1] + right_w

            # 2. Regular per-sample z-grid
            z_edges = np.linspace(z_min, z_max, n_grid + 1)

            # Anchor quantiles with boundary values
            q_ext = np.concatenate([[z_min], qi, [z_max]])

            # 3. Interpolate CDF at bin edges
            cdf_at_edges = np.interp(z_edges, q_ext, alphas_ext)

            # 4. Density = dCDF/dz; clamp ≥ 0 and convert to masses
            bin_widths = np.diff(z_edges)
            masses = np.diff(cdf_at_edges)          # = density * dz
            masses = np.maximum(masses, 0.0)        # 5. Clamp non-negative

            # 6. Renormalize
            total = masses.sum()
            if total > 0:
                masses /= total

            all_bin_edges[i] = z_edges.astype(np.float32)
            all_probas[i]    = masses.astype(np.float32)

        bin_midpoints = (all_bin_edges[:, :-1] + all_bin_edges[:, 1:]) / 2
        mean = (all_probas * bin_midpoints).sum(axis=-1)

        return DistributionPrediction(
            probas=all_probas,
            bin_edges=all_bin_edges,
            bin_midpoints=bin_midpoints,
            mean=mean,
        )
