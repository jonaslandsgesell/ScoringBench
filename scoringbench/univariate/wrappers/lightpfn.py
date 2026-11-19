"""LightPFN quantile-regression wrapper for ScoringBench."""

from __future__ import annotations

import numpy as np

from .base import DistributionPrediction, ProbabilisticWrapper
from .quantile_based import quantiles_to_distribution


_QUANTILE_LEVELS = np.linspace(0.001, 0.999, 999)


class LightPFNWrapper(ProbabilisticWrapper):
    """Wrap LightPFN 1.5's pretrained quantile regressor.

    LightPFN predicts a native grid of 999 quantiles (.001 through .999) from
    synthetic-only pretraining. See https://github.com/GioOtto/LightPFN.
    Weights and code are Apache-2.0 licensed; report: *LightPFN 1.5: Native
    categoricals, quantile regression, wide tables, and fast portable inference
    under 5 million parameters* by Giorgio Ottoboni.
    """

    def __init__(self, device: str = "auto"):
        try:
            from lightpfn import LightPFNRegressor
        except ImportError as exc:
            raise ImportError(
                "Failed to import LightPFN. Install it with `pip install LightPFN`."
            ) from exc

        self._model = LightPFNRegressor(device=device)

    def fit(self, X, y) -> "LightPFNWrapper":
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
        if len(X) != y_arr.size:
            raise ValueError(
                f"X has {len(X)} rows but y has {y_arr.size} values"
            )

        finite = np.isfinite(y_arr)
        if not finite.any():
            raise ValueError("LightPFN requires at least one finite training target")
        if not finite.all():
            row_indices = np.flatnonzero(finite)
            X = X.iloc[row_indices] if hasattr(X, "iloc") else np.asarray(X)[finite]
            y_arr = y_arr[finite]

        self._set_train_range(y_arr)
        self._model.fit(X, y_arr, cat="auto")
        return self

    def predict(self, X) -> np.ndarray:
        return np.asarray(self._model.predict(X), dtype=np.float64).reshape(-1)

    def predict_distribution(self, X) -> DistributionPrediction:
        q = np.asarray(self._model.predict_quantiles(X), dtype=np.float64)
        if q.ndim == 1:
            q = q[np.newaxis, :]
        if q.ndim != 2 or q.shape[1] != _QUANTILE_LEVELS.size:
            raise ValueError(
                "LightPFN predict_quantiles() must return an (n_samples, 999) "
                f"array; got shape {q.shape}"
            )

        # Match LightPFN's mean: trapezoidal integration over its native grid,
        # with the two endpoint tail masses assigned to the outer quantiles.
        mean = (
            0.001 * q[:, 0]
            + np.trapezoid(q, dx=0.001, axis=1)
            + 0.001 * q[:, -1]
        )
        return quantiles_to_distribution(
            q,
            _QUANTILE_LEVELS,
            mean=mean,
            train_range=self._y_train_range,
        )
