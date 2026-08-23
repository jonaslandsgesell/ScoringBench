"""CatBoost MultiQuantile wrapper for ScoringBench."""

from __future__ import annotations

import numpy as np

from .base import DistributionPrediction, ProbabilisticWrapper
from .quantile_based import quantiles_to_distribution


class CatBoostQuantileWrapper(ProbabilisticWrapper):
    """CatBoost MultiQuantile regressor for ScoringBench.

    Trains a single CatBoost model with the ``MultiQuantile`` loss to predict
    quantiles 0.01 through 0.99 simultaneously.  Predicted quantile curves are
    isotonically sorted before being converted to a ``DistributionPrediction``.

    Args:
        n_quantiles: Number of equally-spaced quantile levels in (0, 1).
            Defaults to 99 (levels 0.01, 0.02, …, 0.99).
        iterations: Number of boosting iterations (trees). Defaults to 1000.
        catboost_params: Extra keyword arguments forwarded to
            ``CatBoostRegressor.__init__`` (e.g. ``learning_rate``,
            ``depth``). ``loss_function``, ``iterations``, and
            ``logging_level`` are managed by the wrapper and will be
            ignored here.
    """

    def __init__(
        self,
        n_quantiles: int = 99,
        iterations: int = 1000,
        catboost_params: dict | None = None,
    ):
        self.n_quantiles = n_quantiles
        self.iterations = iterations
        self.catboost_params = catboost_params or {}

        self._alphas = np.linspace(1 / (n_quantiles + 1), n_quantiles / (n_quantiles + 1), n_quantiles)
        # Use the user-requested 0.01..0.99 grid when n_quantiles == 99
        if n_quantiles == 99:
            self._alphas = np.array([q / 100 for q in range(1, 100)])

        self._model = None
        self._y_range: tuple[float, float] = (0.0, 1.0)

    def _build_model(self):
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise ImportError(
                "Failed to import catboost. Install catboost to use this wrapper."
            ) from exc

        quantile_str = ",".join(f"{q:.2f}" for q in self._alphas)
        self._model = CatBoostRegressor(
            iterations=self.iterations,
            loss_function=f"MultiQuantile:alpha={quantile_str}",
            logging_level="Silent",
            **self.catboost_params,
        )

    def fit(self, X, y) -> "CatBoostQuantileWrapper":
        if self._model is None:
            self._build_model()
        y = np.asarray(y, dtype=float)
        self._y_range = (float(y.min()), float(y.max()))
        self._model.fit(X, y)
        return self

    def predict_distribution(self, X) -> DistributionPrediction:
        # q shape: (n_samples, n_quantiles)
        q = np.asarray(self._model.predict(X), dtype=float)
        return quantiles_to_distribution(q, self._alphas, y_range=self._y_range)

    def predict(self, X) -> np.ndarray:
        q = np.asarray(self._model.predict(X), dtype=float)
        if q.ndim == 1:
            return q
        # Point estimate: mean of the quantile particles
        return np.mean(q, axis=1)
