"""Causilo quantile regression adapter."""

import numpy as np

from .base import DistributionPrediction, ProbabilisticWrapper
from .quantile_based import quantiles_to_distribution


class CausiloWrapper(ProbabilisticWrapper):
    """Pretrained Causilo regression with exponential quantile tails.

    Code: https://github.com/nums-ai/causilo (Apache-2.0).
    Weights: https://huggingface.co/nums-ai/causilo (Causilo License v1.0).
    Developed by Nums AI Inc.; the technical report is forthcoming.

    n_estimators controls ensemble size, random_state fixes its permutations,
    and device selects CPU or CUDA (auto prefers CUDA).
    """

    def __init__(self, n_estimators=8, random_state=42, device="auto"):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.device = device
        self._levels = np.r_[0.0001, np.arange(1, 1000) / 1000, 0.9999]

    def fit(self, X, y):
        from causilo import CausiloRegressor

        y = np.asarray(y, dtype=np.float64).reshape(-1)
        valid = np.isfinite(y)
        if not valid.all():
            X = X.iloc[valid] if hasattr(X, "iloc") else np.asarray(X)[valid]
            y = y[valid]
        if not y.size:
            raise ValueError("Causilo requires finite training targets")
        self._set_train_range(y)
        self.model_ = CausiloRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            device=self.device,
        ).fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model_.predict(X)

    def predict_distribution(self, X) -> DistributionPrediction:
        quantiles = self.model_.predict(X, output_type="quantiles", quantiles=self._levels)
        return quantiles_to_distribution(quantiles, self._levels, train_range=self._y_train_range)
