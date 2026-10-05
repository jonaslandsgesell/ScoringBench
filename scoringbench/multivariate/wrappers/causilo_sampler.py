"""Per-dimension conditional sampler backed by the Causilo quantile model."""

from __future__ import annotations

import numpy as np

from ...univariate.wrappers.causilo import CausiloWrapper
from .base_sampler import BaseSampler


class CausiloSampler(BaseSampler):
    """Adapt Causilo's conditional quantiles to the multivariate sampler API."""

    def __init__(self, n_estimators=8, random_state=42, device="auto"):
        self._model = CausiloWrapper(
            n_estimators=n_estimators,
            random_state=random_state,
            device=device,
        )
        self._device = None if device == "auto" else device

    def fit(self, X, y) -> "CausiloSampler":
        self._model.fit(X, y)
        return self

    def predict_mean(self, X) -> np.ndarray:
        return np.asarray(self._model.predict(X), dtype=np.float64).ravel()

    def _row_cdf_grid(self, X):
        nodes = self._model.predict_distribution(X).cdf_nodes
        if not isinstance(nodes, tuple) or len(nodes) != 2:
            raise RuntimeError("Causilo prediction did not provide quantile CDF nodes")
        support, cdf = nodes
        return cdf, support
