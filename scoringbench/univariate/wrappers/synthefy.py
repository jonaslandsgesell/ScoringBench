"""Synthefy Nori wrapper for ScoringBench.

Nori (https://github.com/Synthefy/synthefy-nori) is a tabular foundation model
for regression via in-context learning. The released checkpoint has a native
999-quantile pinball head, so the full predictive distribution is available
through the public API (``output_type="full"``).

ICL semantics: ``fit()`` just stores the labeled context rows; the frozen model
runs in a single forward pass at ``predict``/``predict_distribution`` time.
``predict_distribution()`` reads the model's quantile bank and reported levels,
then uses the shared atom-preserving quantile adapter. Native scoring retains
the reported levels and endpoint tail masses; density scoring uses the separate
linearly resampled view.

Requires ``synthefy-nori`` with the ``output_type="full"`` quantile API
(>= the release that adds probabilistic output; install from source if needed).
"""
from __future__ import annotations

import numpy as np

from .base import DistributionPrediction, ProbabilisticWrapper
from .quantile_based import quantiles_to_distribution


def _to_numeric(X):
    """Coerce a DataFrame / array to a finite float32 matrix (factorize strings)."""
    import pandas as pd
    if isinstance(X, pd.DataFrame):
        X = X.copy()
        for c in X.columns:
            if X[c].dtype == object or str(X[c].dtype).startswith("category"):
                X[c] = pd.factorize(X[c])[0]
        X = X.apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    X = np.asarray(X, dtype=np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


class SynthefyWrapper(ProbabilisticWrapper):
    """Wraps ``synthefy_nori.NoriRegressor`` with the DistributionPrediction API."""

    # Cap the in-context rows for speed/memory (0 = use the full training set).
    CTX_CAP = 0

    def __init__(self, device=None, model_path=None, augmentations=(), **kwargs):
        import torch
        from synthefy_nori import NoriRegressor

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # augmentations default to () — the Yeo-Johnson point ensemble has no
        # distribution-level analogue and is bypassed on the quantile path
        # anyway; keeping it off makes predict() and predict_distribution()
        # consistent.
        self._model = NoriRegressor(
            model_path=model_path,
            device=device,
            augmentations=tuple(augmentations),
            **kwargs,
        )

    def fit(self, X, y) -> "SynthefyWrapper":
        Xn = _to_numeric(X)
        yn = np.asarray(y, dtype=np.float64).reshape(-1)
        if self.CTX_CAP and len(yn) > self.CTX_CAP:
            keep = np.random.default_rng(0).choice(len(yn), self.CTX_CAP, replace=False)
            Xn, yn = Xn[keep], yn[keep]
        self._set_train_range(yn)
        self._model.fit(Xn, yn)
        return self

    def predict(self, X) -> np.ndarray:
        return np.asarray(
            self._model.predict(_to_numeric(X)), dtype=np.float64
        ).reshape(-1)

    def predict_distribution(self, X) -> DistributionPrediction:
        full = self._model.predict(_to_numeric(X), output_type="full")
        if "taus" not in full:
            raise ValueError(
                "Nori output_type='full' must include native quantile levels "
                "in 'taus'; update synthefy-nori instead of assuming levels."
            )
        return quantiles_to_distribution(
            np.asarray(full["quantiles"], dtype=np.float64),
            np.asarray(full["taus"], dtype=np.float64),
            mean=np.asarray(full["mean"], dtype=np.float64),
            train_range=self._y_train_range,
        )
