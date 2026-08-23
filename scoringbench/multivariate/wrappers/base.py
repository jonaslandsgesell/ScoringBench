"""Base class for multivariate ScoringBench model wrappers.

A multivariate wrapper is deliberately tiny compared with the univariate one:
because the forecast is *purely sample based* (see
:class:`scoringbench.multivariate.prediction.MultivariateSamplePrediction`),
there is no grid/PMF/CDF machinery to inherit.  A wrapper only has to

1. ``fit(X, Y)`` on a ``(n, d)`` target matrix, and
2. ``predict_ensemble(X) -> MultivariateSamplePrediction``.

The point prediction ``predict(X)`` defaults to the per-instance ensemble mean,
so subclasses that already produce draws get it for free.
"""

from __future__ import annotations

import numpy as np

from ..prediction import MultivariateSamplePrediction


class MultivariateWrapper:
    """Abstract base for a d-dimensional probabilistic regressor.

    Subclasses implement :meth:`fit` and :meth:`predict_ensemble`.  The base
    class provides a mean-based :meth:`predict`.
    """

    def fit(self, X, Y) -> "MultivariateWrapper":
        """Fit on covariates ``X`` and a ``(n, d)`` target matrix ``Y``."""
        raise NotImplementedError

    def predict_ensemble(self, X) -> MultivariateSamplePrediction:
        """Return a :class:`MultivariateSamplePrediction` for the rows of ``X``."""
        raise NotImplementedError

    def predict(self, X) -> np.ndarray:
        """Point prediction: the per-instance ensemble mean ``(n_test, d)``."""
        return self.predict_ensemble(X).mean


def as_2d_targets(Y) -> np.ndarray:
    """Coerce a target container to a ``(n, d)`` float64 array.

    Accepts a 1-D array (treated as ``d = 1``), a 2-D array, or a pandas
    DataFrame / Series.  Centralised here so every wrapper handles the target
    shape identically.
    """
    if hasattr(Y, "values"):
        Y = Y.values
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    if Y.ndim != 2:
        raise ValueError(f"Y must be 1-D or 2-D; got shape {Y.shape}")
    return Y


def as_2d_features(X) -> np.ndarray:
    """Coerce a covariate container to a ``(n, p)`` float64 array."""
    if hasattr(X, "values"):
        X = X.values
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    return X
