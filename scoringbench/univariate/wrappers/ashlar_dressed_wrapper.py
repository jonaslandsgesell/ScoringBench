"""Ashlar's dressed model (``ashlar.dressed()``), read through its quantiles."""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from .base import DistributionPrediction, ProbabilisticWrapper
from .quantile_based import quantiles_to_distribution

# Quantile levels read from each row's predictive distribution: a fine grid through the body, plus the far tails.
LEVELS = np.concatenate([[0.001, 0.0025], np.linspace(0.005, 0.995, 199), [0.9975, 0.999]])
# The mass each quantile carries when the 203 of them stand in for a density; the mean is their weighted sum.
LEVEL_MASS = np.diff(np.concatenate([[0.0], LEVELS]))
LEVEL_MASS[-1] += 1.0 - LEVELS[-1]

# JAX takes most of a GPU's memory at its first use unless told otherwise, and ScoringBench scores with torch on
# the same card. This asks it to take memory as it needs it, unless the variable is already set.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_device_checked = False


def _check_device():
    """Warn, once, when a GPU is there and JAX cannot use it: Ashlar then trains on the CPU, several times slower."""
    global _device_checked
    if _device_checked:
        return
    _device_checked = True
    import jax
    import torch

    if jax.default_backend() == "cpu" and torch.cuda.is_available():
        warnings.warn(
            "Ashlar is training on the CPU although a GPU is available, because this JAX has no CUDA build. "
            'pip install "jax[cuda12]" (or "jax[cuda13]", to match the machine) lets it use the GPU.',
            stacklevel=3,
        )


class AshlarDressedWrapper(ProbabilisticWrapper):
    """Ashlar's dressed model: networks and label readers trained from scratch, pooled into one distribution.

    Five cross-fitted pairs of a network (a TabM-style ensemble with a flow head) and a label reader, which
    reads a row's distribution from the training labels. The pairs are pooled, the readers' points are smoothed
    into a density, and one more network corrects the pool; the correction follows condensité's PIT boosting.
    Nothing is pretrained and nothing is tuned: the entry runs ``ashlar.dressed()`` with its defaults. The one
    change to its input is that numeric columns are shifted to a training minimum of 0, which works around a
    float32 NaN in Ashlar's quantile feature scaler (see ``_fit_shift``) and leaves the quantile-scaled features
    unchanged.

    The prediction has an exact CDF, and its quantiles at 203 levels go to ScoringBench. A target with at most
    20 distinct training values gets a probability for each value.

    The networks train on the GPU where JAX has its CUDA build, and on the CPU otherwise. On a Tesla T4 a fold of
    2,400 training rows takes 12 to 30 minutes to fit and 2 to 9 minutes to predict, for 8 to 1,024 columns.

    Code: https://github.com/maxmbay/ashlar (Apache-2.0), ``pip install ashlar-tabular``.
    Guide: https://github.com/maxmbay/ashlar/blob/main/docs/dressed.md

    Keyword arguments go to ``ashlar.dressed``.
    """

    supports_categorical = True  # non-numeric DataFrame columns get categorical embeddings

    def __init__(self, **settings):
        self.settings = settings
        self.model_ = None
        self._last = None
        self._shift = None

    # Workaround for a float32 edge case in Ashlar's quantile feature scaler (ashlar.preprocess._quantile_normal,
    # 0.1.2-0.1.3). A training value equal to a column's lowest knot is meant to be caught by
    # ``values - tol < lower_x`` with ``tol = max(1e-7, spacing(lower_x))``; for a negative minimum ``spacing`` is
    # negative, so tol falls back to 1e-7, which float32 cannot subtract from |x| >~ 1. The value then goes through
    # ``interp``, whose rounding can leave it a hair below 0, and ``norm.ppf`` of that is NaN, so the first training
    # loss is not finite (Yolanda, year, and intermittent folds elsewhere). Shifting every numeric column so its
    # training minimum is exactly 0 makes the 1e-7 halo work again. A quantile transform ignores a constant shift,
    # so the model sees the same inputs up to float rounding.
    def _numeric_columns(self, X):
        if hasattr(X, "select_dtypes"):
            return [c for c in X.select_dtypes(include=[np.number]).columns if not pd.api.types.is_bool_dtype(X[c])]
        return None

    def _fit_shift(self, X):
        cols = self._numeric_columns(X)
        if cols is None:
            mins = np.nanmin(np.asarray(X, dtype=np.float64), axis=0)
        else:
            mins = X[cols].astype(np.float64).min(axis=0, skipna=True).to_numpy()
        self._shift = (cols, np.where(np.isfinite(mins), mins, 0.0))

    def _apply_shift(self, X):
        cols, mins = self._shift
        if cols is None:
            return np.asarray(X, dtype=np.float64) - mins
        X = X.copy()
        X[cols] = X[cols].astype(np.float64) - mins
        return X

    def fit(self, X, y) -> "AshlarDressedWrapper":
        _check_device()
        import ashlar

        y = np.asarray(y, dtype=np.float64).reshape(-1)
        finite = np.isfinite(y)
        if not finite.all():
            X = X.iloc[finite] if hasattr(X, "iloc") else np.asarray(X)[finite]
            y = y[finite]
        self._set_train_range(y)
        self._last = None
        self._fit_shift(X)
        self.model_ = ashlar.dressed(**self.settings).fit(self._apply_shift(X), y)
        return self

    def _read(self, X):
        """Each row's quantiles at LEVELS and its mean, kept for a second call on the same rows."""
        if self._last is not None and self._last[0] is X:
            return self._last[1], self._last[2]
        dist = self.model_.predict_dist(self._apply_shift(X))
        q = dist.quantile(LEVELS).T
        # A target with few values has an exact mean; a density's is read from its quantiles, as the model reads it.
        mean = q @ LEVEL_MASS if dist.values is None else dist.probabilities() @ dist.values
        self._last = (X, q, mean)
        return q, mean

    def predict(self, X) -> np.ndarray:
        return self._read(X)[1]

    def predict_distribution(self, X) -> DistributionPrediction:
        q, mean = self._read(X)
        return quantiles_to_distribution(
            q, LEVELS, mean=mean, y_range=self._y_train_range, train_range=self._y_train_range
        )
