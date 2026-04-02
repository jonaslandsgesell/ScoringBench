"""Base classes for ScoringBench probabilistic model wrappers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------

@dataclass
class DistributionPrediction:
    """Unified probabilistic prediction container.

    bin_edges / bin_midpoints may be 1-D (shared grid, same for every sample)
    or 2-D (per-sample grid, e.g. when derived from per-sample quantiles).
    metrics.py handles both cases.
    """
    probas: np.ndarray         # (n_samples, n_bins)  — PMF: mass per bin, sums to 1
    bin_edges: np.ndarray      # (n_bins+1,) or (n_samples, n_bins+1)
    bin_midpoints: np.ndarray  # (n_bins,)   or (n_samples, n_bins)
    mean: np.ndarray           # (n_samples,)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class ProbabilisticWrapper:
    """Base class for ScoringBench model wrappers.

    Subclass and implement fit(), predict(), predict_distribution().
    If predict_distribution() is not yet supported, leave it raising
    NotImplementedError — cv.py will skip distributional metrics gracefully.
    """

    def fit(self, X, y) -> "ProbabilisticWrapper":
        raise NotImplementedError

    def predict(self, X) -> np.ndarray:
        raise NotImplementedError

    def predict_distribution(self, X) -> DistributionPrediction:
        raise NotImplementedError
