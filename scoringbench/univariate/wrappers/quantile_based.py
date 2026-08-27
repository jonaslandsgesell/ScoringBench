"""Quantile -> DistributionPrediction conversion for ScoringBench wrappers.

Thin adapter over :meth:`DistributionPrediction.from_multi_quantile`: it only
sanitizes the caller's raw quantile matrix (monotone-sort, clip probability
levels, replace non-finite values, widen a lone level), then hands the cleaned
grid to the constructor.  The quantile function ``alpha -> q(alpha)`` is read as
a CDF whose nodes become native bin edges directly, so tied quantiles survive as
atoms; see :class:`DistributionPrediction`.
"""

from __future__ import annotations

import numpy as np

from .base import DistributionPrediction


def quantiles_to_distribution(
    q: np.ndarray,
    alphas: np.ndarray,
    mean: np.ndarray | None = None,
    y_range: tuple[float, float] | None = None,
    *,
    train_range: tuple[float, float],
) -> DistributionPrediction:
    """Sanitize a quantile matrix and build an atom-preserving prediction.

    Parameters
    ----------
    q : (n_samples, K) array
        Per-sample quantile values at the probability levels ``alphas``.
    alphas : (K,) array
        Probability levels in (0, 1). Sorted / clipped defensively.
    mean : (n_samples,) array, optional
        Point prediction to report. If ``None`` the PMF mean is used.
    y_range : (lo, hi), optional
        Fallback finite range used to sanitize non-finite quantiles.
    train_range : (y_lo, y_hi), required
        The shared train-target range the density rules regrid onto.
    """
    q = np.asarray(q, dtype=np.float64)
    if q.ndim == 1:
        q = q[np.newaxis, :]
    alphas = np.asarray(alphas, dtype=np.float64).reshape(-1)
    if q.shape[1] != alphas.size:
        raise ValueError(
            f"q has {q.shape[1]} columns but {alphas.size} alphas were given"
        )

    # Sort/clamp levels into [0, 1] so the implied masses are non-negative.
    alphas = np.clip(np.sort(alphas), 0.0, 1.0)
    lo, hi = (float(y_range[0]), float(y_range[1])) if y_range is not None else (0.0, 1.0)
    if not np.all(np.isfinite(q)):
        q = np.nan_to_num(q, nan=lo, posinf=hi, neginf=lo)
    q = np.sort(q, axis=1)  # enforce monotone quantiles per sample

    # A single level is a point mass: keep it a HONEST zero-width atom (two
    # coincident abscissae at the same value) rather than inventing an interval
    # of arbitrary width.  It scores exactly as a Dirac on the native PMF grid, and
    # the density grid still gets positive width from ``train_range`` (``y_hi >
    # y_lo`` is enforced in ``DistributionPrediction.__post_init__``); a truly
    # constant target (``y_hi == y_lo``) is a genuine atom whose density is not
    # reportable and is rejected there.
    if q.shape[1] < 2:
        q = np.concatenate([q, q], axis=1)
        alphas = np.array([0.0, 1.0])

    return DistributionPrediction.from_multi_quantile(
        q, alphas, mean=mean, train_range=train_range
    )
