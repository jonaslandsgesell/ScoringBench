"""Quantile -> DistributionPrediction conversion for ScoringBench wrappers.

``quantiles_to_distribution``
    Convert a per-sample quantile matrix ``q`` (n_samples, K) evaluated at
    probability levels ``alphas`` into a :class:`DistributionPrediction`.
    Parametric models (e.g. NGBoost) feed it their analytic ``ppf`` grid;
    models that expose quantile predictions feed those directly.

Discretization
--------------
The quantiles are *not* used as bin edges: tied quantiles give zero-width bins,
so ``p_k / w_k`` becomes 0/0 (>90% of bins on some datasets).  Instead the
quantile function is read as a CDF and resampled onto a regular grid via
``base.cdf_nodes_to_regular_grid`` -- the single scheme every wrapper shares.
``K`` levels yield ``K - 1`` bins that are equally *wide* rather than equally
*probable*: sharpness shows up as mass concentration, and every width is
strictly positive, so the density-based rules apply.
"""

from __future__ import annotations

import numpy as np

from .base import (
    MIN_PAD,
    DistributionPrediction,
    cdf_nodes_to_regular_grid,
)


def quantiles_to_distribution(
    q: np.ndarray,
    alphas: np.ndarray,
    mean: np.ndarray | None = None,
    y_range: tuple[float, float] | None = None,
) -> DistributionPrediction:
    """Build a regularly binned ``DistributionPrediction`` from quantiles.

    The quantile function ``alpha -> q(alpha)`` is read as a CDF and resampled
    onto ``K - 1`` equally *wide* bins via ``base.cdf_nodes_to_regular_grid``;
    equal width keeps every bin positive when quantiles tie.

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

    Returns
    -------
    DistributionPrediction
        ``bin_edges`` is 2-D (n_samples, K), ``probas`` is (n_samples, K - 1).
    """
    q = np.asarray(q, dtype=np.float64)
    if q.ndim == 1:
        q = q[np.newaxis, :]
    alphas = np.asarray(alphas, dtype=np.float64).reshape(-1)

    if q.shape[1] != alphas.size:
        raise ValueError(
            f"q has {q.shape[1]} columns but {alphas.size} alphas were given"
        )

    # Guard the probability levels: sort ascending and clamp into [0, 1] so the
    # implied bin masses are non-negative, even if a caller passes unsorted or
    # out-of-range alphas.
    alphas = np.clip(np.sort(alphas), 0.0, 1.0)

    lo, hi = (float(y_range[0]), float(y_range[1])) if y_range is not None else (0.0, 1.0)
    if not np.all(np.isfinite(q)):
        q = np.nan_to_num(q, nan=lo, posinf=hi, neginf=lo)

    # Enforce monotonic quantiles per sample (also copies, so q is ours to edit).
    q = np.sort(q, axis=1)

    # A single level spans no interval, so it cannot define a bin; give it a
    # nominal one around the predicted value.
    if q.shape[1] < 2:
        pad = np.maximum(np.abs(q[:, :1]) * 0.1, MIN_PAD)
        q = np.concatenate([q - pad, q + pad], axis=1)
        alphas = np.array([0.0, 1.0])

    n_bins = q.shape[1] - 1

    # The quantile function alpha -> q(alpha) is the CDF; hand its nodes to the
    # shared extend-anchor-resample path.  alphas is one shared row broadcast
    # across samples.
    bin_edges, probas = cdf_nodes_to_regular_grid(q, alphas, n_bins)
    bin_midpoints = (bin_edges[:, :-1] + bin_edges[:, 1:]) / 2
    pmf_mean = np.sum(probas * bin_midpoints, axis=-1)
    out_mean = pmf_mean if mean is None else np.asarray(mean, dtype=np.float64).reshape(-1)

    return DistributionPrediction(
        probas=probas,
        bin_edges=bin_edges,
        bin_midpoints=bin_midpoints,
        mean=out_mean,
    )
