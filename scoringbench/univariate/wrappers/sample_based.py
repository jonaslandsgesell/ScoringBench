"""Shared utilities for sample-based ScoringBench wrappers.

``SampleBasedWrapper``
    Base class for genuinely sample-based models (normalizing flows, BART, …).
    Subclasses only implement ``_draw_samples(X, n)`` returning an
    ``(n_test, n)`` array of conditional draws of the target.  The base class
    accumulates draws in chunks under a wall-clock budget
    (``MAX_SAMPLE_SECONDS``, default 120 s — it may sample for at most two
    minutes, then derives the PMF from whatever was collected) and converts the
    pooled draws into a ``DistributionPrediction``.
"""

from __future__ import annotations

import time

import numpy as np

from .base import (
    DistributionPrediction,
    ProbabilisticWrapper,
    cdf_nodes_to_regular_grid,
)


# ---------------------------------------------------------------------------
# Samples -> DistributionPrediction
# ---------------------------------------------------------------------------

def _ecdf_to_regular_grid(row: np.ndarray, n_bins: int):
    """Bin one row of draws onto a regular grid; return ``(edges, masses)``.

    Nodes are the unique sorted draws with mid-rank (Hazen) CDF values
    ``(cum - c/2) / n`` -- strictly increasing in ``(0, 1)``, so the resampled CDF
    is monotone.  Equal *width* (not equal probability) means a repeated draw's
    mass lands wholly in one bin instead of collapsing the grid; bins may be
    empty but keep a strictly positive width.  A single distinct value is a point
    mass; the support guard invents the only width a histogram can give it.
    """
    u, counts = np.unique(row, return_counts=True)          # u strictly increasing

    if u.shape[0] == 1:
        u = np.repeat(u, 2)                                 # tied pair; the support
        counts = np.array([1, 1])                           # guard supplies the width

    n = counts.sum()
    p_nodes = (np.cumsum(counts) - 0.5 * counts) / n        # mid-rank, strictly increasing
    edges, masses = cdf_nodes_to_regular_grid(u, p_nodes, n_bins)
    return edges[0], masses[0]


def samples_to_distribution(
    samples: np.ndarray,
    n_bins: int = 100,
    y_range: tuple[float, float] | None = None,
) -> DistributionPrediction:
    """Derive a ``DistributionPrediction`` from conditional draws.

    Each row's empirical CDF is resampled onto ``n_bins`` equally *wide* bins (see
    :func:`_ecdf_to_regular_grid`), so every width is ``span / n_bins > 0`` and the
    draws' shape is carried by the masses.  Bins may be empty but keep a strictly
    positive width, so the piecewise-constant density stays well defined.

    Parameters
    ----------
    samples : (n_test, n_draws) array
        Conditional samples of the target, one row per test instance.
    n_bins : int
        Number of equally wide bins per sample (default 100).
    y_range : tuple[float, float], optional
        Fallback range for sanitizing non-finite samples.
    """
    n_bins = max(int(n_bins), 1)

    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim == 1:
        samples = samples[:, None]

    # Replace any non-finite draws with the per-row median; if a whole row is
    # non-finite fall back to the y_range midpoint (or 0 when y_range is None)
    # so estimation stays well defined.
    if not np.all(np.isfinite(samples)):
        finite = np.isfinite(samples)
        fallback = 0.0 if y_range is None else 0.5 * (float(y_range[0]) + float(y_range[1]))
        row_median = np.array([
            np.median(samples[i, finite[i]]) if finite[i].any() else fallback
            for i in range(samples.shape[0])
        ])
        samples = np.where(finite, samples, row_median[:, None])

    n_test = samples.shape[0]

    # Per-row regular grid and masses from the interpolated eCDF.
    bin_edges = np.empty((n_test, n_bins + 1), dtype=np.float64)
    probas = np.empty((n_test, n_bins), dtype=np.float64)
    for i in range(n_test):
        bin_edges[i], probas[i] = _ecdf_to_regular_grid(samples[i], n_bins)

    bin_midpoints = (bin_edges[:, :-1] + bin_edges[:, 1:]) / 2.0

    mean = samples.mean(axis=1)
    return DistributionPrediction(
        probas=probas,
        bin_edges=bin_edges,
        bin_midpoints=bin_midpoints,
        mean=mean,
        is_sample_based=True,
    )


def grid_density_to_distribution(
    grid: np.ndarray,
    density: np.ndarray,
    mean: np.ndarray | None = None,
) -> DistributionPrediction:
    """Build a ``DistributionPrediction`` from a conditional density on a shared grid.

    Used by analytic CDE wrappers that can evaluate ``p(y | x)`` on a fixed
    ``y``-grid (the same grid for every test instance). Each grid point owns one
    bin whose width is the gap to its neighbours (outer half-cells mirrored); the
    per-bin mass is ``density * width``, normalized to sum to 1 per row.

    Parameters
    ----------
    grid : (G,) array
        Shared, increasing grid of ``y`` values (the bin midpoints).
    density : (n_samples, G) or (G,) array
        Conditional density evaluated at ``grid`` for each test instance.
    mean : (n_samples,) array, optional
        Point prediction to report. If ``None`` the PMF mean is used.
    """
    grid = np.asarray(grid, dtype=np.float64).reshape(-1)
    density = np.asarray(density, dtype=np.float64)
    if density.ndim == 1:
        density = density[np.newaxis, :]
    density = np.nan_to_num(density, nan=0.0, posinf=0.0, neginf=0.0)
    density = np.clip(density, 0.0, None)

    G = grid.shape[0]
    edges = np.empty(G + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (grid[:-1] + grid[1:])
    edges[0] = grid[0] - 0.5 * (grid[1] - grid[0])
    edges[-1] = grid[-1] + 0.5 * (grid[-1] - grid[-2])
    widths = np.diff(edges)

    mass = density * widths[None, :]
    totals = mass.sum(axis=1, keepdims=True)
    totals = np.where(totals > 0, totals, 1.0)
    probas = mass / totals

    out_mean = probas @ grid if mean is None else np.asarray(mean, dtype=np.float64).reshape(-1)
    return DistributionPrediction(
        probas=probas,
        bin_edges=edges,
        bin_midpoints=grid,
        mean=out_mean,
    )


# ---------------------------------------------------------------------------
# Sample-based wrapper base
# ---------------------------------------------------------------------------

class SampleBasedWrapper(ProbabilisticWrapper):
    """Base class for models whose predictive density is accessed via sampling.

    Subclasses implement :meth:`_draw_samples`. The PMF is derived from the
    pooled draws; sampling is capped at ``MAX_SAMPLE_SECONDS`` wall-clock
    seconds per ``predict_distribution`` call.

    Class attributes
    ----------------
    N_SAMPLES : int
        Target number of conditional draws per test instance (default 300).
    SAMPLE_CHUNK : int
        Draws requested per call to ``_draw_samples``; the budget is checked
        between chunks. Set equal to ``N_SAMPLES`` for one-shot samplers.
    MAX_SAMPLE_SECONDS : float
        Hard wall-clock cap on sampling (default 120 s). Once exceeded, the PMF
        is built from whatever draws were collected so far.
    N_BINS : int
        Number of equally *wide* bins in the PMF derived from the draws via
        :func:`samples_to_distribution`. Widths are uniform; the draws' shape is
        carried by the per-bin masses.
    """

    N_SAMPLES: int = 300
    SAMPLE_CHUNK: int = 100
    MAX_SAMPLE_SECONDS: float = 120.0
    N_BINS: int = 100

    def _draw_samples(self, X, n_samples: int) -> np.ndarray:
        """Return an ``(n_test, n_samples)`` array of conditional target draws."""
        raise NotImplementedError

    def _collect_samples(self, X) -> np.ndarray:
        target = int(self.N_SAMPLES)
        chunk = int(self.SAMPLE_CHUNK) or target
        collected: list[np.ndarray] = []
        n_have = 0
        start = time.monotonic()
        while n_have < target:
            take = min(chunk, target - n_have)
            s = np.asarray(self._draw_samples(X, take), dtype=np.float64)
            if s.ndim == 1:
                s = s[:, None]
            collected.append(s)
            n_have += s.shape[1]
            if time.monotonic() - start >= self.MAX_SAMPLE_SECONDS:
                break
        if not collected:
            raise RuntimeError("No samples were drawn.")
        return np.concatenate(collected, axis=1)

    def predict_distribution(self, X) -> DistributionPrediction:
        samples = self._collect_samples(X)
        return samples_to_distribution(samples, n_bins=self.N_BINS)

    def predict(self, X) -> np.ndarray:
        return self._collect_samples(X).mean(axis=1)
