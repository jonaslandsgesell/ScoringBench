"""Density-on-a-grid -> DistributionPrediction conversion for ScoringBench wrappers.

The third source adapter, alongside :mod:`quantile_based` (``quantiles_to_distribution``)
and :mod:`sample_based` (``samples_to_distribution``).  This one serves models that
own a genuine conditional density ``p(y | x)`` and can EVALUATE it at arbitrary
``y`` -- ``CDEWrapper``, ``FlexCodeWrapper``, ``SurjectorsWrapper`` -- rather than
models that only expose quantiles or draws.
"""

from __future__ import annotations

import numpy as np

from .base import DistributionPrediction


def grid_density_to_distribution(
    grid: np.ndarray,
    density: np.ndarray,
    mean: np.ndarray | None = None,
    *,
    train_range: tuple[float, float],
) -> DistributionPrediction:
    """Build a ``DistributionPrediction`` from a conditional density on a shared grid.

    Used by analytic CDE wrappers that can evaluate ``p(y | x)`` on a fixed
    ``y``-grid (the same grid for every test instance). Each grid point owns one
    bin whose width is the gap to its neighbours (outer half-cells mirrored); the
    per-bin mass is ``density * width``, normalized to sum to 1 per row.

    The prediction is flagged GRID-NATIVE, but note this is NOT a discretized
    head: these models are CONTINUOUS densities, and the grid is the wrapper's
    own evaluation choice (``n_grid``/``grid_pad``), not something the model
    commits to.  The flag is set only to skip the PCHIP resample -- re-fitting an
    interpolant to a density we can evaluate exactly would add error for nothing.
    So the resampled view keeps this grid, zero-mass-padded to the support.
    Because the grid is a ``linspace``, every bin width is strictly positive and
    the atom branch in ``metrics.unified_bin_density`` is never exercised here.

    Parameters
    ----------
    grid : (G,) array
        Shared, increasing grid of ``y`` values (the bin midpoints).
    density : (n_samples, G) or (G,) array
        Conditional density evaluated at ``grid`` for each test instance.
    mean : (n_samples,) array, optional
        Point prediction to report. If ``None`` the PMF mean is used.
    train_range : (y_lo, y_hi), required
        Shared train-target range the density rules regrid onto.
    """
    grid = np.asarray(grid, dtype=np.float64).reshape(-1)
    density = np.asarray(density, dtype=np.float64)
    if density.ndim == 1:
        density = density[np.newaxis, :]
    density = np.clip(np.nan_to_num(density, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)

    G = grid.shape[0]
    edges = np.empty(G + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (grid[:-1] + grid[1:])
    edges[0] = grid[0] - 0.5 * (grid[1] - grid[0])
    edges[-1] = grid[-1] + 0.5 * (grid[-1] - grid[-2])
    widths = np.diff(edges)

    mass = density * widths[None, :]
    totals = np.where(mass.sum(axis=1, keepdims=True) > 0, mass.sum(axis=1, keepdims=True), 1.0)
    probas = mass / totals
    out_mean = probas @ grid if mean is None else np.asarray(mean, dtype=np.float64).reshape(-1)

    return DistributionPrediction.from_histogram(
        edges, probas, mean=out_mean, train_range=train_range, is_grid_native=True
    )
