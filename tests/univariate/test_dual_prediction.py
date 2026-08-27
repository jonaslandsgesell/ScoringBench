"""Contract for the dual-representation ``DistributionPrediction``.

Background
----------
``test_grid_robustness.py`` established the partition of the reported metrics:

* ``GRID_ROBUST_KEYS`` (CRPS, CRTS, energy, interval / coverage) read only the
  CDF or point--slab geometry, so they are safe on a model's *native* grid and
  are exact even when that grid carries **atoms** (zero-width bins that still
  hold PMF mass).
* ``DENSITY_BASED_KEYS`` (CDE loss, DPD, pseudospherical) read ``f = p_k/w_k``
  and are unbounded as ``w_k -> 0``; they are only comparable once every model
  is resampled onto ONE fixed grid.

So a single prediction needs two views:

* ``DistributionPredictionView``     -- the raw model grid (may hold atoms).
* ``DistributionPredictionView``  -- PCHIP-resampled onto a common grid of
  ``num_bins`` bins spanning ``[y_train_min, y_train_max]``.

The mother :class:`DistributionPrediction` owns both and routes each rule to the
representation that rule is provably correct on.

What these tests pin down
-------------------------
* ``test_native_view_preserves_atoms`` -- the native view keeps zero-width bins
  (atoms) instead of blurring them away.
* ``test_resampled_view_is_regular_common_grid`` -- the resampled view lives on
  a single shared, strictly-positive-width grid of the requested resolution and
  range.
* ``test_grid_robust_paths_agree`` -- **the key requirement**: feeding an
  already-resampled (regular) prediction through BOTH the native and the
  resampled path yields *identical* grid-robust scores.  Because the input is
  already regular, both paths see the same geometry, so the O(h) atom spread
  cannot arise and the two must agree to floating-point tolerance.
* ``test_density_rules_use_resampled_only`` -- density rules are computed from
  the common-grid view no matter which native PMF grid the model used.  Proper
  density rules still distinguish genuinely different distributions; what is
  neutralised is the model's free choice of native PMF grid width, so the SAME
  distribution binned coarsely vs finely scores the same after resampling.
* ``test_density_rules_not_defined_on_native`` -- the native branch does not
  expose the density rules at all (per the design decision: they are meaningless
  on an atom-bearing grid).
"""

import logging

import numpy as np
import pytest

from scoringbench.univariate.metrics import (
    DENSITY_BASED_KEYS,
    GRID_ROBUST_KEYS,
    compute_scoring_rules,
)
from scoringbench.univariate.wrappers import DistributionPrediction

logger = logging.getLogger(__name__)

NUM_EQUALLY_SIZED_BINS = 200
Y_LO, Y_HI = -1.0, 1.0


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _atom_prediction(centres, n_samples=200, seed=0):
    """A native prediction that puts a genuine ATOM at each centre.

    An atom is a zero-width bin carrying PMF mass: two coincident edges at the
    centre with the remaining mass spread over a wide continuous slab.  Returns
    a raw (edges, probas) pair suitable for ``DistributionPrediction``; the
    native view is verbatim, so the atom must survive into it.
    """
    rng = np.random.default_rng(seed)
    n_bins = 4
    edges = np.empty((n_samples, n_bins + 1))
    probas = np.empty((n_samples, n_bins))
    c = np.asarray(centres, dtype=float)
    c = rng.choice(c, size=n_samples)
    for i in range(n_samples):
        ci = c[i]
        # bins: [-1, ci] | [ci, ci] (atom) | [ci, 1] | [1, 1.5]
        edges[i] = np.array([Y_LO, ci, ci, Y_HI, Y_HI + 0.5])
        probas[i] = np.array([0.25, 0.5, 0.20, 0.05])
    return edges, probas


def _regular_prediction(n_samples=200, n_bins=None, lo=None, hi=None, seed=1):
    """A smooth, atom-FREE prediction already on the COMMON grid.

    Defaults to exactly the common evaluation grid (``NUM_EQUALLY_SIZED_BINS`` bins over
    ``[Y_LO, Y_HI]``) so the native view and the resampled view sit on the
    *identical* lattice.  Resampling is then an identity map and the two paths
    must agree to floating-point tolerance -- no O(h) interpolation residual.
    """
    n_bins = NUM_EQUALLY_SIZED_BINS if n_bins is None else n_bins
    lo = Y_LO if lo is None else lo
    hi = Y_HI if hi is None else hi
    rng = np.random.default_rng(seed)
    edges = np.linspace(lo, hi, n_bins + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    probas = np.empty((n_samples, n_bins))
    for i in range(n_samples):
        mu = rng.uniform(-0.4, 0.4)
        w = np.exp(-0.5 * ((mids - mu) / 0.3) ** 2)
        probas[i] = w / w.sum()
    return edges, probas


def _targets(n_samples=200, seed=2):
    rng = np.random.default_rng(seed)
    return rng.uniform(Y_LO, Y_HI, size=n_samples)


def _same_dist_two_resolutions(n_samples=200, n_bins_a=32, n_bins_b=256, seed=4):
    """The SAME underlying Gaussian mixture, binned at two native resolutions.

    Returns ``(edges_a, probas_a, edges_b, probas_b)``.  Both encode the
    identical continuous predictive density; the only difference is how finely
    the model happened to discretise it.  After both are resampled onto the
    common grid, density-based scores must agree -- otherwise the benchmark is
    just rewarding grid resolution ("benchmarking grid widths").
    """
    rng = np.random.default_rng(seed)
    mus = rng.uniform(-0.4, 0.4, size=n_samples)
    sigma = 0.25

    def _grid(n_bins):
        edges = np.linspace(Y_LO - 0.5, Y_HI + 0.5, n_bins + 1)
        mids = 0.5 * (edges[:-1] + edges[1:])
        widths = np.diff(edges)
        probas = np.empty((n_samples, n_bins))
        for i in range(n_samples):
            dens = np.exp(-0.5 * ((mids - mus[i]) / sigma) ** 2)
            mass = dens * widths          # integrate the SAME density over each bin
            probas[i] = mass / mass.sum()
        return edges, probas

    edges_a, probas_a = _grid(n_bins_a)
    edges_b, probas_b = _grid(n_bins_b)
    return edges_a, probas_a, edges_b, probas_b


# ---------------------------------------------------------------------------
# Native view keeps atoms
# ---------------------------------------------------------------------------

def test_native_view_preserves_atoms():
    edges, probas = _atom_prediction(centres=[-0.5, 0.0, 0.5])
    dist = DistributionPrediction(
        probas=probas,
        bin_edges=edges,
        bin_midpoints=0.5 * (edges[..., :-1] + edges[..., 1:]),
        mean=np.zeros(probas.shape[0]),
        num_equally_sized_bins=NUM_EQUALLY_SIZED_BINS,
        train_range=(Y_LO, Y_HI),
    )
    native = dist.native
    widths = np.diff(native.bin_edges, axis=-1)
    # at least one zero-width bin (the atom) must survive on the native view
    assert (widths <= 0).any(), "atom (zero-width bin) was blurred away by native view"


# ---------------------------------------------------------------------------
# Resampled view is a regular common grid
# ---------------------------------------------------------------------------

def test_resampled_view_is_regular_common_grid():
    edges, probas = _atom_prediction(centres=[-0.5, 0.0, 0.5])
    dist = DistributionPrediction(
        probas=probas,
        bin_edges=edges,
        bin_midpoints=0.5 * (edges[..., :-1] + edges[..., 1:]),
        mean=np.zeros(probas.shape[0]),
        num_equally_sized_bins=NUM_EQUALLY_SIZED_BINS,
        train_range=(Y_LO, Y_HI),
    )
    reg = dist.resampled
    e = np.asarray(reg.bin_edges)
    # Grow-only: the atom fixture's hull runs to Y_HI + 0.5 (shared across rows),
    # beyond the train range, so the density grid grows outward to it while the
    # low side floors at Y_LO.  It stays a single shared, regular, positive grid.
    assert e.ndim == 1, "resampled view must be a single shared grid here"
    assert e.shape[0] == NUM_EQUALLY_SIZED_BINS + 1
    assert np.isclose(e[0], Y_LO) and np.isclose(e[-1], Y_HI + 0.5)
    w = np.diff(e)
    assert (w > 0).all(), "resampled grid has non-positive bin widths"
    assert np.allclose(w, w[0]), "resampled grid is not regular"


# ---------------------------------------------------------------------------
# KEY: the two grid-robust paths agree on a regular input
# ---------------------------------------------------------------------------

def test_grid_robust_paths_agree():
    """Resampled prediction fed through native vs resampled path -> identical.

    Input already lives on the COMMON grid (positive-width, atom-free), so the
    resampled resampling is an identity map and both code paths see the same
    geometry -- exact agreement (numerical tol) for every grid-robust rule.
    """
    edges, probas = _regular_prediction()
    y = _targets(probas.shape[0])
    mids = 0.5 * (edges[:-1] + edges[1:])

    # native path: score the raw uniform grid directly
    native_scores = compute_scoring_rules(
        DistributionPrediction(
            probas=probas, bin_edges=edges, bin_midpoints=mids,
            mean=(probas * mids).sum(1),
            num_equally_sized_bins=NUM_EQUALLY_SIZED_BINS, train_range=(Y_LO, Y_HI),
        ),
        y,
        representation="native",
    )
    # resampled path: PCHIP the same uniform grid onto the common grid then score
    resampled_scores = compute_scoring_rules(
        DistributionPrediction(
            probas=probas, bin_edges=edges, bin_midpoints=mids,
            mean=(probas * mids).sum(1),
            num_equally_sized_bins=NUM_EQUALLY_SIZED_BINS, train_range=(Y_LO, Y_HI),
        ),
        y,
        representation="resampled",
    )

    for key in GRID_ROBUST_KEYS:
        n, r = native_scores[key], resampled_scores[key]
        assert np.isfinite(n) and np.isfinite(r)
        # Input already lives on the common grid, so resampling is the identity
        # map: the two paths see the same geometry and must agree to numerical
        # (PCHIP round-trip) tolerance, not merely to an interpolation floor.
        assert abs(n - r) <= 1e-6 + 1e-6 * abs(n), (
            f"grid-robust rule {key} diverged between paths: native={n} resampled={r}"
        )


# ---------------------------------------------------------------------------
# Density rules always come from the common grid
# ---------------------------------------------------------------------------

def test_density_rules_use_resampled_only():
    """Density rules must not reward native PMF grid resolution.

    Principle: a *proper* density rule scores the predictive density f(y).  Two
    genuinely different distributions (a narrow spike vs a wide slab) SHOULD
    score differently -- forcing them equal would make the rule improper.  What
    must be neutralised is the free choice of a model's own native PMF grid width:
    the SAME underlying distribution, discretised coarsely (32 bins) or finely
    (256 bins), must yield the SAME density scores once both are resampled onto
    the common grid.  Otherwise the benchmark just rewards fine binning.
    """
    y = _targets(200)
    edges_a, probas_a, edges_b, probas_b = _same_dist_two_resolutions(len(y))

    def _score(edges, probas):
        mids = 0.5 * (edges[:-1] + edges[1:])
        dist = DistributionPrediction(
            probas=probas, bin_edges=edges, bin_midpoints=mids,
            mean=(probas * mids).sum(1),
            num_equally_sized_bins=NUM_EQUALLY_SIZED_BINS, train_range=(Y_LO, Y_HI),
        )
        return compute_scoring_rules(dist, y)

    coarse = _score(edges_a, probas_a)
    fine = _score(edges_b, probas_b)
    for key in DENSITY_BASED_KEYS:
        c, f = coarse[key], fine[key]
        assert np.isfinite(c) and np.isfinite(f)
        # Same density, resampled from 32 vs 256 native bins onto the common
        # 200-bin grid: only the coarse->common resampling residual remains.
        assert abs(c - f) <= 2e-2 + 2e-2 * abs(c), (
            f"density rule {key} still depends on native PMF grid resolution: "
            f"coarse={c} fine={f}"
        )


# ---------------------------------------------------------------------------
# Density rules are not defined on the native branch
# ---------------------------------------------------------------------------

def test_density_rules_not_defined_on_native():
    edges, probas = _atom_prediction(centres=[-0.5, 0.0, 0.5])
    y = _targets(probas.shape[0])
    mids = 0.5 * (edges[..., :-1] + edges[..., 1:])
    dist = DistributionPrediction(
        probas=probas, bin_edges=edges, bin_midpoints=mids,
        mean=np.zeros(probas.shape[0]),
        num_equally_sized_bins=NUM_EQUALLY_SIZED_BINS, train_range=(Y_LO, Y_HI),
    )
    native_only = compute_scoring_rules(dist, y, representation="native")
    for key in DENSITY_BASED_KEYS:
        assert key not in native_only, (
            f"density rule {key} must NOT be computed on the native branch"
        )
    # but grid-robust rules are present on the native branch
    for key in GRID_ROBUST_KEYS:
        assert key in native_only
