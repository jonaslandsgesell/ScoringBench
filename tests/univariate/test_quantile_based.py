"""Tests for ``quantiles_to_distribution`` (quantiles -> atom-preserving PMF).

The quantile function ``alpha -> q(alpha)`` is read as a CDF and its nodes are
used DIRECTLY as bin edges (``resampling_grid.cdf_nodes_to_native_PMF_grid``): the
quantile values themselves ARE the edges and the cumulative mass at edge ``k`` is
``alpha_k``, used verbatim (NO ``C = 0/1`` anchoring, no invented tail).  ``K``
levels therefore give ``K`` edges and ``K - 1`` bins, and the bin masses are the
CDF increments ``diff(alphas)`` RENORMALISED to sum to 1 -- the small ``~1/K``
tail mass beyond the outermost quantiles is folded back into the bins by that
renormalization rather than pushed onto an invented tail.  No resampling, no
uniform grid.  This is the NATIVE view: tied quantiles stay coincident as
zero-width Dirac bins (atoms) so the grid-robust rules (CRPS, CRTS, energy,
coverage) score them exactly.  The density rules read the resampled view instead
(``.resampled``), which neutralises the atoms onto the grow-only grid.

The properties that matter here:

* shapes / PMF validity (rows sum to 1, non-negative),
* the edges are the quantiles verbatim -- no added tail edges,
* the masses are ``diff(alphas)`` renormalised to 1 -- the CDF increments with
  the tail mass folded in,
* atoms (tied quantiles) survive as zero-width bins on the native PMF grid,
* defensive handling of unsorted / non-finite / degenerate inputs.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from scoringbench.univariate.wrappers.base import DistributionPrediction
from scoringbench.univariate.wrappers.quantile_based import quantiles_to_distribution


ALPHAS_9 = np.linspace(0.1, 0.9, 9)


def _q2d(q, alphas, **kwargs):
    """``quantiles_to_distribution`` with a data-derived ``train_range``.

    ``train_range`` is a required keyword on the production function -- it is the
    train-target range the density (resampled) view grows outward from.  These
    unit tests only exercise the quantile -> native PMF conversion and never
    compare models, so the finite min/max of the quantile block is a valid range;
    it is widened when every value ties so ``y_hi > y_lo``.  A caller may still
    pass ``train_range`` explicitly to override.
    """
    if "train_range" not in kwargs:
        finite = np.asarray(q, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        lo = float(finite.min()) if finite.size else 0.0
        hi = float(finite.max()) if finite.size else 1.0
        if hi <= lo:
            hi = lo + 1.0
        kwargs["train_range"] = (lo, hi)
    return quantiles_to_distribution(q, alphas, **kwargs)


def _normal_quantiles(alphas, locs, scales):
    """(n, K) matrix of exact Normal quantiles."""
    locs = np.asarray(locs, dtype=float)[:, None]
    scales = np.asarray(scales, dtype=float)[:, None]
    return norm.ppf(np.asarray(alphas)[None, :], loc=locs, scale=scales)


def _interior_masses(alphas):
    """The exact native masses for ``K`` levels: ``diff(alphas)`` renormalised.

    The nodes are unanchored, so the bin masses are just the forward differences
    of the (sorted) alphas divided by their sum -- the ``~alpha_0`` / ``~(1 -
    alpha_{K-1})`` tail mass is folded into all bins by that renormalization.
    """
    d = np.diff(np.sort(alphas))
    return d / d.sum()


def _cdf_at(dist, x):
    """Interpolate the reconstructed CDF of row 0 at points ``x``."""
    edges = dist.bin_edges[0]
    cdf_edges = np.concatenate([[0.0], np.cumsum(dist.probas[0])])
    return np.interp(x, edges, cdf_edges)


# ---------------------------------------------------------------------------
# Shapes / basic contract
# ---------------------------------------------------------------------------

def test_default_grid_size_matches_number_of_quantiles():
    q = _normal_quantiles(ALPHAS_9, [0.0, 5.0], [1.0, 2.0])
    dist = _q2d(q, ALPHAS_9)

    k = len(ALPHAS_9)
    assert isinstance(dist, DistributionPrediction)
    # K levels -> K edges (the quantiles verbatim), K - 1 bins.
    assert dist.probas.shape == (2, k - 1)
    assert dist.bin_edges.shape == (2, k)
    assert dist.bin_midpoints.shape == (2, k - 1)
    assert dist.mean.shape == (2,)


@pytest.mark.parametrize("n_alphas", [1, 2, 7, 64, 257])
def test_grid_size_tracks_the_number_of_quantile_levels(n_alphas):
    alphas = np.linspace(1 / (n_alphas + 1), n_alphas / (n_alphas + 1), n_alphas)
    q = _normal_quantiles(alphas, [0.0, -3.0, 1.0], [1.0, 0.5, 4.0])
    dist = _q2d(q, alphas)

    # A single level cannot define an interval, so it becomes a zero-width atom
    # (the value repeated -> two coincident edges); K columns give K - 1 bins.
    k = max(n_alphas, 2)
    assert dist.probas.shape == (3, k - 1)
    assert dist.bin_edges.shape == (3, k)
    np.testing.assert_allclose(dist.probas.sum(axis=1), 1.0, atol=1e-12)


def test_one_dimensional_input_is_promoted_to_a_single_row():
    q = norm.ppf(ALPHAS_9)
    dist = _q2d(q, ALPHAS_9)
    k = len(ALPHAS_9)
    assert dist.probas.shape == (1, k - 1)
    assert dist.bin_edges.shape == (1, k)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        _q2d(np.zeros((2, 4)), ALPHAS_9)


# ---------------------------------------------------------------------------
# The grid itself
# ---------------------------------------------------------------------------

def test_edges_are_the_quantiles_verbatim():
    """The quantile values ARE the bin edges -- no resampling, no tail edges.

    Two rows with wildly different scales (sigma = 1e-3 and 50) are converted in
    one call; the edges of each row are exactly its (sorted) quantiles.
    """
    q = _normal_quantiles(ALPHAS_9, [0.0, 100.0], [1e-3, 50.0])
    dist = _q2d(q, ALPHAS_9)

    # Edges equal the quantiles verbatim -- no tail extension.
    np.testing.assert_allclose(dist.bin_edges, np.sort(q, axis=1), rtol=1e-12)
    assert np.all(np.diff(dist.bin_edges, axis=1) >= 0.0)
    np.testing.assert_allclose(
        dist.bin_midpoints, (dist.bin_edges[:, :-1] + dist.bin_edges[:, 1:]) / 2
    )


def test_support_is_the_quantile_hull_no_invented_tail():
    """The support is exactly ``[q_0, q_{K-1}]`` -- no invented tail either side.

    The mass below ``alpha_0`` / above ``alpha_{K-1}`` is folded into the
    outermost bins by renormalisation rather than placed on an invented tail, so
    the reported support is the quantile hull itself.
    """
    q = _normal_quantiles(ALPHAS_9, [0.0, 5.0], [1.0, 2.0])
    dist = _q2d(q, ALPHAS_9)

    np.testing.assert_allclose(dist.bin_edges[:, 0], q[:, 0], rtol=1e-12)
    np.testing.assert_allclose(dist.bin_edges[:, -1], q[:, -1], rtol=1e-12)


def test_masses_are_the_renormalised_alpha_increments():
    """Masses are the CDF increments ``diff(alphas)`` renormalised to 1.

    The quantiles are the edges and ``C`` at them is ``alpha``; renormalising the
    unanchored increments, the bin masses are the forward differences of the
    alphas divided by their sum -- independent of the quantile *values*, which
    only set where the mass sits, not how much.
    """
    q = _normal_quantiles(ALPHAS_9, [0.0, 3.0], [1.0, 0.5])
    dist = _q2d(q, ALPHAS_9)

    expected = _interior_masses(ALPHAS_9)
    for i in range(dist.probas.shape[0]):
        np.testing.assert_allclose(dist.probas[i], expected, rtol=1e-12)
    np.testing.assert_allclose(dist.probas.sum(axis=1), 1.0, atol=1e-12)


def test_unequally_spaced_levels_give_the_renormalised_increments():
    """The grid is the quantiles, the masses are ``diff(alphas)`` renormalised.

    ``alphas = [.1, .2, .5, .9]`` at ``q = [0, 1, 2, 3]``.  The edges are the
    quantiles ``[0, 1, 2, 3]`` (no tails) and the 3 raw increments are
    ``diff([.1, .2, .5, .9]) = [.1, .3, .4]``; renormalised by their sum ``.8``
    they become ``[.125, .375, .5]``.
    """
    alphas = np.array([0.1, 0.2, 0.5, 0.9])
    q = np.array([[0.0, 1.0, 2.0, 3.0]])
    dist = _q2d(q, alphas)

    np.testing.assert_allclose(dist.bin_edges[0], [0.0, 1.0, 2.0, 3.0], rtol=1e-12)
    np.testing.assert_allclose(dist.probas[0], [0.125, 0.375, 0.5], rtol=1e-12)


def test_pmf_is_valid():
    rng = np.random.default_rng(0)
    q = np.sort(rng.normal(size=(20, len(ALPHAS_9))) * 3.0, axis=1)
    dist = _q2d(q, ALPHAS_9)

    assert np.all(dist.probas >= 0.0)
    assert np.all(np.isfinite(dist.probas))
    assert np.all(np.isfinite(dist.bin_edges))
    np.testing.assert_allclose(dist.probas.sum(axis=1), 1.0, atol=1e-12)
    # Reconstructed CDF is non-decreasing.
    cdf = np.cumsum(dist.probas, axis=1)
    assert np.all(np.diff(cdf, axis=1) >= -1e-15)


# ---------------------------------------------------------------------------
# Correctness of the CDF construction
# ---------------------------------------------------------------------------

def test_masses_are_exactly_the_renormalised_increments():
    """Pin the construction to the bit: masses == ``diff(alphas) / sum``.

    The quantile values are used verbatim as edges and ``C`` at each is its
    ``alpha``, so the masses do not depend on the quantiles at all -- they are the
    renormalised forward differences of the alpha vector, per row identical.
    """
    rng = np.random.default_rng(1)
    alphas = np.linspace(0.02, 0.98, 25)
    q = np.sort(rng.normal(loc=rng.normal(size=(6, 1)), scale=2.0, size=(6, 25)), axis=1)

    dist = _q2d(q, alphas)
    expected = _interior_masses(alphas)
    for i in range(q.shape[0]):
        np.testing.assert_allclose(dist.probas[i], expected, rtol=0, atol=1e-12)


def test_cdf_at_the_quantiles_is_the_renormalised_cumulative():
    """``F_hat(q_k)`` is the renormalised cumulative of the alpha increments.

    Because the quantiles are used verbatim as edges, the reconstructed CDF hits
    each ``q_k`` at ``cumsum(diff(alphas))[k-1] / sum`` -- the alphas rescaled so
    the outermost levels land on ``0`` and ``1`` (no resampling error otherwise).
    """
    for k in (9, 51, 199):
        alphas = np.linspace(0.02, 0.98, k)
        q = _normal_quantiles(alphas, [0.0], [1.0])
        dist = _q2d(q, alphas)
        cdf = np.concatenate([[0.0], np.cumsum(dist.probas[0])])
        got = np.interp(q[0], dist.bin_edges[0], cdf)
        expected = np.concatenate([[0.0], np.cumsum(_interior_masses(alphas))])
        np.testing.assert_allclose(got, expected, atol=1e-12)


def test_recovers_normal_cdf_and_moments():
    alphas = np.linspace(0.001, 0.999, 199)
    loc, scale = 2.5, 1.5
    q = _normal_quantiles(alphas, [loc], [scale])
    dist = _q2d(q, alphas)

    x = np.linspace(loc - 3 * scale, loc + 3 * scale, 101)
    np.testing.assert_allclose(_cdf_at(dist, x), norm.cdf(x, loc, scale), atol=5e-3)

    assert dist.mean[0] == pytest.approx(loc, abs=0.05)
    var = np.sum(dist.probas[0] * (dist.bin_midpoints[0] - dist.mean[0]) ** 2)
    assert np.sqrt(var) == pytest.approx(scale, rel=0.1)


def test_cdf_error_decreases_with_more_quantiles():
    """Finer quantile grids => the reconstructed CDF converges to the truth.

    The residual floor is the folded-in tail mass ``~alpha_0``, not the bin
    width: with levels starting at ``alpha_0 = 1 / (k + 1)`` the outermost level
    controls how much mass is redistributed, so convergence is driven by it.
    """
    x = np.linspace(-3.0, 3.0, 201)
    errors, floors = [], []
    for k in (9, 33, 129):
        alphas = np.linspace(1 / (k + 1), k / (k + 1), k)
        q = _normal_quantiles(alphas, [0.0], [1.0])
        dist = _q2d(q, alphas)
        errors.append(np.max(np.abs(_cdf_at(dist, x) - norm.cdf(x))))
        floors.append(float(alphas[0]))

    assert errors[1] < errors[0]
    assert errors[2] < errors[1]
    assert errors[2] < 3 * floors[2]


# ---------------------------------------------------------------------------
# Defensive handling
# ---------------------------------------------------------------------------

def test_unsorted_quantiles_and_alphas_are_repaired():
    alphas = np.array([0.75, 0.25, 0.5])
    q = np.array([[2.0, 0.0, 1.0], [4.0, 1.0, 1.5]])

    dist = _q2d(q, alphas)
    ref = _q2d(np.sort(q, axis=1), np.sort(alphas))

    np.testing.assert_allclose(dist.probas, ref.probas)
    np.testing.assert_allclose(dist.bin_edges, ref.bin_edges)


def test_non_finite_quantiles_are_sanitized_into_y_range():
    q = np.array([[np.nan, 0.5, np.inf], [-np.inf, 0.2, 0.9]])
    dist = _q2d(q, np.array([0.25, 0.5, 0.75]), y_range=(0.0, 1.0))

    assert np.all(np.isfinite(dist.bin_edges))
    assert np.all(np.isfinite(dist.probas))
    np.testing.assert_allclose(dist.probas.sum(axis=1), 1.0, atol=1e-12)


def test_tied_quantiles_survive_as_atoms_on_the_native_grid():
    """Tied quantiles stay coincident: the native PMF grid keeps zero-width bins.

    This is the whole point of the native view.  Rather than blurring a tie away,
    the tied quantiles are used verbatim as edges, so the run of equal values is a
    zero-width Dirac bin carrying the tie's mass.  The grid-robust rules score
    that exactly; the density rules read ``.resampled`` (a positive-width grid)
    instead.  ``compute_metrics`` must stay finite on both.
    """
    from scoringbench.univariate.metrics import compute_metrics

    alphas = np.linspace(0.01, 0.99, 30)
    q = np.full((3, 30), 7.0)
    q[1] = np.concatenate([np.full(15, 1.0), np.full(15, 2.0)])  # two atoms
    q[2] = np.linspace(6.0, 8.0, 30)

    dist = _q2d(q, alphas)

    widths = np.diff(dist.bin_edges, axis=1)
    # Native grid: zero-width bins (atoms) are allowed and expected on tied rows.
    assert np.all(widths >= 0.0)
    assert np.any(widths[1] == 0.0), "tied quantiles must stay coincident (atoms)"
    # The edges are the quantiles verbatim, so the support spans them.
    assert dist.bin_edges[1, 0] == pytest.approx(1.0)
    assert dist.bin_edges[1, -1] == pytest.approx(2.0)
    np.testing.assert_allclose(dist.probas.sum(axis=1), 1.0, atol=1e-12)
    # The fully tied (point-mass) row is a Dirac at 7.0.
    assert dist.mean[0] == pytest.approx(7.0, abs=1e-3)

    metrics = compute_metrics(dist, np.array([7.0, 1.5, 7.0]))
    for key, value in metrics.items():
        if value is not None:
            assert np.isfinite(value), f"{key} = {value}"


def test_single_quantile_level():
    q = np.array([[3.0], [0.0]])
    dist = _q2d(q, np.array([0.5]))

    # One level is a point mass: it becomes a zero-width atom (the value repeated
    # -> 2 coincident edges, 1 Dirac bin holding all the mass), NOT a fabricated
    # interval of arbitrary width.
    assert dist.probas.shape == (2, 1)
    np.testing.assert_allclose(np.diff(dist.bin_edges, axis=1), 0.0, atol=0.0)
    np.testing.assert_allclose(dist.bin_edges[:, 0], q[:, 0], rtol=1e-12)
    np.testing.assert_allclose(dist.probas.sum(axis=1), 1.0, atol=1e-12)


def test_explicit_mean_overrides_pmf_mean():
    q = _normal_quantiles(ALPHAS_9, [0.0, 5.0], [1.0, 2.0])
    supplied = np.array([-1.0, 42.0])

    dist = _q2d(q, ALPHAS_9, mean=supplied)
    np.testing.assert_allclose(dist.mean, supplied)

    pmf_mean = _q2d(q, ALPHAS_9).mean
    np.testing.assert_allclose(
        pmf_mean, np.sum(dist.probas * dist.bin_midpoints, axis=1)
    )


def test_rows_are_independent():
    """Row i's output must not depend on the other rows in the batch."""
    q = _normal_quantiles(ALPHAS_9, [0.0, 500.0, -20.0], [1.0, 0.01, 7.0])
    batch = _q2d(q, ALPHAS_9)

    for i in range(q.shape[0]):
        single = _q2d(q[i : i + 1], ALPHAS_9)
        np.testing.assert_allclose(batch.probas[i], single.probas[0], atol=1e-12)
        np.testing.assert_allclose(batch.bin_edges[i], single.bin_edges[0], rtol=1e-12)


def test_metrics_are_finite_on_the_output():
    from scoringbench.univariate.metrics import compute_metrics

    rng = np.random.default_rng(3)
    alphas = np.linspace(0.01, 0.99, 50)
    y = rng.normal(size=40)
    q = _normal_quantiles(alphas, y, np.full(40, 1.0))
    dist = _q2d(q, alphas)

    metrics = compute_metrics(dist, y)
    for key, value in metrics.items():
        if value is not None:
            assert np.isfinite(value), f"{key} = {value}"
