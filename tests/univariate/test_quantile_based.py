"""Tests for ``quantiles_to_distribution`` (quantiles -> atom-preserving PMF).

The quantile function ``alpha -> q(alpha)`` is read as a CDF and its nodes are
used DIRECTLY as bin edges (``resampling_grid.cdf_nodes_to_native_PMF_grid``): the
quantile values themselves ARE the edges and the cumulative mass at edge ``k`` is
``alpha_k``, used verbatim (no invented tail).  The tail masses the model does
not place -- ``alpha_0`` below ``q_0`` and ``1 - alpha_{K-1}`` above ``q_{K-1}``
-- are kept as zero-width atoms AT the outermost quantiles, so every reported
level stays exact.  ``K`` interior levels therefore give ``K + 2`` edges and
``K + 1`` bins with masses ``[alpha_0, diff(alphas), 1 - alpha_{K-1}]``; no atom
is added where a level already is ``0`` / ``1``.  No resampling, no uniform
grid.  This is the NATIVE view: tied quantiles stay coincident as zero-width
Dirac bins (atoms) so the grid-robust rules (CRPS, CRTS, energy, coverage) score
them exactly.  The density rules read the resampled view instead
(``.resampled``), which neutralises the atoms onto the grow-only grid.

The properties that matter here:

* shapes / PMF validity (rows sum to 1, non-negative),
* the edges are the quantiles verbatim -- the outermost ones doubled as tail
  atoms, no edge beyond the quantile hull,
* the masses are the exact level increments including the two tail atoms,
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


def _native_masses(alphas):
    """The exact native masses: ``[alpha_0, diff(alphas), 1 - alpha_{K-1}]``.

    The interior increments are the reported level differences verbatim; the two
    tail masses are atoms on the outermost quantiles (omitted at ``0`` / ``1``).
    """
    a = np.sort(alphas)
    parts = [np.diff(a)]
    if a[0] > 0.0:
        parts.insert(0, a[:1])
    if a[-1] < 1.0:
        parts.append(1.0 - a[-1:])
    return np.concatenate(parts)


def _with_tail_atoms(q):
    """Quantile rows with the outermost values doubled (the tail-atom edges)."""
    q = np.sort(q, axis=1)
    return np.concatenate([q[:, :1], q, q[:, -1:]], axis=1)


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
    # K interior levels -> K + 2 edges (quantiles + two tail atoms), K + 1 bins.
    assert dist.probas.shape == (2, k + 1)
    assert dist.bin_edges.shape == (2, k + 2)
    assert dist.bin_midpoints.shape == (2, k + 1)
    assert dist.mean.shape == (2,)


@pytest.mark.parametrize("n_alphas", [1, 2, 7, 64, 257])
def test_grid_size_tracks_the_number_of_quantile_levels(n_alphas):
    alphas = np.linspace(1 / (n_alphas + 1), n_alphas / (n_alphas + 1), n_alphas)
    q = _normal_quantiles(alphas, [0.0, -3.0, 1.0], [1.0, 0.5, 4.0])
    dist = _q2d(q, alphas)

    # A single level cannot define an interval, so it becomes a zero-width atom
    # (the value repeated at levels [0, 1] -> one bin, no tail atoms); K > 1
    # interior levels give K + 1 bins (two tail atoms).
    n_bins = 1 if n_alphas == 1 else n_alphas + 1
    assert dist.probas.shape == (3, n_bins)
    assert dist.bin_edges.shape == (3, n_bins + 1)
    np.testing.assert_allclose(dist.probas.sum(axis=1), 1.0, atol=1e-12)


def test_one_dimensional_input_is_promoted_to_a_single_row():
    q = norm.ppf(ALPHAS_9)
    dist = _q2d(q, ALPHAS_9)
    k = len(ALPHAS_9)
    assert dist.probas.shape == (1, k + 1)
    assert dist.bin_edges.shape == (1, k + 2)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        _q2d(np.zeros((2, 4)), ALPHAS_9)


# ---------------------------------------------------------------------------
# The grid itself
# ---------------------------------------------------------------------------

def test_edges_are_the_quantiles_verbatim():
    """The quantile values ARE the bin edges -- no resampling, no invented tail.

    Two rows with wildly different scales (sigma = 1e-3 and 50) are converted in
    one call; the edges of each row are exactly its (sorted) quantiles, with the
    outermost two doubled as the zero-width tail atoms.
    """
    q = _normal_quantiles(ALPHAS_9, [0.0, 100.0], [1e-3, 50.0])
    dist = _q2d(q, ALPHAS_9)

    np.testing.assert_allclose(dist.bin_edges, _with_tail_atoms(q), rtol=1e-12)
    assert np.all(np.diff(dist.bin_edges, axis=1) >= 0.0)
    np.testing.assert_allclose(
        dist.bin_midpoints, (dist.bin_edges[:, :-1] + dist.bin_edges[:, 1:]) / 2
    )


def test_support_is_the_quantile_hull_no_invented_tail():
    """The support is exactly ``[q_0, q_{K-1}]`` -- no invented tail either side.

    The mass below ``alpha_0`` / above ``alpha_{K-1}`` sits as atoms ON the
    outermost quantiles rather than on an invented tail, so the reported support
    is the quantile hull itself.
    """
    q = _normal_quantiles(ALPHAS_9, [0.0, 5.0], [1.0, 2.0])
    dist = _q2d(q, ALPHAS_9)

    np.testing.assert_allclose(dist.bin_edges[:, 0], q[:, 0], rtol=1e-12)
    np.testing.assert_allclose(dist.bin_edges[:, -1], q[:, -1], rtol=1e-12)


def test_masses_are_the_level_increments_with_tail_atoms():
    """Masses are ``[alpha_0, diff(alphas), 1 - alpha_{K-1}]`` -- no rescaling.

    The quantiles are the edges and ``C`` at them is ``alpha``, so the bin masses
    are the level increments verbatim plus the two tail atoms -- independent of
    the quantile *values*, which only set where the mass sits, not how much.
    """
    q = _normal_quantiles(ALPHAS_9, [0.0, 3.0], [1.0, 0.5])
    dist = _q2d(q, ALPHAS_9)

    expected = _native_masses(ALPHAS_9)
    for i in range(dist.probas.shape[0]):
        np.testing.assert_allclose(dist.probas[i], expected, rtol=1e-12)
    np.testing.assert_allclose(dist.probas.sum(axis=1), 1.0, atol=1e-12)


def test_unequally_spaced_levels_keep_their_increments():
    """``alphas = [.1, .2, .5, .9]`` at ``q = [0, 1, 2, 3]``.

    The edges are ``[0, 0, 1, 2, 3, 3]`` (quantiles + tail atoms) and the masses
    ``[.1, .1, .3, .4, .1]``: the lower tail atom, ``diff(alphas)`` verbatim, and
    the upper tail atom.
    """
    alphas = np.array([0.1, 0.2, 0.5, 0.9])
    q = np.array([[0.0, 1.0, 2.0, 3.0]])
    dist = _q2d(q, alphas)

    np.testing.assert_allclose(dist.bin_edges[0], [0.0, 0.0, 1.0, 2.0, 3.0, 3.0], rtol=1e-12)
    np.testing.assert_allclose(dist.probas[0], [0.1, 0.1, 0.3, 0.4, 0.1], rtol=1e-12)


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

def test_masses_are_exactly_the_level_increments():
    """Pin the construction to the bit: masses == ``_native_masses(alphas)``.

    The quantile values are used verbatim as edges and ``C`` at each is its
    ``alpha``, so the masses do not depend on the quantiles at all -- they are the
    level increments plus the two tail atoms, per row identical.
    """
    rng = np.random.default_rng(1)
    alphas = np.linspace(0.02, 0.98, 25)
    q = np.sort(rng.normal(loc=rng.normal(size=(6, 1)), scale=2.0, size=(6, 25)), axis=1)

    dist = _q2d(q, alphas)
    expected = _native_masses(alphas)
    for i in range(q.shape[0]):
        np.testing.assert_allclose(dist.probas[i], expected, rtol=0, atol=1e-12)


def test_cdf_at_the_quantiles_is_exactly_alpha():
    """``F_hat(q_k) = alpha_k``: every reported level is kept, nothing rescaled.

    The cumulative mass through the bin ending at ``q_k`` (the lower tail atom
    first) is exactly ``alpha_k``; the last bin (upper tail atom) closes at 1.
    """
    for k in (9, 51, 199):
        alphas = np.linspace(0.02, 0.98, k)
        q = _normal_quantiles(alphas, [0.0], [1.0])
        dist = _q2d(q, alphas)
        cdf_right = np.cumsum(dist.probas[0])
        np.testing.assert_allclose(cdf_right[:-1], alphas, atol=1e-12)
        assert cdf_right[-1] == pytest.approx(1.0, abs=1e-12)


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

    The residual floor is the tail atom mass ``alpha_0`` (the CDF jumps to it at
    ``q_0``), not the bin width: with levels starting at ``alpha_0 = 1 / (k + 1)``
    the outermost level controls the error, so convergence is driven by it.
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
