"""Tests for ``quantiles_to_distribution`` (quantiles -> regularly binned PMF).

The quantiles are *not* used as bin edges.  The quantile function is read as a
CDF, extended by one local spacing on each side with ``alpha = 0`` / ``alpha = 1``
pinned at the extended ends, and linearly interpolated onto ``K - 1`` equally
*wide* bins spanning that support.  ``K`` levels still give ``K - 1`` bins, but
they are equally wide rather than equally probable, so the model's sharpness
shows up as mass concentration instead of as bin spacing.  The properties that
matter downstream are:

* shapes / PMF validity (rows sum to 1, non-negative),
* the grid is *regular* -- every width is ``span / n_bins > 0``, which is what
  makes ``p_k / w_k`` well defined for the density-based rules even when the
  model's quantiles tie,
* the masses are exactly the increments of the interpolated CDF, so the
  reconstructed CDF converges to the true one as the quantile grid is refined,
* defensive handling of unsorted / non-finite / degenerate inputs.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from scoringbench.univariate.wrappers.base import DistributionPrediction, _monotone_cdf_at
from scoringbench.univariate.wrappers.quantile_based import quantiles_to_distribution


ALPHAS_9 = np.linspace(0.1, 0.9, 9)


def _normal_quantiles(alphas, locs, scales):
    """(n, K) matrix of exact Normal quantiles."""
    locs = np.asarray(locs, dtype=float)[:, None]
    scales = np.asarray(scales, dtype=float)[:, None]
    return norm.ppf(np.asarray(alphas)[None, :], loc=locs, scale=scales)


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
    dist = quantiles_to_distribution(q, ALPHAS_9)

    assert isinstance(dist, DistributionPrediction)
    assert dist.probas.shape == (2, len(ALPHAS_9) - 1)
    assert dist.bin_edges.shape == (2, len(ALPHAS_9))
    assert dist.bin_midpoints.shape == (2, len(ALPHAS_9) - 1)
    assert dist.mean.shape == (2,)


@pytest.mark.parametrize("n_alphas", [1, 2, 7, 64, 257])
def test_grid_size_tracks_the_number_of_quantile_levels(n_alphas):
    alphas = np.linspace(1 / (n_alphas + 1), n_alphas / (n_alphas + 1), n_alphas)
    q = _normal_quantiles(alphas, [0.0, -3.0, 1.0], [1.0, 0.5, 4.0])
    dist = quantiles_to_distribution(q, alphas)

    # A single level cannot define an interval, so it is padded into one bin.
    n_bins = max(n_alphas - 1, 1)
    assert dist.probas.shape == (3, n_bins)
    assert dist.bin_edges.shape == (3, n_bins + 1)
    np.testing.assert_allclose(dist.probas.sum(axis=1), 1.0, atol=1e-12)


def test_one_dimensional_input_is_promoted_to_a_single_row():
    q = norm.ppf(ALPHAS_9)
    dist = quantiles_to_distribution(q, ALPHAS_9)
    assert dist.probas.shape == (1, len(ALPHAS_9) - 1)
    assert dist.bin_edges.shape == (1, len(ALPHAS_9))


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        quantiles_to_distribution(np.zeros((2, 4)), ALPHAS_9)


# ---------------------------------------------------------------------------
# The grid itself
# ---------------------------------------------------------------------------

def test_edges_are_a_regular_grid():
    """The grid is equally spaced, whatever the quantiles look like.

    Two rows with wildly different scales (sigma = 1e-3 and 50) are converted in
    one call; each still gets a *regular* per-sample grid, because the spacing is
    ``(z_max - z_min) / n_bins`` by construction rather than inherited from the
    quantile spacing.
    """
    q = _normal_quantiles(ALPHAS_9, [0.0, 100.0], [1e-3, 50.0])
    dist = quantiles_to_distribution(q, ALPHAS_9)

    widths = np.diff(dist.bin_edges, axis=1)
    assert np.all(widths > 0.0)
    # Equal up to float64 rounding of the linspace, relative to the row's own scale.
    spread = (widths.max(axis=1) - widths.min(axis=1)) / widths.mean(axis=1)
    assert np.all(spread < 1e-12), f"grid is not regular: {spread}"
    np.testing.assert_allclose(
        dist.bin_midpoints, (dist.bin_edges[:, :-1] + dist.bin_edges[:, 1:]) / 2
    )


def test_support_extends_one_local_spacing_beyond_the_quantiles():
    """The tails get somewhere to live: ``z_min = q_0 - (q_1 - q_0)`` and mirror.

    The mass below ``alpha_0`` and above ``alpha_{K-1}`` is *placed* on that
    extension rather than dropped, which is why the output needs no
    renormalisation and the reported CDF is the model's own.
    """
    q = _normal_quantiles(ALPHAS_9, [0.0, 5.0], [1.0, 2.0])
    dist = quantiles_to_distribution(q, ALPHAS_9)

    np.testing.assert_allclose(dist.bin_edges[:, 0], q[:, 0] - (q[:, 1] - q[:, 0]), rtol=1e-12)
    np.testing.assert_allclose(dist.bin_edges[:, -1], q[:, -1] + (q[:, -1] - q[:, -2]), rtol=1e-12)
    # The predicted quantiles are strictly inside the grid.
    assert np.all(dist.bin_edges[:, 0] < q[:, 0])
    assert np.all(dist.bin_edges[:, -1] > q[:, -1])


def test_equally_spaced_levels_give_equally_wide_bins():
    """Equal *probability* levels no longer mean equal mass -- they mean equal width.

    On a Normal the mass must concentrate near the mode, so the interior bins
    carry visibly more than ``1 / n_bins`` and the tail bins visibly less.  That
    asymmetry is exactly the sharpness the old equiprobable grid hid in its bin
    spacing.
    """
    alphas = np.linspace(0.001, 0.999, 51)
    q = _normal_quantiles(alphas, [0.0, 3.0], [1.0, 0.5])
    dist = quantiles_to_distribution(q, alphas)

    widths = np.diff(dist.bin_edges, axis=1)
    spread = (widths.max(axis=1) - widths.min(axis=1)) / widths.mean(axis=1)
    assert np.all(spread < 1e-12), "widths, not masses, are the uniform thing now"

    uniform = 1.0 / (len(alphas) - 1)
    n_bins = dist.probas.shape[1]
    for i in range(dist.probas.shape[0]):
        centre = dist.probas[i, n_bins // 2]
        assert centre > 2 * uniform, f"row {i}: mode bin {centre} not concentrated"
        assert dist.probas[i, 0] < uniform, f"row {i}: tail bin not depleted"
    np.testing.assert_allclose(dist.probas.sum(axis=1), 1.0, atol=1e-12)


def test_unequally_spaced_levels_give_the_interpolated_masses():
    """The grid and masses are the monotone-cubic CDF increments over the bins.

    ``alphas = [.1, .2, .5, .9]`` at ``q = [0, 1, 2, 3]``.  The support extends to
    ``z_min = 0 - (1 - 0) = -1`` and ``z_max = 3 + (3 - 2) = 4``, so the 3 bins cut
    from ``[-1, 4]`` have edges ``-1, 2/3, 7/3, 4``.  The CDF
    ``(-1,0), (0,.1), (1,.2), (2,.5), (3,.9), (4,1)`` is interpolated at those
    edges with the same monotone shape-preserving cubic the wrapper uses
    (:func:`_monotone_cdf_at`) and differenced into the masses -- not the straight
    line the old scheme used.  The edges are unchanged (they are set by the
    support, not the interpolant).
    """
    alphas = np.array([0.1, 0.2, 0.5, 0.9])
    q = np.array([[0.0, 1.0, 2.0, 3.0]])
    dist = quantiles_to_distribution(q, alphas)

    np.testing.assert_allclose(dist.bin_edges[0], [-1.0, 2 / 3, 7 / 3, 4.0], rtol=1e-12)

    edges = dist.bin_edges[0]
    x = np.concatenate([[edges[0]], q[0], [edges[-1]]])
    y = np.concatenate([[0.0], alphas, [1.0]])
    expected = np.maximum(np.diff(_monotone_cdf_at(x, y, edges)), 0.0)
    expected /= expected.sum()
    np.testing.assert_allclose(dist.probas[0], expected, rtol=1e-12)


def test_pmf_is_valid():
    rng = np.random.default_rng(0)
    q = np.sort(rng.normal(size=(20, len(ALPHAS_9))) * 3.0, axis=1)
    dist = quantiles_to_distribution(q, ALPHAS_9)

    assert np.all(dist.probas >= 0.0)
    assert np.all(np.isfinite(dist.probas))
    assert np.all(np.isfinite(dist.bin_edges))
    np.testing.assert_allclose(dist.probas.sum(axis=1), 1.0, atol=1e-12)
    # Reconstructed CDF is non-decreasing.
    cdf = np.cumsum(dist.probas, axis=1)
    assert np.all(np.diff(cdf, axis=1) >= -1e-15)


# ---------------------------------------------------------------------------
# Correctness of the CDF re-sampling
# ---------------------------------------------------------------------------

def test_masses_are_exactly_the_interpolated_cdf_increments():
    """The masses are the monotone-cubic CDF increments -- nothing dropped.

    This pins the construction to the bit: build the anchored quantile CDF by
    hand, interpolate it at the output edges with the *same* monotone cubic the
    wrapper uses (:func:`_monotone_cdf_at`), difference it, and it must equal
    ``probas`` exactly.  Because ``alpha = 0`` / ``alpha = 1`` sit on the extended
    ends, the increments already sum to 1 and the wrapper only normalises by that
    unit total.
    """
    rng = np.random.default_rng(1)
    alphas = np.linspace(0.02, 0.98, 25)
    q = np.sort(rng.normal(loc=rng.normal(size=(6, 1)), scale=2.0, size=(6, 25)), axis=1)

    dist = quantiles_to_distribution(q, alphas)

    for i in range(q.shape[0]):
        edges = dist.bin_edges[i]
        x = np.concatenate([[edges[0]], q[i], [edges[-1]]])
        y = np.concatenate([[0.0], alphas, [1.0]])
        expected = np.maximum(np.diff(_monotone_cdf_at(x, y, edges)), 0.0)
        expected /= expected.sum()
        np.testing.assert_allclose(dist.probas[i], expected, rtol=0, atol=1e-12)


def test_cdf_at_the_quantiles_converges_to_the_levels():
    """``F_hat(q_k) -> alpha_k`` as the quantile grid is refined.

    The output CDF is piecewise linear on the *regular* grid, so it only agrees
    with the model's own levels up to the resampling error -- which shrinks as the
    quantile grid gets denser relative to the bin width.  This is the price of
    trading equal probability for strictly positive widths, and it falls off
    fast: ~3e-2 at 9 levels, ~6e-4 at 51 and beyond.
    """
    errors = []
    for k in (9, 51, 199):
        alphas = np.linspace(0.02, 0.98, k)
        q = _normal_quantiles(alphas, [0.0], [1.0])
        dist = quantiles_to_distribution(q, alphas)
        cdf = np.concatenate([[0.0], np.cumsum(dist.probas[0])])
        got = np.interp(q[0], dist.bin_edges[0], cdf)
        errors.append(float(np.max(np.abs(got - alphas))))

    assert errors[0] < 5e-2, errors
    assert errors[1] < errors[0] / 10, errors
    assert errors[2] < 2e-3, errors


def test_recovers_normal_cdf_and_moments():
    alphas = np.linspace(0.001, 0.999, 199)
    loc, scale = 2.5, 1.5
    q = _normal_quantiles(alphas, [loc], [scale])
    dist = quantiles_to_distribution(q, alphas)

    x = np.linspace(loc - 3 * scale, loc + 3 * scale, 101)
    np.testing.assert_allclose(_cdf_at(dist, x), norm.cdf(x, loc, scale), atol=5e-3)

    assert dist.mean[0] == pytest.approx(loc, abs=0.05)
    var = np.sum(dist.probas[0] * (dist.bin_midpoints[0] - dist.mean[0]) ** 2)
    assert np.sqrt(var) == pytest.approx(scale, rel=0.1)


def test_cdf_error_decreases_with_more_quantiles():
    """Finer quantile grids => the reconstructed CDF converges to the truth.

    The residual floor is the dropped tail mass, not the bin width: with levels
    starting at ``alpha_0 = 1 / (k + 1)`` the renormalisation shifts the whole
    CDF by about ``alpha_0``, so convergence is driven by the outermost level.
    """
    x = np.linspace(-3.0, 3.0, 201)
    errors, floors = [], []
    for k in (9, 33, 129):
        alphas = np.linspace(1 / (k + 1), k / (k + 1), k)
        q = _normal_quantiles(alphas, [0.0], [1.0])
        dist = quantiles_to_distribution(q, alphas)
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

    dist = quantiles_to_distribution(q, alphas)
    ref = quantiles_to_distribution(np.sort(q, axis=1), np.sort(alphas))

    np.testing.assert_allclose(dist.probas, ref.probas)
    np.testing.assert_allclose(dist.bin_edges, ref.bin_edges)


def test_non_finite_quantiles_are_sanitized_into_y_range():
    q = np.array([[np.nan, 0.5, np.inf], [-np.inf, 0.2, 0.9]])
    dist = quantiles_to_distribution(q, np.array([0.25, 0.5, 0.75]), y_range=(0.0, 1.0))

    assert np.all(np.isfinite(dist.bin_edges))
    assert np.all(np.isfinite(dist.probas))
    np.testing.assert_allclose(dist.probas.sum(axis=1), 1.0, atol=1e-12)


def test_tied_quantiles_stay_well_defined():
    """Tied quantiles would collapse bins to zero width; regridding prevents it.

    This is the representation's one hazard.  Rather than using the tied quantiles
    as edges, the CDF they define is resampled onto an *equally spaced* grid over
    the same (tail-extended) support -- equal width instead of equal probability
    is what keeps every width strictly positive, so nothing reaches the metrics as
    ``p / 0``.  A tie becomes an atom: its whole mass lands in the single output
    bin that ``metrics`` scores that value against.
    """
    from scoringbench.univariate.metrics import compute_metrics

    alphas = np.linspace(0.01, 0.99, 30)
    q = np.full((3, 30), 7.0)
    q[1] = np.concatenate([np.full(15, 1.0), np.full(15, 2.0)])  # two atoms
    q[2] = np.linspace(6.0, 8.0, 30)

    dist = quantiles_to_distribution(q, alphas)

    widths = np.diff(dist.bin_edges, axis=1)
    assert np.all(widths > 0.0), "no zero-width bin may survive construction"
    # The repair only moves interior edges: the support of each row is unchanged.
    assert dist.bin_edges[1, 0] == pytest.approx(1.0)
    assert dist.bin_edges[1, -1] == pytest.approx(2.0)
    np.testing.assert_allclose(dist.probas.sum(axis=1), 1.0, atol=1e-12)
    # The fully tied row gets a nominal span so it still has a support.
    assert dist.mean[0] == pytest.approx(7.0, abs=1e-3)

    metrics = compute_metrics(dist, np.array([7.0, 1.5, 7.0]))
    for key, value in metrics.items():
        if value is not None:
            assert np.isfinite(value), f"{key} = {value}"


def test_single_quantile_level():
    dist = quantiles_to_distribution(np.array([[3.0], [0.0]]), np.array([0.5]))

    assert dist.probas.shape == (2, 1)
    assert np.all(np.diff(dist.bin_edges, axis=1) > 0)
    np.testing.assert_allclose(dist.probas.sum(axis=1), 1.0, atol=1e-12)


def test_explicit_mean_overrides_pmf_mean():
    q = _normal_quantiles(ALPHAS_9, [0.0, 5.0], [1.0, 2.0])
    supplied = np.array([-1.0, 42.0])

    dist = quantiles_to_distribution(q, ALPHAS_9, mean=supplied)
    np.testing.assert_allclose(dist.mean, supplied)

    pmf_mean = quantiles_to_distribution(q, ALPHAS_9).mean
    np.testing.assert_allclose(
        pmf_mean, np.sum(dist.probas * dist.bin_midpoints, axis=1)
    )


def test_rows_are_independent():
    """Row i's output must not depend on the other rows in the batch."""
    q = _normal_quantiles(ALPHAS_9, [0.0, 500.0, -20.0], [1.0, 0.01, 7.0])
    batch = quantiles_to_distribution(q, ALPHAS_9)

    for i in range(q.shape[0]):
        single = quantiles_to_distribution(q[i : i + 1], ALPHAS_9)
        np.testing.assert_allclose(batch.probas[i], single.probas[0], atol=1e-12)
        np.testing.assert_allclose(batch.bin_edges[i], single.bin_edges[0], rtol=1e-12)


def test_metrics_are_finite_on_the_output():
    from scoringbench.univariate.metrics import compute_metrics

    rng = np.random.default_rng(3)
    alphas = np.linspace(0.01, 0.99, 50)
    y = rng.normal(size=40)
    q = _normal_quantiles(alphas, y, np.full(40, 1.0))
    dist = quantiles_to_distribution(q, alphas)

    metrics = compute_metrics(dist, y)
    for key, value in metrics.items():
        if value is not None:
            assert np.isfinite(value), f"{key} = {value}"
