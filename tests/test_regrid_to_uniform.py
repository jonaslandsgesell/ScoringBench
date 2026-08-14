"""Tests for the CDF-resampling regridder in ``scoringbench.wrappers.base``.

Why this module exists
----------------------
Every wrapper hands the metrics a histogram: a PMF plus bin edges.  When the
edges come from predicted quantiles, a target with repeated values makes adjacent
quantiles coincide, and the resulting bin has *zero width*.  That is fatal
downstream rather than merely ugly:

* ``metrics.unified_bin_density`` forms ``p_k / w_k``, which is undefined at
  ``w_k = 0`` and explodes as ``w_k -> 0``;
* ``compute_dpd_scores`` clamps the density at ``eps = 100 * finfo.eps`` and, for
  ``beta ~ 0``, returns ``-log(g_y)``, so a collapsed bin turns into a huge but
  finite log score that silently dominates a benchmark average.

``regrid_to_uniform`` removes the failure at the source by re-expressing the same
distribution on an *equally spaced* grid: it reads the histogram's CDF, extends
the support by one local spacing per side (TabICL's convention), and resamples
that CDF at the edges of a regular grid with the same number of bins.  Equal
*width* instead of equal *probability* is exactly what makes every width strictly
positive no matter how badly the input ties.

The invariants asserted here are therefore the contract the metrics rely on:
strictly positive widths, mass conserved to 1, non-negative masses, bin count
preserved, and idempotence (a grid already regular is returned untouched, which is
what lets the quantile/sample constructors interpolate once and not again inside
``DistributionPrediction.__post_init__``).
"""

from __future__ import annotations

import numpy as np
import pytest

from scoringbench.wrappers.base import (
    DistributionPrediction,
    regrid_to_uniform,
    regular_support,
)

# Magnitudes chosen to exercise the guard's three regimes: near zero, where an
# absolute pad is the only sensible floor; mid-range; and far from the origin,
# where one ULP is larger than any absolute pad (at 1e9 an ULP is ~1.2e-7) so a
# span-blind floor would let a 200-bin grid round back onto itself.
MAGNITUDES = [0.0, 1e-30, 1e-9, 1.0, 1e3, 1e6, 1e9, 1e12, -1e9, -1.0, 1e300]


def assert_valid(edges, probas, n_bins):
    """Assert the histogram contract the metrics depend on."""
    e = np.atleast_2d(edges)
    p = np.atleast_2d(probas)
    assert np.all(np.isfinite(e)), "non-finite bin edge"
    assert np.all(np.isfinite(p)), "non-finite mass"
    assert p.shape[-1] == n_bins, "bin count changed"
    assert e.shape[-1] == n_bins + 1

    w = np.diff(e, axis=-1)
    # The whole point of the module: no zero-width bin survives, so p_k / w_k in
    # unified_bin_density is always well defined.
    assert np.all(w > 0.0), f"non-positive width {w.min():.3e}"
    assert np.all(p >= 0.0), "negative mass"
    np.testing.assert_allclose(p.sum(axis=-1), 1.0, atol=1e-12)

    # Regular grid: all widths equal to within the round-off of placing an edge
    # at a coordinate (a couple of ULP of the edges, not of the span).
    span = w.sum(axis=-1, keepdims=True)
    tol = 4.0 * np.spacing(np.abs(e).max(axis=-1, keepdims=True))
    assert np.all(np.abs(w - span / n_bins) <= tol), "grid not regular"


# ---------------------------------------------------------------------------
# The collapsed-bin cases that motivated the module
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("coord", MAGNITUDES)
def test_totally_collapsed_grid_gains_width(coord):
    """Every edge identical -- the worst input -- still yields a usable grid.

    This is the limit of a constant target: all 200 predicted quantiles equal, so
    the input has no support at all.  There is no correct width to return, only a
    representable one, and the guard must supply it at the row's own magnitude.
    """
    n_bins = 200
    edges = np.full(n_bins + 1, coord)
    probas = np.full((1, n_bins), 1.0 / n_bins)
    e_out, p_out = regrid_to_uniform(edges, probas)
    assert_valid(e_out, p_out, n_bins)
    # Width must be resolvable at this coordinate, not just nominally nonzero.
    w = np.diff(np.atleast_2d(e_out), axis=-1)
    assert np.all(w > np.spacing(abs(coord))), "width below one ULP of coordinate"


def test_interior_collapse_is_removed_and_atom_not_split():
    """A run of tied interior edges: mass survives whole, the tie does not.

    The tied edges at 1.0 hold an atom of ``0.3 + 0.4 = 0.7``.  The atom must not
    be smeared across output bins, and it must land in the bin that ``metrics``
    will score a target ``y = 1.0`` against -- bins are ``(left, right]`` there,
    and ``np.interp`` credits a tied CDF jump to the *last* matching node, which
    puts the jump on the closing edge of that same bin.

    The bin also legitimately contains the 0.1 from the input's ``(0, 1]``, so the
    exact statement is about the CDF: everything at or below 1.0 sits in bins up
    to and including ``k``, i.e. ``0.1 + 0.7 = 0.8``.
    """
    edges = np.array([0.0, 1.0, 1.0, 1.0, 4.0])
    probas = np.array([[0.1, 0.3, 0.4, 0.2]])
    e_out, p_out = regrid_to_uniform(edges, probas)
    assert_valid(e_out, p_out, 4)

    e2 = np.atleast_2d(e_out)[0]
    k = int(np.searchsorted(e2[1:], 1.0).clip(0, 3))
    # The atom is intact inside bin k (no leakage into k+1 and beyond).
    assert p_out[0, k] >= 0.7 - 1e-12
    # The CDF at 1.0 is reproduced exactly by the bins up to and including k.
    assert p_out[0, : k + 1].sum() == pytest.approx(0.8, abs=1e-12)


def test_zero_mass_bins_are_preserved_as_zero():
    """Empty input bins must not acquire mass out of nowhere."""
    edges = np.array([0.0, 1.0, 2.0, 3.0])
    probas = np.array([[0.5, 0.0, 0.5]])
    e_out, p_out = regrid_to_uniform(edges, probas)
    assert_valid(e_out, p_out, 3)
    assert p_out.sum() == pytest.approx(1.0, abs=1e-12)


def test_two_ulp_wide_grid_is_accepted():
    """A grid already regular but only ULP-wide is left alone, not "fixed"."""
    edges = np.array([1.0, 1.0 + 2**-52, 1.0 + 2**-51])
    probas = np.array([[0.5, 0.5]])
    e_out, p_out = regrid_to_uniform(edges, probas)
    assert_valid(e_out, p_out, 2)


# ---------------------------------------------------------------------------
# Structure-preserving properties
# ---------------------------------------------------------------------------

def test_regular_input_is_returned_unchanged():
    """Idempotence: an already-regular grid passes straight through.

    This is what allows ``quantiles_to_distribution`` and
    ``samples_to_distribution`` to do their own interpolation and then hand the
    result to ``DistributionPrediction`` without the tails being extended and the
    CDF resampled a second time.
    """
    edges = np.linspace(-3.0, 3.0, 11)
    probas = np.full((4, 10), 0.1)
    e_out, p_out = regrid_to_uniform(edges, probas)
    assert e_out is edges and p_out is probas


def test_shared_grid_stays_one_dimensional():
    """A shared regular grid must keep ``ndim == 1``.

    ``metrics`` branches on ``bin_edges.ndim`` and the 1-D path is the cheap one,
    so silently broadcasting to per-sample edges would be a performance
    regression even though the numbers would agree.
    """
    edges = np.linspace(0.0, 1.0, 6)
    probas = np.full((32, 5), 0.2)
    e_out, _ = regrid_to_uniform(edges, probas)
    assert np.asarray(e_out).ndim == 1


def test_repaired_shared_grid_stays_one_dimensional():
    """The *repair* path must also keep a shared grid shared.

    Above only covers the regular fixed point, which returns its input and so
    cannot regress.  This is the case that actually bit: TabPFN's borders carry a
    tied pair, so every real prediction takes the resampling branch, and
    broadcasting there hands ``metrics`` per-sample edges holding ``n_samples``
    identical rows -- moving the energy score onto its per-sample branch, which
    rebuilds the ``(n_bins, n_bins)`` pairwise-distance matrix once per sample
    (measured ~144x slower) for bit-identical edge values.
    """
    edges = np.array([0.0, 1.0, 1.0, 2.0, 3.0])  # one tie -> not regular
    probas = np.full((32, 4), 0.25)

    e_out, p_out = regrid_to_uniform(edges, probas)

    assert np.asarray(e_out).ndim == 1, "shared grid was broadcast to per-sample"
    assert np.all(np.diff(np.asarray(e_out)) > 0.0)
    assert np.asarray(p_out).shape == probas.shape


def test_repaired_shared_grid_matches_explicit_per_sample_repair():
    """Staying 1-D is a representation change only: the numbers are unchanged.

    Feeding the same tied grid pre-broadcast to per-sample rows must reproduce the
    shared result *exactly*, so the fast path is not trading accuracy for speed.
    """
    edges = np.array([-2.0, -1.0, -1.0, 0.5, 0.5, 0.5, 3.0])  # several ties
    rng = np.random.default_rng(11)
    p = rng.random((8, 6))
    p /= p.sum(axis=-1, keepdims=True)

    e_shared, p_shared = regrid_to_uniform(edges, p)
    e_per, p_per = regrid_to_uniform(np.broadcast_to(edges, (8, 7)), p)

    # Bit-for-bit: same affine map, same interpolant, same difference order.
    np.testing.assert_array_equal(
        np.broadcast_to(np.asarray(e_shared), np.asarray(e_per).shape),
        np.asarray(e_per),
    )
    np.testing.assert_array_equal(np.asarray(p_shared), np.asarray(p_per))


def test_shared_repair_is_idempotent():
    """Repairing the repaired shared grid changes nothing further."""
    edges = np.array([0.0, 0.0, 1.0, 2.0, 2.0, 5.0])
    probas = np.full((4, 5), 0.2)

    e1, p1 = regrid_to_uniform(edges, probas)
    e2, p2 = regrid_to_uniform(e1, p1)

    assert np.asarray(e2).ndim == 1
    np.testing.assert_array_equal(np.asarray(e2), np.asarray(e1))
    np.testing.assert_array_equal(np.asarray(p2), np.asarray(p1))


def test_uniform_pmf_on_regular_grid_is_exactly_preserved():
    """Resampling a uniform density onto its own support reproduces it."""
    n_bins = 8
    edges = np.linspace(-1.0, 1.0, n_bins + 1)
    probas = np.full((1, n_bins), 1.0 / n_bins)
    e_out, p_out = regrid_to_uniform(edges, probas)
    np.testing.assert_allclose(p_out, 1.0 / n_bins, atol=1e-15)
    np.testing.assert_allclose(np.asarray(e_out), edges, atol=0.0)


@pytest.mark.parametrize("n_bins", [1, 2, 3, 7, 200])
def test_random_ties_across_bin_counts(n_bins):
    """Randomised grids with heavy ties, over the bin counts wrappers produce."""
    rng = np.random.default_rng(n_bins)
    rows = 6
    # Draw few distinct values so ties are frequent, then sort into edges.
    raw = rng.integers(0, 3, size=(rows, n_bins + 1)).astype(np.float64)
    edges = np.sort(raw, axis=-1)
    p = rng.random((rows, n_bins))
    p /= p.sum(axis=-1, keepdims=True)
    e_out, p_out = regrid_to_uniform(edges, p)
    assert_valid(e_out, p_out, n_bins)


# ---------------------------------------------------------------------------
# Support convention
# ---------------------------------------------------------------------------

def test_support_extends_by_one_local_spacing():
    """TabICL's rule: one outer gap per side, read off the row itself."""
    x = np.array([[0.0, 1.0, 5.0, 9.0]])
    z_min, z_max = regular_support(x, n_bins=3)
    assert z_min[0, 0] == pytest.approx(-1.0)   # 0 - (1 - 0)
    assert z_max[0, 0] == pytest.approx(13.0)   # 9 + (9 - 5)


def test_support_is_scale_free():
    """Scaling the input scales the support, so no external length is imposed."""
    x = np.array([[0.0, 1.0, 5.0, 9.0]])
    lo1, hi1 = regular_support(x, 3)
    lo2, hi2 = regular_support(x * 1000.0, 3)
    assert lo2[0, 0] == pytest.approx(lo1[0, 0] * 1000.0)
    assert hi2[0, 0] == pytest.approx(hi1[0, 0] * 1000.0)


@pytest.mark.parametrize("coord", MAGNITUDES)
@pytest.mark.parametrize("n_bins", [1, 200, 1000])
def test_degenerate_support_stays_resolvable(coord, n_bins):
    """A tied row gets a span wide enough to cut ``n_bins`` distinct edges."""
    x = np.full((1, 5), coord)
    z_min, z_max = regular_support(x, n_bins)
    span = z_max[0, 0] - z_min[0, 0]
    assert span > 0.0
    assert span >= n_bins * np.spacing(max(abs(coord), abs(z_max[0, 0])))
    # And the grid actually cut from it has no repeated edge.
    grid = z_min[0, 0] + (z_max[0, 0] - z_min[0, 0]) * np.linspace(0, 1, n_bins + 1)
    assert np.all(np.diff(grid) > 0.0)


def test_support_is_centred_when_widened():
    """The degeneracy guard grows the span symmetrically about its own centre."""
    x = np.full((1, 4), 7.0)
    z_min, z_max = regular_support(x, n_bins=10)
    assert 0.5 * (z_min[0, 0] + z_max[0, 0]) == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# Integration through the container
# ---------------------------------------------------------------------------

def test_post_init_regrids_and_is_idempotent():
    """``DistributionPrediction`` regrids on construction, and only once."""
    edges = np.array([0.0, 2.0, 2.0, 2.0, 6.0])
    probas = np.array([[0.25, 0.25, 0.25, 0.25]])
    d = DistributionPrediction(probas, edges, None, np.zeros(1))
    assert_valid(d.bin_edges, d.probas, 4)

    again = DistributionPrediction(d.probas, d.bin_edges, None, np.zeros(1))
    np.testing.assert_array_equal(np.asarray(again.bin_edges), np.asarray(d.bin_edges))
    np.testing.assert_array_equal(again.probas, d.probas)


def test_post_init_recomputes_midpoints_for_new_grid():
    """Midpoints must describe the grid actually stored, not the one passed in."""
    edges = np.array([0.0, 1.0, 1.0, 4.0])
    probas = np.array([[0.2, 0.3, 0.5]])
    stale = np.array([-99.0, -99.0, -99.0])
    d = DistributionPrediction(probas, edges, stale, np.zeros(1))
    e = np.atleast_2d(d.bin_edges)
    expected = 0.5 * (e[..., :-1] + e[..., 1:])
    np.testing.assert_allclose(np.atleast_2d(d.bin_midpoints), expected, atol=0.0)


def test_natively_gridded_model_is_not_touched():
    """TabPFN already predicts on its own regular Riemann borders.

    Its bin edges are part of the trained model's discretisation, so regridding
    them would resample a distribution the model defined exactly -- the flag
    exists to leave that grid, and its object identity, alone.
    """
    edges = np.linspace(-4.0, 4.0, 21)
    probas = np.full((3, 20), 0.05)
    d = DistributionPrediction(
        probas, edges, None, np.zeros(3), is_natively_gridded_model=True
    )
    assert d.bin_edges is edges
    assert d.probas is probas
