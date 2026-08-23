"""Tests for the cheap zero-width guard on the natively-gridded path.

A model flagged ``is_natively_gridded_model=True`` (e.g. TabPFN's bar-distribution
borders) is trusted to emit a regular histogram grid, so ``DistributionPrediction``
must leave that grid *untouched* -- resampling could only blur it.  But a tied
border would collapse a bin to zero width, whose histogram density ``p / w`` is
``0/0``.  :func:`_sanitize_native_grid` closes that gap:

* a clean grid (all widths > 0) is returned as the *same objects*, so the
  "reaches metrics untouched" contract holds byte for byte;
* only when a tie actually appears does the row fall back to
  :func:`regrid_to_uniform`, which resamples onto a positive-width grid.

Either way the emitted PMF is a valid distribution summing to 1.
"""

from __future__ import annotations

import numpy as np

from scoringbench.univariate.wrappers.base import (
    DistributionPrediction,
    _sanitize_native_grid,
    regrid_to_uniform,
)


# ---------------------------------------------------------------------------
# _sanitize_native_grid: the guard itself
# ---------------------------------------------------------------------------

def test_clean_grid_is_returned_unchanged_same_objects():
    """A positive-width grid passes straight through -- same objects, no copy."""
    edges = np.linspace(-2.0, 2.0, 9)
    probas = np.full((3, 8), 1.0 / 8)

    out_e, out_p = _sanitize_native_grid(edges, probas)

    # The contract is "reaches metrics untouched": identical objects, not merely
    # equal values, so a clean model pays nothing and nothing is silently rebuilt.
    assert out_e is edges
    assert out_p is probas


def test_clean_irregular_but_positive_grid_is_untouched():
    """Widths need only be positive, not uniform, to be left alone."""
    edges = np.array([0.0, 0.5, 1.5, 4.0, 4.1])
    probas = np.array([[0.1, 0.2, 0.3, 0.4]])

    out_e, out_p = _sanitize_native_grid(edges, probas)

    assert out_e is edges
    assert out_p is probas


def test_tied_border_triggers_repair_to_positive_widths():
    """A repeated border collapses a bin; the guard resamples to fix it."""
    # Edges 1.0 and 1.0 tie -> the second bin has zero width.
    edges = np.array([0.0, 1.0, 1.0, 3.0])
    probas = np.array([[0.25, 0.5, 0.25]])

    out_e, out_p = _sanitize_native_grid(edges, probas)

    w = np.diff(np.asarray(out_e), axis=-1)
    assert np.all(w > 0.0)
    # Bin count is preserved by the repair.
    assert np.asarray(out_p).shape[-1] == probas.shape[-1]


def test_repair_matches_regrid_to_uniform():
    """On the repair path the guard is exactly ``regrid_to_uniform``."""
    edges = np.array([0.0, 1.0, 1.0, 3.0])
    probas = np.array([[0.25, 0.5, 0.25]])

    got_e, got_p = _sanitize_native_grid(edges, probas)
    ref_e, ref_p = regrid_to_uniform(edges, probas)

    np.testing.assert_allclose(np.asarray(got_e), np.asarray(ref_e))
    np.testing.assert_allclose(np.asarray(got_p), np.asarray(ref_p))


# ---------------------------------------------------------------------------
# PMF validity: sums to 1 on both paths
# ---------------------------------------------------------------------------

def test_clean_path_pmf_sums_to_one():
    edges = np.linspace(-3.0, 3.0, 11)
    rng = np.random.default_rng(0)
    raw = rng.random((5, 10))
    probas = raw / raw.sum(axis=-1, keepdims=True)

    _, out_p = _sanitize_native_grid(edges, probas)

    np.testing.assert_allclose(np.asarray(out_p).sum(axis=-1), 1.0, atol=1e-12)


def test_repair_path_pmf_sums_to_one():
    edges = np.array([0.0, 1.0, 1.0, 2.0, 2.0, 5.0])  # two collapsed bins
    probas = np.array([[0.2, 0.1, 0.3, 0.15, 0.25]])

    _, out_p = _sanitize_native_grid(edges, probas)

    np.testing.assert_allclose(np.asarray(out_p).sum(axis=-1), 1.0, atol=1e-12)
    assert np.all(np.asarray(out_p) >= 0.0)


def test_repair_path_multirow_pmf_sums_to_one():
    edges = np.array([-1.0, 0.0, 0.0, 1.0])  # shared grid with one tie
    probas = np.array([[0.3, 0.4, 0.3], [0.1, 0.8, 0.1], [0.5, 0.0, 0.5]])

    _, out_p = _sanitize_native_grid(edges, probas)

    np.testing.assert_allclose(np.asarray(out_p).sum(axis=-1), 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Wired into DistributionPrediction.__post_init__
# ---------------------------------------------------------------------------

def test_post_init_leaves_clean_native_grid_untouched():
    edges = np.linspace(0.0, 10.0, 6)
    probas = np.array([[0.1, 0.2, 0.4, 0.2, 0.1]])
    mean = np.array([5.0])

    dp = DistributionPrediction(
        probas=probas,
        bin_edges=edges,
        bin_midpoints=np.zeros(5),  # recomputed in __post_init__
        mean=mean,
        is_natively_gridded_model=True,
    )

    # Untouched grid: same object, and midpoints recomputed to match.
    assert dp.bin_edges is edges
    np.testing.assert_allclose(dp.probas.sum(axis=-1), 1.0, atol=1e-12)
    np.testing.assert_allclose(dp.bin_midpoints, 0.5 * (edges[:-1] + edges[1:]))


def test_post_init_repairs_tied_native_grid_and_keeps_valid_pmf():
    edges = np.array([0.0, 2.0, 2.0, 5.0])  # tie -> zero-width bin
    probas = np.array([[0.3, 0.4, 0.3]])
    mean = np.array([2.5])

    dp = DistributionPrediction(
        probas=probas,
        bin_edges=edges,
        bin_midpoints=np.zeros(3),
        mean=mean,
        is_natively_gridded_model=True,
    )

    w = np.diff(np.asarray(dp.bin_edges), axis=-1)
    assert np.all(w > 0.0)
    np.testing.assert_allclose(dp.probas.sum(axis=-1), 1.0, atol=1e-12)
    # bin_midpoints match the (possibly rewritten) edges.
    expected_mid = 0.5 * (dp.bin_edges[..., :-1] + dp.bin_edges[..., 1:])
    np.testing.assert_allclose(dp.bin_midpoints, expected_mid)
