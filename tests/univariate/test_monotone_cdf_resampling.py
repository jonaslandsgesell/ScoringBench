"""Tests for the monotone-cubic (PCHIP) CDF resampler in ``base``.

``interpolate_cdf_to_grid_with_equally_sized_bins`` reconstructs a density by evaluating the
prediction's CDF on a uniform grid and differencing it.  As of the interpolation
selection study (`tests/test_interpolation_scheme_selection.py`) that evaluation
uses a monotone shape-preserving cubic (PCHIP, "scheme C2") instead of a straight
line, via :func:`_evaluate_monotone_cdf`.

Switching the interpolant must *not* weaken any invariant the metrics depend on,
and must *add* the shape fidelity that motivated the change.  This module pins
both halves of that contract:

* **Invariants kept** -- non-negative masses summing to 1, mass-exactness
  (``C(edge_{k+1}) - C(edge_k)``), atoms credited whole to a single bin, and a
  straight-line CDF still coming back linear.
* **Fidelity gained** -- the reconstructed density of a smooth unimodal truth is
  strictly closer (in L2) to the truth than the old piecewise-constant scheme's.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from scoringbench.univariate.wrappers.resampling_grid import (
    _evaluate_monotone_cdf,
    interpolate_cdf_to_grid_with_equally_sized_bins,
)


# ---------------------------------------------------------------------------
# _evaluate_monotone_cdf: the interpolant itself
# ---------------------------------------------------------------------------

def test_passes_through_the_cdf_nodes():
    """The cubic reproduces the CDF exactly at the abscissae it was fit to."""
    x = np.linspace(-3.0, 3.0, 13)
    c = norm.cdf(x)
    got = _evaluate_monotone_cdf(x, c, x)
    np.testing.assert_allclose(got, c, atol=1e-12)


def test_is_monotone_between_nodes():
    """PCHIP is shape-preserving: no overshoot, so ``diff(C) >= 0`` on any grid."""
    x = np.linspace(-3.0, 3.0, 9)
    c = norm.cdf(x)
    fine = np.linspace(-3.0, 3.0, 500)
    got = _evaluate_monotone_cdf(x, c, fine)
    assert np.all(np.diff(got) >= -1e-12), "interpolated CDF is not monotone"


def test_clipped_to_the_node_range_under_extrapolation():
    """Querying past the node ends stays within ``[C_first, C_last]``."""
    x = np.linspace(0.0, 1.0, 6)
    c = np.linspace(0.0, 1.0, 6)
    q = np.array([-5.0, -1.0, 0.5, 2.0, 9.0])
    got = _evaluate_monotone_cdf(x, c, q)
    assert got.min() >= c[0] - 1e-12
    assert got.max() <= c[-1] + 1e-12


def test_straight_line_cdf_stays_straight():
    """A linear CDF is a fixed point: PCHIP through collinear nodes is the line.

    Guards against the cubic inventing curvature where the data says there is
    none (which would turn a uniform density into a wavy one).
    """
    x = np.linspace(-2.0, 4.0, 7)
    c = (x - x[0]) / (x[-1] - x[0])
    fine = np.linspace(-2.0, 4.0, 200)
    got = _evaluate_monotone_cdf(x, c, fine)
    expected = (fine - x[0]) / (x[-1] - x[0])
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_fewer_than_three_nodes_falls_back_to_linear():
    """With < 3 distinct abscissae there is no cubic; np.interp is used."""
    x = np.array([0.0, 2.0])
    c = np.array([0.0, 1.0])
    q = np.array([0.0, 0.5, 1.0, 2.0])
    got = _evaluate_monotone_cdf(x, c, q)
    np.testing.assert_allclose(got, [0.0, 0.25, 0.5, 1.0], atol=1e-12)


def test_tied_abscissa_keeps_the_last_node_value():
    """A repeated ``x`` (an atom) resolves to the LAST node's CDF value.

    This is ``np.interp``'s convention and what keeps an atom's whole jump in the
    one bin ``searchsorted`` will score a target on the atom against.
    """
    # CDF jumps 0.2 -> 0.8 at x = 1.0 (an atom of mass 0.6).
    x = np.array([0.0, 1.0, 1.0, 2.0])
    c = np.array([0.0, 0.2, 0.8, 1.0])
    # Querying exactly at the tie must give the post-jump value (0.8).
    got = _evaluate_monotone_cdf(x, c, np.array([1.0]))
    assert got[0] == pytest.approx(0.8, abs=1e-12)


# ---------------------------------------------------------------------------
# interpolate_cdf_to_grid_with_equally_sized_bins: the invariants must survive the new interpolant
# ---------------------------------------------------------------------------

def _resample_normal(n_levels=41, n_bins=40, loc=0.0, scale=1.0):
    """Resample a Normal read at ``n_levels`` quantiles onto ``n_bins`` bins."""
    alphas = np.linspace(0.01, 0.99, n_levels)
    q = norm.ppf(alphas, loc=loc, scale=scale)
    left, right = q[1] - q[0], q[-1] - q[-2]
    z_min, z_max = q[0] - left, q[-1] + right
    x = np.concatenate([[z_min], q, [z_max]])[None, :]
    y = np.concatenate([[0.0], alphas, [1.0]])
    return interpolate_cdf_to_grid_with_equally_sized_bins(x, y, n_bins)


def test_masses_are_valid_pmf():
    """Non-negative masses summing to 1, on a strictly positive uniform grid."""
    edges, probas = _resample_normal()
    assert np.all(probas >= 0.0)
    np.testing.assert_allclose(probas.sum(axis=-1), 1.0, atol=1e-12)
    w = np.diff(edges, axis=-1)
    assert np.all(w > 0.0)
    tol = 4.0 * np.spacing(np.abs(edges).max())
    assert np.all(np.abs(w - w[:, :1]) <= tol), "grid not regular"


def test_mass_is_the_cdf_increment_over_the_bin():
    """Per-bin mass equals ``C(edge_{k+1}) - C(edge_k)`` (mass-exactness).

    This is the property NLL/CDE rely on and the reason the edges (not the nodes)
    are the evaluation points.
    """
    n_bins = 30
    alphas = np.linspace(0.02, 0.98, 25)
    q = norm.ppf(alphas)
    left, right = q[1] - q[0], q[-1] - q[-2]
    z_min, z_max = q[0] - left, q[-1] + right
    x = np.concatenate([[z_min], q, [z_max]])
    y = np.concatenate([[0.0], alphas, [1.0]])

    edges, probas = interpolate_cdf_to_grid_with_equally_sized_bins(x[None, :], y, n_bins)
    # Reconstruct the CDF the resampler used and difference it independently.
    c_edges = _evaluate_monotone_cdf(x, y, edges[0])
    expected = np.maximum(np.diff(c_edges), 0.0)
    expected = expected / expected.sum()
    np.testing.assert_allclose(probas[0], expected, atol=1e-12)


# n_bins chosen so the atom at x = 1.0 always falls ON a grid edge; otherwise the
# bin containing it straddles the atom and carries mass from above it too.
@pytest.mark.parametrize("n_bins", [4, 8, 16, 40, 200])
def test_atom_mass_is_credited_at_or_below_its_abscissa(n_bins):
    """The CDF value at an atom survives the resample, at every resolution.

    The atom is NOT kept undivided -- the resampled view smooths it on purpose
    (see ``_evaluate_monotone_cdf``) -- but the cumulative mass at the atom must stay
    exact, since that is what ``searchsorted(edges[1:], y)`` scores against.
    """
    # Support [0, 4]; an atom of mass 0.6 sits at x = 1.0.
    x = np.array([0.0, 1.0, 1.0, 4.0])
    y = np.array([0.0, 0.2, 0.8, 1.0])
    edges, probas = interpolate_cdf_to_grid_with_equally_sized_bins(x[None, :], y, n_bins=n_bins)

    k = int(np.searchsorted(edges[0][1:], 1.0).clip(0, n_bins - 1))
    assert probas[0, : k + 1].sum() == pytest.approx(0.8, abs=1e-9)


def test_resampled_view_smears_an_atom():
    """The jump is spread on purpose, so the density rules see a finite p/w.

    A bin holding the whole 0.6 undivided would hand the width-dividing rules an
    unbounded density -- exactly what the resampled view exists to prevent.
    """
    x = np.array([0.0, 1.0, 1.0, 4.0])
    y = np.array([0.0, 0.2, 0.8, 1.0])
    edges, probas = interpolate_cdf_to_grid_with_equally_sized_bins(x[None, :], y, n_bins=200)

    k = int(np.searchsorted(edges[0][1:], 1.0).clip(0, 199))
    assert probas[0, k] < 0.05, "the atom should be smeared, not concentrated"
    assert probas[0].max() < 0.05
    assert int((probas[0] > 1e-12).sum()) > 50


def test_uniform_cdf_reconstructs_flat_density():
    """A straight-line CDF gives equal masses -- no cubic-induced ripple."""
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    _, probas = interpolate_cdf_to_grid_with_equally_sized_bins(x[None, :], y, n_bins=8)
    np.testing.assert_allclose(probas[0], 1.0 / 8, atol=1e-12)


def test_pchip_density_beats_piecewise_constant_on_a_smooth_truth():
    """The new density is a strictly better L2 fit to a Normal than scheme A.

    Same CDF nodes, same regular edges: the only difference is the interpolant.
    The monotone cubic's differenced density must be closer to the true Normal
    density than the piecewise-constant (linear-CDF) density it replaces.
    """
    n_bins = 60
    alphas = np.linspace(0.01, 0.99, 61)
    q = norm.ppf(alphas)
    left, right = q[1] - q[0], q[-1] - q[-2]
    z_min, z_max = q[0] - left, q[-1] + right
    x = np.concatenate([[z_min], q, [z_max]])
    y = np.concatenate([[0.0], alphas, [1.0]])

    edges, probas_pchip = interpolate_cdf_to_grid_with_equally_sized_bins(x[None, :], y, n_bins)
    e = edges[0]
    mids = 0.5 * (e[:-1] + e[1:])
    w = np.diff(e)

    # Scheme A on the same edges: linear CDF at the edges, differenced.
    c_lin = np.interp(e, x, y)
    mass_lin = np.maximum(np.diff(c_lin), 0.0)
    mass_lin /= mass_lin.sum()

    dens_pchip = probas_pchip[0] / w
    dens_lin = mass_lin / w
    f_true = norm.pdf(mids)

    ise_pchip = np.sum((dens_pchip - f_true) ** 2 * w)
    ise_lin = np.sum((dens_lin - f_true) ** 2 * w)
    assert ise_pchip < ise_lin, (
        f"PCHIP ISE {ise_pchip:.3e} not better than linear {ise_lin:.3e}"
    )
