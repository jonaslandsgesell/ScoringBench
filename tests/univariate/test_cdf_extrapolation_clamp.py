"""Outside its node hull the resampled CDF must be FLAT (``C_first`` / ``C_last``).

``_evaluate_monotone_cdf`` linearly interpolates the CDF nodes.  Past the
outermost node it must stay flat.  An earlier cubic (PCHIP) extrapolated there
swung to ``+-inf`` and inverted the CDF, collapsing a forecast whose hull sits
inside one grid bin to the uniform ``1/n_bins`` fallback; these tests guard
against that regression.

Contract pinned here, against closed forms:

* ``C(q) = C_first`` for ``q <= x_first`` and ``C(q) = C_last`` for ``q >= x_last``,
  at any distance (the clamp, not the cubic, decides the tails);
* inside the hull the result is the linear interpolant, accurate to the
  interpolation error of the known CDF;
* on the resampled grid a forecast sharper than one grid bin keeps all its
  interior mass ``C_last - C_first`` in the bin that holds it; the unreported
  tails ``C_first`` / ``1 - C_last`` are uniform slabs out to the support edge
  (no renormalization, no truncation to the hull).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import cauchy, expon, norm

from scoringbench.univariate.wrappers.resampling_grid import (
    _evaluate_monotone_cdf,
    resample_cdf_nodes_to_support_outer_hull_y_train_set_y_instance_prediction_grid as resample,
)

ALPHAS = np.linspace(0.01, 0.99, 99)

# name -> frozen scipy distribution; covers symmetric, skewed and heavy-tailed CDFs.
DISTS = {
    "normal": norm(loc=8.0, scale=1e-3),
    "expon": expon(loc=8.0, scale=1e-3),
    "cauchy": cauchy(loc=8.0, scale=1e-3),
}


def _nodes(dist):
    return dist.ppf(ALPHAS), ALPHAS


# ---------------------------------------------------------------------------
# Closed-form extrapolation target: flat beyond the hull
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", DISTS)
@pytest.mark.parametrize("dist_in_hulls", [0.5, 3.0, 50.0, 1e3, 1e5])
def test_tails_are_flat_at_any_distance(name, dist_in_hulls):
    x, c = _nodes(DISTS[name])
    hull = x[-1] - x[0]
    left = x[0] - dist_in_hulls * hull
    right = x[-1] + dist_in_hulls * hull
    got = _evaluate_monotone_cdf(x, c, np.array([left, right]))
    np.testing.assert_allclose(got, [c[0], c[-1]], atol=1e-12)


@pytest.mark.parametrize("name", DISTS)
def test_never_inverted_nor_non_monotone_over_a_wide_query_range(name):
    """The CDF over a grid 1e4 hulls wide is non-decreasing and inside [C_first, C_last]."""
    x, c = _nodes(DISTS[name])
    hull = x[-1] - x[0]
    q = np.linspace(x[0] - 1e4 * hull, x[-1] + 1e4 * hull, 4001)
    got = _evaluate_monotone_cdf(x, c, q)
    assert np.all(np.diff(got) >= -1e-12), "CDF decreases somewhere"
    assert got.min() >= c[0] - 1e-12 and got.max() <= c[-1] + 1e-12
    assert np.all(got[q <= x[0]] == pytest.approx(c[0], abs=1e-12))
    assert np.all(got[q >= x[-1]] == pytest.approx(c[-1], abs=1e-12))


def test_tails_are_flat_with_tied_nodes_in_the_middle():
    """Collapsing an atom must not disturb the clamped tails."""
    x = np.array([0.0, 1.0, 1.0, 1.0, 2.0, 5.0])
    c = np.array([0.05, 0.2, 0.5, 0.8, 0.9, 0.95])
    got = _evaluate_monotone_cdf(x, c, np.array([-1e6, 1e6]))
    np.testing.assert_allclose(got, [0.05, 0.95], atol=1e-12)


# ---------------------------------------------------------------------------
# Inside the hull: unchanged interpolant, accurate against the analytic CDF
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["normal", "expon"])
def test_interior_matches_analytic_cdf(name):
    dist = DISTS[name]
    x, c = _nodes(dist)
    q = np.linspace(x[0], x[-1], 2001)
    got = _evaluate_monotone_cdf(x, c, q)
    # Linear-interpolation error on 99 nodes of a smooth CDF is O(h^2 max|f'|/8).
    np.testing.assert_allclose(got, dist.cdf(q), atol=1e-3)


# ---------------------------------------------------------------------------
# Consequence: a forecast sharper than a grid bin keeps its mass in one bin
# ---------------------------------------------------------------------------

def _expected_resampled_cdf(edges, x, c, lo, hi):
    """Closed-form resampled CDF at ``edges``: linear lower slab on ``[lo, x_0]``
    carrying ``c_0``, the linear interpolant inside the hull, linear upper slab
    on ``[x_last, hi]`` carrying ``1 - c_last``."""
    G = np.interp(edges, x, c)
    left, right = edges <= x[0], edges >= x[-1]
    if x[0] > lo:
        G[left] = c[0] * (edges[left] - lo) / (x[0] - lo)
    if hi > x[-1]:
        G[right] = c[-1] + (1.0 - c[-1]) * (edges[right] - x[-1]) / (hi - x[-1])
    return G


@pytest.mark.parametrize("sd", [1e-4, 1e-3, 4e-3])
@pytest.mark.parametrize("name", ["normal", "expon"])
def test_forecast_inside_one_bin_is_not_replaced_by_uniform(name, sd):
    y_lo, y_hi, n_bins = 6.2, 13.9, 200          # bin width 0.0385
    dist = DISTS[name].dist(loc=8.0, scale=sd)
    x, c = dist.ppf(ALPHAS), ALPHAS
    edges, probas = resample(x, c, y_lo, y_hi, n_bins)
    edges, probas = np.atleast_2d(edges)[0], np.atleast_2d(probas)[0]

    expected = np.diff(_expected_resampled_cdf(edges, x, c, y_lo, y_hi))
    # Interpolation error of C at the one interior edge is the only slack.
    np.testing.assert_allclose(probas, expected, atol=5e-3)
    np.testing.assert_allclose(probas.sum(), 1.0, atol=1e-12)
    # The interior mass (0.98) sits in the bin(s) covering the hull, not spread out.
    k = int(np.searchsorted(edges, x[0], side="right") - 1)
    assert probas[k:k + 2].sum() > c[-1] - c[0] - 1e-9
    assert probas.max() > 0.5, "mass spread over the whole grid (uniform fallback)"


def test_bins_outside_the_hull_carry_only_the_uniform_tail_slabs():
    dist = norm(loc=8.0, scale=0.05)              # hull ~ 0.23 = 6 bins
    y_lo, y_hi = 6.2, 13.9
    x, c = dist.ppf(ALPHAS), ALPHAS
    edges, probas = resample(x, c, y_lo, y_hi, 200)
    edges, probas = np.atleast_2d(edges)[0], np.atleast_2d(probas)[0]
    np.testing.assert_allclose(probas.sum(), 1.0, atol=1e-12)
    width = edges[1:] - edges[:-1]
    below, above = edges[1:] <= x[0], edges[:-1] >= x[-1]
    # Fully-outside bins: uniform density c_0/(x_0-lo) resp. (1-c_last)/(hi-x_last).
    np.testing.assert_allclose(probas[below], c[0] * width[below] / (x[0] - y_lo), rtol=1e-9)
    np.testing.assert_allclose(probas[above], (1 - c[-1]) * width[above] / (y_hi - x[-1]), rtol=1e-9)
    np.testing.assert_allclose(probas[below].sum() + probas[above].sum(),
                               c[0] + 1 - c[-1], atol=2 * width.max())
    # Interior bins carry the analytic Gaussian mass, NOT renormalised (levels stay exact).
    inside = (edges[:-1] >= x[0]) & (edges[1:] <= x[-1])
    analytic = np.diff(dist.cdf(edges))[inside]
    np.testing.assert_allclose(probas[inside], analytic, atol=5e-3)
