"""Density view: unreported tail mass sits as uniform slabs out to the support edge.

A quantile head reports levels ``[alpha_0, alpha_K]`` only; the native view keeps
``alpha_0`` / ``1 - alpha_K`` as atoms at the outermost quantiles.  The density view
spreads the same masses uniformly over ``[L, q_0]`` and ``[q_K, U]`` (``[L, U]`` the
grow-only support) instead of renormalizing the interior, so interior density is
not inflated by ``1 / (alpha_K - alpha_0)`` and in-support targets beyond the hull
keep a positive density.  A tail whose hull already reaches the support edge has
no room for a slab and is folded into the interior as before.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from scoringbench.univariate.metrics import _score_one_view
from scoringbench.univariate.wrappers.base import DistributionPrediction

TRAIN = (-4.0, 4.0)


def _view(prediction):
    view = prediction.resampled
    return np.asarray(view.bin_edges).reshape(-1), np.asarray(view.probas).reshape(-1)


def _binned_cdf_at_edge(edges, probas, value):
    """Binned CDF at ``value``, which the test places exactly on a grid edge."""
    k = int(np.argmin(np.abs(edges - value)))
    assert edges[k] == pytest.approx(value, abs=1e-12)
    return np.r_[0.0, np.cumsum(probas)][k]


def test_tail_masses_are_uniform_slabs_and_interior_is_not_renormalized():
    levels = np.arange(1, 20) / 20
    # Quantiles on multiples of the bin width 8/200, so every node is a grid edge.
    q = np.round(stats.norm.ppf(levels) / 0.04) * 0.04
    edges, probas = _view(DistributionPrediction.from_multi_quantile(q[None, :], levels, train_range=TRAIN))
    assert _binned_cdf_at_edge(edges, probas, q[0]) == pytest.approx(levels[0], abs=1e-12)
    assert _binned_cdf_at_edge(edges, probas, q[-1]) == pytest.approx(levels[-1], abs=1e-12)
    density = probas / np.diff(edges)
    below = edges[1:] <= q[0] + 1e-12
    above = edges[:-1] >= q[-1] - 1e-12
    np.testing.assert_allclose(density[below], levels[0] / (q[0] - TRAIN[0]), rtol=1e-9)
    np.testing.assert_allclose(density[above], (1 - levels[-1]) / (TRAIN[1] - q[-1]), rtol=1e-9)
    exact = np.diff(np.interp(edges, q, levels))
    inside = ~below & ~above
    np.testing.assert_allclose(probas[inside], exact[inside], rtol=1e-9)


def test_tail_is_folded_when_the_hull_reaches_the_support_edge():
    levels = np.array([0.1, 0.5, 0.9])
    q = np.array([-5.0, 0.0, 0.4])          # below the train range: lower edge = q_0; 0.4 is an edge
    edges, probas = _view(DistributionPrediction.from_multi_quantile(q[None, :], levels, train_range=TRAIN))
    assert edges[0] == q[0]
    assert probas.sum() == pytest.approx(1.0)
    assert 1.0 - _binned_cdf_at_edge(edges, probas, q[-1]) == pytest.approx(0.1 / 0.9, rel=1e-9)


def test_sample_density_is_positive_between_train_edge_and_extreme_draw():
    draws = np.random.default_rng(0).normal(scale=0.5, size=(1, 300))
    prediction = DistributionPrediction.from_samples(draws, train_range=TRAIN)
    score = _score_one_view(prediction.resampled, np.array([-3.5]), True, False)
    edges, probas = _view(prediction)
    k = int(np.searchsorted(edges, draws.min())) - 1     # last edge below the min draw
    slab_density = (0.5 / 300) / (draws.min() - TRAIN[0])
    np.testing.assert_allclose(probas[:k] / np.diff(edges)[:k], slab_density, rtol=1e-9)
    assert score["dpd_beta_0.01"] < -1.0     # f(y) > 0, so the point term is active


def test_tied_outermost_quantiles_keep_the_slab_and_the_atom_masses_separate():
    levels = np.array([0.1, 0.2, 0.5, 0.9])
    q = np.array([0.0, 0.0, 1.0, 2.0])      # atom of 0.1 at the lowest quantile; 0.0 and 1.0 are edges
    edges, probas = _view(DistributionPrediction.from_multi_quantile(q[None, :], levels, train_range=TRAIN))
    assert _binned_cdf_at_edge(edges, probas, 0.0) == pytest.approx(0.1, abs=1e-12)
    assert _binned_cdf_at_edge(edges, probas, 1.0) == pytest.approx(0.5, abs=1e-12)
    np.testing.assert_allclose(probas[edges[1:] <= 1e-12] / np.diff(edges)[edges[1:] <= 1e-12], 0.1 / 4.0)


def test_rows_with_and_without_room_are_handled_independently():
    levels = np.array([0.2, 0.5, 0.8])
    q = np.array([[-1.0, 0.0, 1.0], [-6.0, 0.0, 1.0]])
    prediction = DistributionPrediction.from_multi_quantile(q, levels, train_range=TRAIN)
    edges = np.asarray(prediction.resampled.bin_edges)
    probas = np.asarray(prediction.resampled.probas)
    np.testing.assert_allclose(probas.sum(axis=1), 1.0)
    assert edges[0, 0] == TRAIN[0] and edges[1, 0] == -6.0
    below_first = probas[0, edges[0, 1:] <= -1.0 + 1e-12].sum()
    assert below_first == pytest.approx(0.2, abs=0.01)      # slab with room
    # No room below q_0 = -6 (the support starts there): 0.2 folded, upper slab 0.2 / 0.8.
    above_second = probas[1, edges[1, :-1] >= 1.0 - 1e-12].sum()
    assert above_second == pytest.approx(0.25, abs=1e-9)


def _exact_scores(law, y):
    grid = np.linspace(-8, 8, 200001)
    out = {}
    for beta in (0.01, 1.0):
        integral = np.trapezoid(law.pdf(grid) ** (1 + beta), grid)
        out[f"dpd_beta_{beta}"] = float((integral - (1 + 1 / beta) * law.pdf(y) ** beta).mean())
    return out


# Thresholds sit between the renormalized (old) and slab errors measured on this toy:
# old |DPD 0.01 err| = 7.75 / 1.91 / 0.84 and |CDE err| = 0.0104 / 0.0007 / 0.0271.
@pytest.mark.parametrize("source, dpd_tol, cde_tol", [
    ("quantiles_19", 0.5, 0.008),
    ("quantiles_99", 0.1, 0.0007),
    ("samples_300", 0.3, 0.03),
])
def test_density_scores_of_the_true_law_approach_the_exact_scores(source, dpd_tol, cde_tol):
    rng = np.random.default_rng(0)
    law = stats.norm(0, 0.7)
    y = law.rvs(600, random_state=rng)
    if source.startswith("quantiles"):
        k = int(source.split("_")[1])
        levels = np.arange(1, k + 1) / (k + 1)
        prediction = DistributionPrediction.from_multi_quantile(
            np.repeat(law.ppf(levels)[None, :], y.size, 0), levels, train_range=TRAIN)
    else:
        prediction = DistributionPrediction.from_samples(
            law.rvs((y.size, 300), random_state=rng), train_range=TRAIN)
    got = _score_one_view(prediction.resampled, y, True, False)
    exact = _exact_scores(law, y)
    assert abs(got["dpd_beta_0.01"] - exact["dpd_beta_0.01"]) < dpd_tol
    assert abs(got["dpd_beta_1.0"] - exact["dpd_beta_1.0"]) < cde_tol
