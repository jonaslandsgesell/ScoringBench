"""Tail atoms on the native quantile grid keep every reported level exact.

A quantile forecast at levels ``alpha_0 < ... < alpha_{K-1}`` only states
``F(q_k) = alpha_k``; the tail masses ``alpha_0`` and ``1 - alpha_{K-1}`` are
known but not where they sit.  ``cdf_nodes_to_native_PMF_grid`` keeps them as
zero-width atoms AT the outermost quantiles instead of renormalizing the interior
increments (which rescaled every level to ``(alpha - alpha_0) / (alpha_{K-1} -
alpha_0)`` and dropped 95% coverage of a calibrated 0.01..0.99 forecast to ~93%).

These tests pin the grid contract and its scoring consequences: exact coverage
for calibrated forecasts, exact interval endpoints, truthful clamping for levels
the model did not report, and finite/consistent scores on the tail atoms.  The
sample-based native grid follows the same rule for its ``1/2n`` Hazen tails.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from scoringbench.univariate.metrics import COVERAGE_LEVELS, compute_scoring_rules
from scoringbench.univariate.wrappers.base import DistributionPrediction
from scoringbench.univariate.wrappers.quantile_based import quantiles_to_distribution
from scoringbench.univariate.wrappers.resampling_grid import cdf_nodes_to_native_PMF_grid

ALPHAS_99 = np.linspace(0.01, 0.99, 99)


def _normal_q(alphas, locs, scales=1.0):
    locs = np.asarray(locs, dtype=float)[:, None]
    scales = np.broadcast_to(np.asarray(scales, dtype=float), locs.shape[:1])[:, None]
    return locs + scales * norm.ppf(np.asarray(alphas))[None, :]


def _stratified_calibrated_targets(n, seed=0):
    """Locations and targets with PIT values exactly on a stratified uniform grid."""
    rng = np.random.default_rng(seed)
    mu = rng.normal(0.0, 3.0, n)
    u = rng.permutation((np.arange(n) + 0.5) / n)
    return mu, mu + norm.ppf(u)


def _score(q, alphas, y, train_range=None):
    if train_range is None:
        train_range = (float(np.min(q)) - 1.0, float(np.max(q)) + 1.0)
    dist = quantiles_to_distribution(q, alphas, train_range=train_range)
    return compute_scoring_rules(dist, np.asarray(y, dtype=float))


# ---------------------------------------------------------------------------
# Grid contract
# ---------------------------------------------------------------------------

def test_tail_masses_become_atoms_at_the_outermost_quantiles():
    q = np.array([[0.0, 1.0, 2.0, 3.0]])
    alphas = np.array([0.1, 0.2, 0.5, 0.9])
    edges, probas = cdf_nodes_to_native_PMF_grid(q, np.broadcast_to(alphas, q.shape))

    np.testing.assert_array_equal(edges, [[0.0, 0.0, 1.0, 2.0, 3.0, 3.0]])
    np.testing.assert_allclose(probas, [[0.1, 0.1, 0.3, 0.4, 0.1]], atol=1e-15)


def test_cdf_hits_every_reported_level_exactly():
    q = _normal_q(ALPHAS_99, [0.0, 5.0], [1.0, 2.0])
    dist = quantiles_to_distribution(q, ALPHAS_99, train_range=(-10.0, 15.0))
    cdf_right = np.cumsum(dist.native.probas, axis=1)

    # Mass through bin j is alpha_j (bin 0 is the lower tail atom), then 1.
    np.testing.assert_allclose(cdf_right[:, :-1], np.broadcast_to(ALPHAS_99, (2, 99)), atol=1e-12)
    np.testing.assert_allclose(cdf_right[:, -1], 1.0, atol=1e-12)


@pytest.mark.parametrize("alphas", [np.linspace(0.0, 1.0, 11), np.linspace(0.0, 0.9, 10),
                                    np.linspace(0.1, 1.0, 10)])
def test_no_atom_is_added_where_the_level_already_reaches_0_or_1(alphas):
    q = _normal_q(np.clip(alphas, 1e-6, 1 - 1e-6), [0.0])
    edges, probas = cdf_nodes_to_native_PMF_grid(q, alphas[None, :])
    expected_bins = len(alphas) - 1 + int(alphas[0] > 0) + int(alphas[-1] < 1)

    assert probas.shape == (1, expected_bins)
    assert edges.shape == (1, expected_bins + 1)
    np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-15)


def test_symmetric_levels_keep_the_mean_and_median_at_the_center():
    q = _normal_q(ALPHAS_99, [2.5], [1.7])
    dist = quantiles_to_distribution(q, ALPHAS_99, train_range=(-5.0, 10.0))

    assert dist.mean[0] == pytest.approx(2.5, abs=1e-12)
    assert dist.median[0] == pytest.approx(2.5, abs=1e-12)


# ---------------------------------------------------------------------------
# Scoring consequences
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alphas", [ALPHAS_99, np.linspace(0.001, 0.999, 999),
                                    np.linspace(0.005, 0.995, 200)],
                         ids=["K99", "K999", "K200"])
def test_calibrated_quantile_forecast_has_nominal_coverage(alphas):
    mu, y = _stratified_calibrated_targets(4000)
    result = _score(_normal_q(alphas, mu), alphas, y)

    for level in COVERAGE_LEVELS:
        assert result[f"coverage_{level}"] == pytest.approx(level / 100.0, abs=2e-3), level


def test_interval_endpoints_are_the_reported_quantiles():
    # Levels on a 0.005 grid contain 0.025 / 0.975 exactly, so the 95% interval is
    # [q(0.025), q(0.975)]: a hair inside is covered, a hair outside is not.
    alphas = np.linspace(0.005, 0.995, 199)
    q = _normal_q(alphas, np.zeros(4))
    lo, hi = norm.ppf(0.025), norm.ppf(0.975)
    eps = 1e-9
    result = _score(q, alphas, [lo + eps, hi - eps, lo - eps, hi + eps])

    assert result["coverage_95"] == pytest.approx(0.5)
    width = hi - lo
    expected = width + (2.0 / 0.05) * eps * 2 / 4
    assert result["interval_score_95"] == pytest.approx(expected, rel=1e-7)


def test_unreported_levels_clamp_to_the_outermost_quantile():
    # A 0.05..0.95 forecast says nothing beyond its 90% band: the 95% interval is
    # that band (q_0.05, q_0.95), so a calibrated target is covered only 90% of
    # the time -- the model is not credited for a tail it did not report.
    alphas = np.linspace(0.05, 0.95, 19)
    mu, y = _stratified_calibrated_targets(4000, seed=1)
    result = _score(_normal_q(alphas, mu), alphas, y)

    assert result["coverage_90"] == pytest.approx(0.90, abs=2e-3)
    assert result["coverage_95"] == pytest.approx(0.90, abs=2e-3)


def test_calibrated_quantile_forecast_has_uniform_pit():
    mu, y = _stratified_calibrated_targets(4000, seed=2)
    result = _score(_normal_q(ALPHAS_99, mu), ALPHAS_99, y)

    # Stratified PITs; targets below q_0.01 get PIT 0 (support clamp), so the KS
    # statistic is bounded by that 1% tail plus interpolation error.
    assert result["pit_ks_stat"] < 0.012


def test_crps_converges_to_the_closed_form_normal_crps():
    mu, y = _stratified_calibrated_targets(4000, seed=3)
    z = y - mu
    exact = np.mean(z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    alphas = np.linspace(0.001, 0.999, 999)
    result = _score(_normal_q(alphas, mu), alphas, y)

    assert result["crps"] == pytest.approx(exact, rel=1e-3)
    assert result["energy_score_beta_1.0"] == pytest.approx(result["crps"], rel=1e-9)


@pytest.mark.parametrize("y", [[-50.0, 50.0, 0.0], [norm.ppf(0.01), norm.ppf(0.99), 0.0]],
                         ids=["outside-support", "on-tail-atoms"])
def test_targets_beyond_or_on_the_tail_atoms_score_finitely(y):
    q = _normal_q(ALPHAS_99, np.zeros(3))
    result = _score(q, ALPHAS_99, y, train_range=(-3.0, 3.0))

    nonfinite = {k: v for k, v in result.items() if v is not None and not np.isfinite(v)}
    assert not nonfinite
    assert result["coverage_95"] == pytest.approx(1 / 3)
    assert 0.0 <= result["pit_ks_stat"] <= 1.0


# ---------------------------------------------------------------------------
# Sample-based native grid: Hazen tails 1/2n as atoms at the min / max draw
# ---------------------------------------------------------------------------

def _hazen_draws(mu, n_draws):
    """Draws at the exact Hazen positions, so eCDF node j sits at level (j-0.5)/n."""
    return np.asarray(mu)[:, None] + norm.ppf((np.arange(n_draws) + 0.5) / n_draws)[None, :]


def test_sample_grid_keeps_hazen_tails_as_atoms():
    n_draws, n_bins = 100, 50
    draws = _hazen_draws([0.0, 4.0], n_draws)
    dist = DistributionPrediction.from_samples(draws, n_bins=n_bins, train_range=(-5.0, 9.0))
    edges, probas = dist.native.bin_edges, dist.native.probas

    assert probas.shape == (2, n_bins + 2)
    assert edges.shape == (2, n_bins + 3)
    np.testing.assert_allclose(edges[:, 0], edges[:, 1])
    np.testing.assert_allclose(edges[:, -1], edges[:, -2])
    np.testing.assert_allclose(edges[:, 0], draws[:, 0])
    np.testing.assert_allclose(edges[:, -1], draws[:, -1])
    np.testing.assert_allclose(probas[:, 0], 0.5 / n_draws, atol=1e-15)
    np.testing.assert_allclose(probas[:, -1], 0.5 / n_draws, atol=1e-15)
    np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(probas >= 0.0)


@pytest.mark.parametrize("n_draws", [100, 300])
def test_calibrated_sample_forecast_has_nominal_coverage(n_draws):
    mu, y = _stratified_calibrated_targets(4000, seed=4)
    dist = DistributionPrediction.from_samples(
        _hazen_draws(mu, n_draws), n_bins=400, train_range=(float(y.min()), float(y.max())))
    result = compute_scoring_rules(dist, y)

    for level in COVERAGE_LEVELS:
        assert result[f"coverage_{level}"] == pytest.approx(level / 100.0, abs=3e-3), level


def test_point_mass_samples_stay_a_point_mass():
    draws = np.full((2, 50), 3.0)
    dist = DistributionPrediction.from_samples(draws, n_bins=10, train_range=(0.0, 6.0))
    result = compute_scoring_rules(dist, np.array([3.0, 4.0]))

    np.testing.assert_allclose(dist.native.bin_edges, 3.0)
    np.testing.assert_allclose(dist.native.probas.sum(axis=1), 1.0, atol=1e-12)
    assert result["crps"] == pytest.approx(0.5, abs=1e-12)
    assert result["coverage_95"] == pytest.approx(0.5)
