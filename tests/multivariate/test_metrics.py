"""Comprehensive tests for the multivariate sample-based scoring rules.

Coverage
--------
* Output contract: keys present, all finite, all plain floats.
* Energy score:
    - non-negativity (proper-forecast lower bound, clamped),
    - propriety in expectation (true forecaster beats a mis-located one),
    - analytic value for a known ensemble,
    - translation invariance of ES(β=1),
    - permutation invariance across coordinates.
* Variogram score:
    - zero when the forecast matches observed differences exactly,
    - propriety (correct dependence beats mis-specified dependence),
    - permutation invariance across coordinates.
* Dawid–Sebastiani:
    - matches the closed form (y-μ)ᵀΣ⁻¹(y-μ) + logdet Σ on a fixed ensemble,
    - propriety (well-located ensemble beats a shifted one) in expectation.
* Point metrics: MAE/RMSE against hand-computed Euclidean errors.
"""

from __future__ import annotations

import numpy as np
import pytest

from scoringbench.multivariate.metrics import (
    ENERGY_BETAS,
    SCORING_RULE_KEYS,
    VARIOGRAM_ORDERS,
    compute_metrics,
    compute_point_metrics,
    compute_scoring_rules,
)
from scoringbench.multivariate.prediction import MultivariateSamplePrediction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_pred(samples: np.ndarray) -> MultivariateSamplePrediction:
    return MultivariateSamplePrediction(samples=np.asarray(samples, dtype=float))


def gaussian_ensemble(mu, cov, m, n_test, seed=0):
    """Draw an (n_test, m, d) ensemble from N(mu, cov) for each instance."""
    rng = np.random.default_rng(seed)
    mu = np.atleast_2d(mu)
    d = mu.shape[-1]
    out = np.empty((n_test, m, d))
    for t in range(n_test):
        out[t] = rng.multivariate_normal(mu[t % mu.shape[0]], cov, size=m)
    return out


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

def test_metric_keys_present_and_finite():
    rng = np.random.default_rng(0)
    samples = rng.normal(size=(8, 50, 3))
    y = rng.normal(size=(8, 3))
    m = compute_metrics(make_pred(samples), y)

    for key in SCORING_RULE_KEYS:
        assert key in m, f"missing scoring-rule key {key}"
    assert "mae" in m and "rmse" in m
    for k, v in m.items():
        assert isinstance(v, float), f"{k} is not a float"
        assert np.isfinite(v), f"{k} is not finite"


def test_energy_and_variogram_key_naming():
    rng = np.random.default_rng(1)
    m = compute_scoring_rules(make_pred(rng.normal(size=(3, 20, 2))), rng.normal(size=(3, 2)))
    for b in ENERGY_BETAS:
        assert f"energy_score_beta_{b:g}" in m
    for p in VARIOGRAM_ORDERS:
        assert f"variogram_score_p_{p:g}" in m
    assert "dawid_sebastiani" in m


# ---------------------------------------------------------------------------
# Energy score
# ---------------------------------------------------------------------------

def test_energy_score_non_negative():
    rng = np.random.default_rng(2)
    samples = rng.normal(size=(10, 60, 3))
    y = rng.normal(size=(10, 3))
    m = compute_scoring_rules(make_pred(samples), y)
    for b in ENERGY_BETAS:
        assert m[f"energy_score_beta_{b:g}"] >= 0.0


def test_energy_score_analytic_two_point_ensemble():
    """Deterministic ensemble -> hand-computable energy score (β=1).

    Ensemble draws {a, b}, observation y.
      term1 = ½(‖a−y‖ + ‖b−y‖)
      term2 (fair) = ‖a−b‖   (two ordered off-diagonal pairs / (2·1))
      ES = term1 − ½·term2
    """
    a = np.array([0.0, 0.0])
    b = np.array([2.0, 0.0])
    y = np.array([1.0, 0.0])
    samples = np.stack([a, b])[None, :, :]  # (1, 2, 2)

    term1 = 0.5 * (np.linalg.norm(a - y) + np.linalg.norm(b - y))  # 0.5*(1+1)=1
    term2 = np.linalg.norm(a - b)  # 2
    expected = term1 - 0.5 * term2  # 1 - 1 = 0

    got = compute_scoring_rules(make_pred(samples), y[None, :])["energy_score_beta_1"]
    assert got == pytest.approx(max(expected, 0.0), abs=1e-9)


def test_energy_score_is_proper_in_expectation():
    """True forecaster should score lower than a badly mis-located one."""
    d = 3
    cov = np.eye(d)
    n_test = 200
    y = np.zeros((n_test, d))  # observations at origin

    true_ens = gaussian_ensemble(np.zeros(d), cov, m=80, n_test=n_test, seed=10)
    bad_ens = gaussian_ensemble(np.full(d, 3.0), cov, m=80, n_test=n_test, seed=11)

    es_true = compute_scoring_rules(make_pred(true_ens), y)["energy_score_beta_1"]
    es_bad = compute_scoring_rules(make_pred(bad_ens), y)["energy_score_beta_1"]
    assert es_true < es_bad


def test_energy_score_translation_invariant():
    """ES(β=1) is invariant to a common shift of forecast and observation."""
    rng = np.random.default_rng(3)
    samples = rng.normal(size=(6, 40, 3))
    y = rng.normal(size=(6, 3))
    shift = np.array([5.0, -2.0, 1.0])

    base = compute_scoring_rules(make_pred(samples), y)["energy_score_beta_1"]
    shifted = compute_scoring_rules(
        make_pred(samples + shift), y + shift
    )["energy_score_beta_1"]
    assert base == pytest.approx(shifted, rel=1e-9)


def test_energy_score_coordinate_permutation_invariant():
    """Euclidean norm is invariant under a shared coordinate permutation."""
    rng = np.random.default_rng(4)
    samples = rng.normal(size=(5, 40, 4))
    y = rng.normal(size=(5, 4))
    perm = [2, 0, 3, 1]

    base = compute_scoring_rules(make_pred(samples), y)["energy_score_beta_1"]
    permd = compute_scoring_rules(
        make_pred(samples[:, :, perm]), y[:, perm]
    )["energy_score_beta_1"]
    assert base == pytest.approx(permd, rel=1e-9)


# ---------------------------------------------------------------------------
# Variogram score
# ---------------------------------------------------------------------------

def test_variogram_score_zero_for_perfect_deterministic_forecast():
    """If every draw equals the observation, E|Y_a−Y_b|^p == |y_a−y_b|^p -> VS=0."""
    y = np.array([[1.0, 4.0, -2.0]])
    samples = np.repeat(y[:, None, :], 30, axis=1)  # (1, 30, 3), all equal to y
    m = compute_scoring_rules(make_pred(samples), y)
    for p in VARIOGRAM_ORDERS:
        assert m[f"variogram_score_p_{p:g}"] == pytest.approx(0.0, abs=1e-9)


def test_variogram_score_detects_wrong_dependence():
    """A forecast with correct marginals but wrong cross-dependence scores worse.

    Observations are perfectly correlated (y2 = y1). A forecast that reproduces
    that dependence beats one with independent coordinates.
    """
    n_test = 300
    rng = np.random.default_rng(20)
    z = rng.normal(size=n_test)
    y = np.stack([z, z], axis=1)  # perfectly dependent observations

    # Correct: draws also perfectly dependent around each y.
    dep = np.empty((n_test, 60, 2))
    indep = np.empty((n_test, 60, 2))
    for t in range(n_test):
        common = rng.normal(scale=0.3, size=60)
        dep[t, :, 0] = z[t] + common
        dep[t, :, 1] = z[t] + common  # same noise -> dependent
        indep[t, :, 0] = z[t] + rng.normal(scale=0.3, size=60)
        indep[t, :, 1] = z[t] + rng.normal(scale=0.3, size=60)  # independent noise

    vs_dep = compute_scoring_rules(make_pred(dep), y)["variogram_score_p_0.5"]
    vs_indep = compute_scoring_rules(make_pred(indep), y)["variogram_score_p_0.5"]
    assert vs_dep < vs_indep


def test_variogram_score_coordinate_permutation_invariant():
    rng = np.random.default_rng(21)
    samples = rng.normal(size=(5, 40, 4))
    y = rng.normal(size=(5, 4))
    perm = [3, 1, 0, 2]
    base = compute_scoring_rules(make_pred(samples), y)["variogram_score_p_0.5"]
    permd = compute_scoring_rules(
        make_pred(samples[:, :, perm]), y[:, perm]
    )["variogram_score_p_0.5"]
    assert base == pytest.approx(permd, rel=1e-9)


# ---------------------------------------------------------------------------
# Dawid–Sebastiani
# ---------------------------------------------------------------------------

def test_dawid_sebastiani_matches_closed_form():
    """DSS from a fixed ensemble equals (y−μ)ᵀΣ⁻¹(y−μ) + logdet Σ.

    Σ uses the unbiased (m−1) sample covariance plus the module ridge.
    """
    from scoringbench.multivariate.metrics import _DSS_RIDGE

    rng = np.random.default_rng(30)
    m = 200
    d = 3
    samples = rng.normal(size=(1, m, d))
    y = np.array([[0.5, -1.0, 2.0]])

    mu = samples[0].mean(axis=0)
    centered = samples[0] - mu
    cov = centered.T @ centered / (m - 1) + _DSS_RIDGE * np.eye(d)
    diff = y[0] - mu
    expected = diff @ np.linalg.solve(cov, diff) + np.log(np.linalg.det(cov))

    got = compute_scoring_rules(make_pred(samples), y)["dawid_sebastiani"]
    assert got == pytest.approx(expected, rel=1e-6)


def test_dawid_sebastiani_prefers_well_located_ensemble():
    d = 2
    cov = np.eye(d)
    n_test = 150
    y = np.zeros((n_test, d))
    good = gaussian_ensemble(np.zeros(d), cov, m=120, n_test=n_test, seed=31)
    bad = gaussian_ensemble(np.full(d, 4.0), cov, m=120, n_test=n_test, seed=32)
    dss_good = compute_scoring_rules(make_pred(good), y)["dawid_sebastiani"]
    dss_bad = compute_scoring_rules(make_pred(bad), y)["dawid_sebastiani"]
    assert dss_good < dss_bad


def test_dawid_sebastiani_finite_for_degenerate_ensemble():
    """A (near-)degenerate marginal must not blow up thanks to the ridge."""
    n_test = 4
    d = 3
    rng = np.random.default_rng(33)
    samples = rng.normal(size=(n_test, 50, d))
    samples[:, :, 2] = 7.0  # coordinate 2 is constant -> singular without ridge
    y = rng.normal(size=(n_test, d))
    dss = compute_scoring_rules(make_pred(samples), y)["dawid_sebastiani"]
    assert np.isfinite(dss)


# ---------------------------------------------------------------------------
# Point metrics
# ---------------------------------------------------------------------------

def test_point_metrics_hand_computed():
    y_true = np.array([[0.0, 0.0], [1.0, 1.0]])
    y_pred = np.array([[3.0, 4.0], [1.0, 1.0]])  # errors: 5, 0
    m = compute_point_metrics(y_true, y_pred)
    assert m["mae"] == pytest.approx((5.0 + 0.0) / 2)
    assert m["rmse"] == pytest.approx(np.sqrt((25.0 + 0.0) / 2))


def test_point_metrics_zero_for_exact():
    y = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    m = compute_point_metrics(y, y)
    assert m["mae"] == pytest.approx(0.0)
    assert m["rmse"] == pytest.approx(0.0)


def test_point_metrics_handles_1d_input():
    y_true = np.array([0.0, 2.0])
    y_pred = np.array([1.0, 0.0])  # abs errors 1, 2
    m = compute_point_metrics(y_true, y_pred)
    assert m["mae"] == pytest.approx(1.5)
    assert m["rmse"] == pytest.approx(np.sqrt((1 + 4) / 2))


# ---------------------------------------------------------------------------
# compute_metrics integration
# ---------------------------------------------------------------------------

def test_compute_metrics_merges_point_and_rules():
    rng = np.random.default_rng(40)
    samples = rng.normal(size=(5, 30, 3))
    y = rng.normal(size=(5, 3))
    full = compute_metrics(make_pred(samples), y)
    rules = compute_scoring_rules(make_pred(samples), y)
    points = compute_point_metrics(y, samples.mean(axis=1))
    assert set(full) == set(rules) | set(points)
