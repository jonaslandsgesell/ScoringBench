"""Comprehensive tests for the multivariate sample-based scoring rules.

Coverage
--------
* Output contract: keys present, all finite, all plain floats.
* Energy score:
    - unbiased sample estimates, including legitimate negative values,
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
* Point metrics: mean Euclidean error, RMSE, and element-wise MAE.
"""

from __future__ import annotations

from itertools import product
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import torch

from scoringbench.multivariate.cv import run_fold
from scoringbench.multivariate.estimators import cross_norm_expectation, pairwise_norm_expectation
from scoringbench.multivariate.metrics import (
    ENERGY_BETAS,
    SCORING_RULE_KEYS,
    VARIOGRAM_ORDERS,
    _avg_marginal_energy_scores,
    _dawid_sebastiani,
    _energy_scores,
    _geometric_median,
    _variogram_scores,
    compute_elementwise_mae,
    compute_mean_euclidean_error,
    compute_metrics,
    compute_point_metrics,
    compute_rmse,
    compute_scoring_rules,
)
from scoringbench.multivariate.prediction import MultivariateSamplePrediction
from scoringbench.univariate.metrics import compute_energy_score_histogram_corrected


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
    assert "mean_euclidean_error" in m and "rmse" in m
    assert "elementwise_mae" in m
    assert "mae" not in m
    for k, v in m.items():
        assert isinstance(v, float), f"{k} is not a float"
        assert np.isfinite(v), f"{k} is not finite"


def test_compute_metrics_uses_geometric_median_for_mean_euclidean_error(monkeypatch):
    pred = make_pred(np.array([[[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]]))
    median = np.full((1, 2), 1.0 - 1.0 / np.sqrt(3.0))
    monkeypatch.setattr(
        "scoringbench.multivariate.metrics.compute_scoring_rules",
        lambda pred, y_true: {},
    )

    metrics = compute_metrics(pred, median)

    assert metrics["mean_euclidean_error"] == pytest.approx(0.0, abs=1e-8)
    assert compute_mean_euclidean_error(median, pred.mean) > 0.3
    assert metrics["rmse"] == compute_rmse(median, pred.mean)


def test_compute_metrics_uses_marginal_medians_for_elementwise_mae(monkeypatch):
    pred = make_pred(np.array([
        [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]],
        [[1.0, 4.0], [1.0, 8.0], [9.0, 4.0]],
    ]))
    marginal_medians = np.median(pred.samples, axis=1)
    monkeypatch.setattr(
        "scoringbench.multivariate.metrics.compute_scoring_rules",
        lambda pred, y_true: {},
    )

    metrics = compute_metrics(pred, marginal_medians)

    assert metrics["elementwise_mae"] == pytest.approx(0.0)
    assert metrics["mean_euclidean_error"] > 0.1
    assert metrics["rmse"] == compute_rmse(marginal_medians, pred.mean)
    assert metrics["rmse"] > 0.1


@pytest.mark.parametrize("samples, expected", [
    ([[0.0], [0.0], [9.0]], [0.0]),
    ([[4.0, 7.0]], [4.0, 7.0]),
    ([[3.0, -2.0]] * 3, [3.0, -2.0]),
    ([[0.0, 0.0], [2.0, 4.0]], [1.0, 2.0]),
    ([[0.0, 0.0], [0.0, 0.0], [9.0, 4.0]], [0.0, 0.0]),
    ([[0.0, 0.0], [4.0, 0.0], [-2.0, 1.0]], [0.0, 0.0]),
    ([[0.0, 0.0], [1.0, 2.0], [9.0, 18.0]], [1.0, 2.0]),
])
def test_geometric_median_degenerate_ensembles(samples, expected):
    median = _geometric_median(np.asarray(samples, dtype=float)[None, :, :])
    np.testing.assert_allclose(median[0], expected, atol=1e-9)


@pytest.mark.parametrize("scale", [0.001, 1.0, 1000.0])
def test_geometric_median_rotation_translation_and_scale(scale):
    samples = np.array([[[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]])
    rotation = np.array([[0.6, -0.8], [0.8, 0.6]])
    shift = np.array([5.0, -7.0])
    expected = np.full((1, 2), 1.0 - 1.0 / np.sqrt(3.0)) @ rotation * scale + shift
    median = _geometric_median(samples @ rotation * scale + shift)
    np.testing.assert_allclose(median, expected, rtol=0.0, atol=1e-8 * scale)


def test_geometric_median_minimizes_ensemble_euclidean_error():
    samples = np.random.default_rng(123).exponential(size=(5, 80, 3))
    median = _geometric_median(samples)
    offsets = median[:, None, :] - samples
    distances = np.linalg.norm(offsets, axis=-1)
    gradient = (offsets / distances[:, :, None]).mean(axis=1)
    np.testing.assert_allclose(gradient, 0.0, atol=1e-8)


def test_energy_and_variogram_key_naming():
    rng = np.random.default_rng(1)
    m = compute_scoring_rules(make_pred(rng.normal(size=(3, 20, 2))), rng.normal(size=(3, 2)))
    for b in ENERGY_BETAS:
        assert f"energy_score_beta_{b:g}" in m
    for p in VARIOGRAM_ORDERS:
        assert f"variogram_score_p_{p:g}" in m
    assert "dawid_sebastiani" in m


@pytest.mark.parametrize("degenerate", [False, True])
def test_multivariate_scoring_kernels_match_cpu_reference(scoring_device, degenerate):
    rng = np.random.default_rng(42)
    samples = rng.normal(size=(5, 30, 3))
    targets = rng.normal(size=(5, 3))
    if degenerate:
        samples[:, :, 2] = 7.0
    expected = compute_scoring_rules(make_pred(samples), targets)
    samples_tensor = torch.tensor(samples, dtype=torch.float64, device=scoring_device)
    targets_tensor = torch.tensor(targets, dtype=torch.float64, device=scoring_device)
    assert samples_tensor.device.type == scoring_device.type
    result = {
        **_energy_scores(samples_tensor, targets_tensor, ENERGY_BETAS),
        **_avg_marginal_energy_scores(samples_tensor, targets_tensor, ENERGY_BETAS),
        **_variogram_scores(samples_tensor, targets_tensor, VARIOGRAM_ORDERS),
        **_dawid_sebastiani(samples_tensor, targets_tensor),
    }
    assert result == pytest.approx(expected, rel=1e-10, abs=1e-10)


# ---------------------------------------------------------------------------
# Energy score
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("beta", ENERGY_BETAS)
@pytest.mark.parametrize("n_targets", [1, 2])
def test_energy_score_clamping_against_exact_calibrated_distribution(beta, n_targets, scoring_device):
    """Exhaust all 64 equiprobable (draw1, draw2, observation) combinations.

    All three are independent with P(-1)=P(1)=1/4 and P(0)=1/2. The exact
    expected energy score is half E|X-X'|**beta = 1/4 + 2**beta/16.
    """
    outcomes = np.array(list(product([-1.0, 0.0, 0.0, 1.0], repeat=3)))
    samples = torch.tensor(
        np.repeat(outcomes[:, :2, None], n_targets, axis=2),
        dtype=torch.float64, device=scoring_device,
    )
    targets = torch.tensor(
        np.repeat(outcomes[:, 2, None], n_targets, axis=1),
        dtype=torch.float64, device=scoring_device,
    )
    term1 = cross_norm_expectation(samples, targets, beta)
    term2 = pairwise_norm_expectation(samples, beta)
    assert term1.device.type == scoring_device.type
    assert term2.device.type == scoring_device.type
    assert torch.all(term1 >= 0.0)
    assert torch.all(term2 >= 0.0)

    scale = n_targets ** (beta / 2.0)
    exact_marginal = 0.25 + 2.0 ** beta / 16.0
    exact_joint = scale * exact_marginal
    estimates = term1 - 0.5 * term2
    assert estimates.mean().item() == pytest.approx(exact_joint, abs=1e-12)
    assert _energy_scores(samples, targets, [beta])[f"energy_score_beta_{beta:g}"] == pytest.approx(
        exact_joint, abs=1e-12,
    )
    assert _avg_marginal_energy_scores(samples, targets, [beta])[
        f"average_marginal_energy_score_beta_{beta:g}"
    ] == pytest.approx(exact_marginal, abs=1e-12)

    clipping_bias = scale * max(0.0, 2.0 ** (beta - 1.0) - 1.0) / 16.0
    clipped = estimates.clamp(min=0.0).mean().item()
    assert clipped == pytest.approx(exact_joint + clipping_bias, abs=1e-12)
    if beta > 1.0:
        assert estimates.min().item() < -0.4 * scale
        assert clipped > exact_joint

    edges = torch.tensor([-1.0, -1.0, 0.0, 0.0, 1.0, 1.0], dtype=torch.float64, device=scoring_device)
    masses = torch.tensor([[0.25, 0.0, 0.5, 0.0, 0.25]], dtype=torch.float64, device=scoring_device)
    observations = torch.tensor([-1.0, 0.0, 0.0, 1.0], dtype=torch.float64, device=scoring_device)
    exact_histogram_score = compute_energy_score_histogram_corrected(
        masses.expand(4, -1), edges, observations, [beta],
    )[f"energy_score_beta_{beta}"]
    assert exact_histogram_score >= 0.0
    assert exact_histogram_score == pytest.approx(exact_marginal, abs=1e-12)


@pytest.mark.parametrize("n_targets", [1, 2])
def test_energy_score_retains_negative_estimates(n_targets):
    samples = np.repeat(np.array([[[-1.0], [1.0]]]), n_targets, axis=-1)
    metrics = compute_scoring_rules(make_pred(samples), np.zeros((1, n_targets)))
    expected_marginal = 1.0 - np.sqrt(2.0)

    assert metrics["energy_score_beta_1.5"] == pytest.approx(
        n_targets ** 0.75 * expected_marginal,
    )
    assert metrics["average_marginal_energy_score_beta_1.5"] == pytest.approx(expected_marginal)


@pytest.mark.parametrize("n_targets", [1, 2])
def test_energy_score_is_unbiased_over_two_draw_forecasts(n_targets):
    samples = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    samples = np.repeat(samples[:, :, None], n_targets, axis=-1)
    metrics = compute_scoring_rules(make_pred(samples), np.zeros((4, n_targets)))

    for beta in ENERGY_BETAS:
        expected_marginal = 1.0 - 2.0 ** beta / 4.0
        assert metrics[f"energy_score_beta_{beta:g}"] == pytest.approx(
            n_targets ** (beta / 2.0) * expected_marginal,
        )
        assert metrics[f"average_marginal_energy_score_beta_{beta:g}"] == pytest.approx(
            expected_marginal,
        )


def test_energy_score_non_negative_for_metric_exponents():
    rng = np.random.default_rng(2)
    samples = rng.normal(size=(10, 60, 3))
    y = rng.normal(size=(10, 3))
    m = compute_scoring_rules(make_pred(samples), y)
    for b in (beta for beta in ENERGY_BETAS if beta <= 1.0):
        assert m[f"energy_score_beta_{b:g}"] >= 0.0


def test_energy_score_analytic_two_point_ensemble():
    """Deterministic ensemble -> hand-computable energy score (β=1).

    Ensemble draws {a, b}, observation y.
      term1 = ½(‖a−y‖ + ‖b−y‖)
    term2 = ‖a−b‖   (two ordered off-diagonal pairs / (2·1))
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
    assert got == pytest.approx(expected, abs=1e-9)


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

def test_compute_point_metrics_delegates_to_individual_functions(monkeypatch):
    pred = make_pred(np.ones((2, 3, 2)))
    y_true = np.zeros((2, 2))
    expected = {"mean_euclidean_error": 2.0, "rmse": 3.0, "elementwise_mae": 1.0}
    functions = []
    for name, value in expected.items():
        function = Mock(return_value=value)
        monkeypatch.setattr(f"scoringbench.multivariate.metrics.compute_{name}", function)
        functions.append(function)

    assert compute_point_metrics(pred, y_true) == expected
    for function in functions:
        function.assert_called_once()
        np.testing.assert_array_equal(function.call_args.args[0], y_true)
        np.testing.assert_array_equal(function.call_args.args[1], pred.mean)


@pytest.mark.parametrize("samples", [
    [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]],
    [[0.0, -9.0], [0.0, 0.0], [7.0, 1.0], [100.0, 1.0]],
    [[1.0, 3.0], [1.0, 3.0], [1.0, 3.0]],
    [[2.0, -4.0]],
    [[0.0], [0.0], [9.0]],
], ids=["dependent-skewed", "even-draws", "ties", "single-draw", "one-target"])
def test_marginal_medians_minimize_sample_elementwise_mae(samples):
    """Piecewise-linear L1 loss has a minimum at observed coordinate values."""
    samples = np.asarray(samples)
    marginal_medians = np.median(samples, axis=0)
    median_loss = compute_elementwise_mae(
        samples, np.broadcast_to(marginal_medians, samples.shape),
    )
    coordinates = [np.unique(coordinate) for coordinate in samples.T]
    candidates = np.stack(np.meshgrid(*coordinates, indexing="ij"), axis=-1)
    candidate_losses = [
        compute_elementwise_mae(samples, np.broadcast_to(candidate, samples.shape))
        for candidate in candidates.reshape(-1, samples.shape[-1])
    ]
    assert median_loss == pytest.approx(min(candidate_losses))

    for alternative in (samples.mean(axis=0), _geometric_median(samples[None, :, :])[0]):
        alternative_loss = compute_elementwise_mae(
            samples, np.broadcast_to(alternative, samples.shape),
        )
        assert median_loss <= alternative_loss + 1e-12


def test_point_metrics_hand_computed():
    y_true = np.array([[0.0, 0.0], [1.0, 1.0]])
    y_pred = np.array([[3.0, 4.0], [1.0, 1.0]])  # errors: 5, 0
    assert compute_mean_euclidean_error(y_true, y_pred) == pytest.approx((5.0 + 0.0) / 2)
    assert compute_rmse(y_true, y_pred) == pytest.approx(np.sqrt((25.0 + 0.0) / 2))
    assert compute_elementwise_mae(y_true, y_pred) == pytest.approx(7.0 / 4.0)
    pred = make_pred(y_pred[:, None, :])
    assert compute_point_metrics(pred, y_true) == pytest.approx({
        "mean_euclidean_error": 2.5,
        "rmse": np.sqrt(12.5),
        "elementwise_mae": 1.75,
    })


def test_point_metrics_zero_for_exact():
    y = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert compute_mean_euclidean_error(y, y) == pytest.approx(0.0)
    assert compute_rmse(y, y) == pytest.approx(0.0)
    assert compute_elementwise_mae(y, y) == pytest.approx(0.0)


@pytest.mark.parametrize("true_column", [False, True])
@pytest.mark.parametrize("pred_column", [False, True])
def test_point_metrics_handles_1d_input(true_column, pred_column):
    y_true = np.array([0.0, 2.0])
    y_pred = np.array([1.0, 0.0])  # abs errors 1, 2
    if true_column:
        y_true = y_true[:, None]
    if pred_column:
        y_pred = y_pred[:, None]
    assert compute_mean_euclidean_error(y_true, y_pred) == pytest.approx(1.5)
    assert compute_rmse(y_true, y_pred) == pytest.approx(np.sqrt((1 + 4) / 2))
    assert compute_elementwise_mae(y_true, y_pred) == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# compute_metrics integration
# ---------------------------------------------------------------------------

def test_compute_metrics_merges_point_and_rules():
    rng = np.random.default_rng(40)
    samples = rng.normal(size=(5, 30, 3))
    y = rng.normal(size=(5, 3))
    full = compute_metrics(make_pred(samples), y)
    rules = compute_scoring_rules(make_pred(samples), y)
    assert set(full) == set(rules) | {"mean_euclidean_error", "rmse", "elementwise_mae"}


def test_run_fold_includes_all_scoring_rules_and_point_metrics():
    pred = make_pred(np.random.default_rng(41).normal(size=(2, 8, 2)))
    model = Mock()
    model.predict_ensemble.return_value = pred
    features = pd.DataFrame({"feature": [0.0, 1.0]})
    targets = pd.DataFrame([[0.0, 0.0], [1.0, 1.0]])

    results = run_fold(
        features, features.copy(), targets, targets.copy(),
        {"sample_based": lambda: model}, seed=0,
    )
    metrics = results["sample_based"]
    expected_rules = compute_scoring_rules(pred, targets.to_numpy())
    expected_point = compute_point_metrics(pred, targets.to_numpy())

    assert set(metrics) == set(SCORING_RULE_KEYS) | set(expected_point) | {
        "fit_time", "predict_time", "train_time",
    }
    assert {key: metrics[key] for key in SCORING_RULE_KEYS} == pytest.approx(expected_rules)
    assert {key: metrics[key] for key in expected_point} == pytest.approx(expected_point)
    model.predict_ensemble.assert_called_once()
    model.predict.assert_not_called()


def test_run_fold_requires_predictive_samples():
    model = Mock()
    model.predict_ensemble.side_effect = NotImplementedError("Predictive samples are required")
    features = pd.DataFrame({"feature": [0.0, 1.0]})
    targets = pd.DataFrame([[0.0, 0.0], [1.0, 1.0]])

    results = run_fold(
        features, features.copy(), targets, targets.copy(),
        {"point_only": lambda: model}, seed=0,
    )
    metrics = results["point_only"]

    assert metrics["error_type"] == "NotImplementedError"
    assert metrics["error"] == "Predictive samples are required"
    assert "elementwise_mae" not in metrics
    model.predict.assert_not_called()
