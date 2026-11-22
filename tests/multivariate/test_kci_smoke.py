"""Smoke tests for KCI, residual dcorr and the scale filter."""

import numpy as np
import pytest

from scoringbench.multivariate.ci_screen import (
    kci_pvalue,
    residual_distance_correlation,
    screen_kci,
)
from scoringbench.multivariate import ci_screen

pytest.importorskip("causallearn")


def test_kci_smoke():
    """Smoke test: KCI p-value is tiny on conditionally dependent data, large on independent."""
    rng = np.random.default_rng(42)
    n = 150
    x = rng.normal(size=(n, 2))
    # Conditionally dependent targets: Y1 = sin(X) + e1, Y2 = cos(X) + 0.8*e1 + e2
    e1, e2 = rng.normal(size=(2, n))
    y1_dep = np.sin(x[:, 0]) + e1
    y2_dep = np.cos(x[:, 1]) + 0.8 * e1 + 0.5 * e2

    p_dep = kci_pvalue(y2_dep, y1_dep, x)
    assert p_dep < 0.05, f"Expected small p-value on dependent data, got {p_dep}"

    # Conditionally independent targets: Y1 = sin(X) + e1, Y2 = cos(X) + e2
    y2_ind = np.cos(x[:, 1]) + e2
    p_ind = kci_pvalue(y2_ind, y1_dep, x)
    assert p_ind > 0.05, f"Expected large p-value on independent data, got {p_ind}"


def test_residual_distance_correlation_smoke():
    rng = np.random.default_rng(42)
    n = 150
    x = rng.normal(size=(n, 2))
    e1, e2 = rng.normal(size=(2, n))
    Y_dep = np.c_[np.sin(x[:, 0]) + e1, np.cos(x[:, 1]) + 0.8 * e1 + 0.5 * e2]
    Y_ind = np.c_[np.sin(x[:, 0]) + e1, np.cos(x[:, 1]) + e2]

    dcor_dep = residual_distance_correlation(Y_dep, x)
    dcor_ind = residual_distance_correlation(Y_ind, x)

    assert dcor_dep > 0.2, f"Expected large residual dcor on dependent data, got {dcor_dep}"
    assert dcor_ind < dcor_dep, f"Independent dcor ({dcor_ind}) should be lower than dependent ({dcor_dep})"


def test_screen_kci_decision():
    rng = np.random.default_rng(42)
    n = 150
    x = rng.normal(size=(n, 2))
    e1, e2 = rng.normal(size=(2, n))
    Y_dep = np.c_[np.sin(x[:, 0]) + e1, np.cos(x[:, 1]) + 0.8 * e1 + 0.5 * e2]

    res = screen_kci(x, Y_dep)
    assert res["selected"] is True
    assert res["kci_p"] < 0.05
    assert res["residual_dcor"] >= 0.175
    assert res["n_targets"] == 2


@pytest.mark.parametrize("ratio,selected", [(499.0, True), (500.0, True), (501.0, False)])
def test_scale_boundary(monkeypatch, ratio, selected):
    values = np.arange(100, dtype=float)
    monkeypatch.setattr(ci_screen, "kci_pvalue", lambda *args: 0.001)
    monkeypatch.setattr(ci_screen, "residual_distance_correlation", lambda *args, **kw: 0.5)
    result = screen_kci(values[:, None], np.column_stack([values, values * ratio]))
    assert result["selected"] is selected
    assert result["scale_ablation"] is (not selected)


@pytest.mark.parametrize("pvalue,effect,reason", [
    (0.05, 0.5, "kci_not_significant"), (0.001, 0.17, "small_effect"),
    (0.001, 0.175, "selected"),
])
def test_sequential_gates(monkeypatch, pvalue, effect, reason):
    values = np.arange(100, dtype=float)
    monkeypatch.setattr(ci_screen, "kci_pvalue", lambda *args: pvalue)
    calls = []
    monkeypatch.setattr(ci_screen, "residual_distance_correlation",
                        lambda *args, **kw: calls.append(True) or effect)
    result = screen_kci(values[:, None], np.column_stack([values, values]))
    assert result["reason"] == reason
    assert bool(calls) == (pvalue < 0.05)


def test_constant_targets_are_rejected():
    rng = np.random.default_rng(12)
    features = rng.normal(size=(100, 20))
    targets = rng.normal(size=(100, 2))
    targets[:, 1] = 1
    assert screen_kci(features, targets)["reason"] == "constant_target"


@pytest.mark.parametrize("n_features", [20, 30, 576])
@pytest.mark.parametrize("target_dim", [2, 3, 5, 10])
def test_target_dimension_does_not_limit_conditioning_features(monkeypatch, n_features, target_dim):
    rng = np.random.default_rng(12)
    features = rng.normal(size=(100, n_features))
    targets = rng.normal(size=(100, target_dim))
    seen_blocks = []

    def kci(target, block, conditioning):
        np.testing.assert_array_equal(conditioning, features)
        np.testing.assert_array_equal(target, targets[:, block.shape[1]])
        seen_blocks.append(block.shape[1])
        return 0.001

    monkeypatch.setattr(ci_screen, "kci_pvalue", kci)
    monkeypatch.setattr(ci_screen, "residual_distance_correlation", lambda *args, **kw: 0.5)
    result = screen_kci(features, targets)
    assert seen_blocks == list(range(1, target_dim))
    assert result["selected"] is True
    assert result["n_features"] == n_features
    assert result["n_targets"] == target_dim
