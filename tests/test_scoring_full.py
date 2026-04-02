"""Comprehensive scoring and wrapper tests (reduced to avoid duplicates).

Run with:  cd ScoringBench && python -m pytest tests/ -v

This file is derived from the original comprehensive test suite but omits
the small `TestPointMetrics` class which is already covered by
`tests/test_metrics.py` to avoid redundancy.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from numpy.testing import assert_allclose
import warnings
from sklearn.exceptions import UndefinedMetricWarning

from scoringbench.wrappers.base import DistributionPrediction
from scoringbench.metrics import (
    compute_point_metrics,
    compute_scoring_rules,
    compute_metrics,
    ENERGY_BETAS,
)

# If PyTorch is installed but CUDA on this host is broken/unsupported, force
# CPU mode for tests to avoid AcceleratorError during tensor creation.
try:
    import torch as _torch  # optional
    try:
        _torch.cuda.is_available = lambda: False
    except Exception:
        pass
except Exception:
    _torch = None

# Suppress noisy warnings produced on some CI/hosts:
# - Torch emits a UserWarning when computing std over trivial inputs
# - sklearn raises UndefinedMetricWarning for R^2 on <2 samples
warnings.filterwarnings("ignore", message=r".*std.*degrees of freedom.*", category=UserWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_dist(probas, bin_edges, bin_midpoints=None, mean=None):
    """Build a DistributionPrediction, deriving midpoints/mean if omitted."""
    probas = np.asarray(probas, dtype=np.float64)
    bin_edges = np.asarray(bin_edges, dtype=np.float64)
    if bin_midpoints is None:
        if bin_edges.ndim == 1:
            bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
        else:
            bin_midpoints = (bin_edges[:, :-1] + bin_edges[:, 1:]) / 2
    else:
        bin_midpoints = np.asarray(bin_midpoints, dtype=np.float64)
    if mean is None:
        mids = bin_midpoints[None, :] if bin_midpoints.ndim == 1 else bin_midpoints
        mean = (probas * mids).sum(axis=-1)
    else:
        mean = np.asarray(mean, dtype=np.float64)
    return DistributionPrediction(
        probas=probas, bin_edges=bin_edges,
        bin_midpoints=bin_midpoints, mean=mean,
    )


# ---------------------------------------------------------------------------
# 2  Delta (point-mass) distribution — all scores should be ≈ 0
# ---------------------------------------------------------------------------


class TestDeltaDistribution:
    """All probability mass on the bin containing y_true."""

    @pytest.fixture
    def delta(self):
        edges = np.arange(6, dtype=float)
        probas = np.array([[0.0, 0.0, 1.0, 0.0, 0.0]])
        y = np.array([2.5])
        return _make_dist(probas, edges), y

    def test_crps(self, delta):
        r = compute_scoring_rules(*delta)
        assert r["crps"] == pytest.approx(0.0, abs=1e-6)

    def test_log_score(self, delta):
        r = compute_scoring_rules(*delta)
        assert r["log_score"] == pytest.approx(0.0, abs=1e-6)

    def test_sharpness(self, delta):
        r = compute_scoring_rules(*delta)
        assert r["sharpness"] == pytest.approx(0.0, abs=1e-6)

    def test_crls(self, delta):
        r = compute_scoring_rules(*delta)
        assert r["crls"] == pytest.approx(0.0, abs=1e-4)

    def test_energy_scores(self, delta):
        r = compute_scoring_rules(*delta)
        for b in ENERGY_BETAS:
            assert r[f"energy_score_beta_{b}"] == pytest.approx(0.0, abs=1e-6)

    def test_wcrps_variants(self, delta):
        r = compute_scoring_rules(*delta)
        assert r["wcrps_left"] == pytest.approx(0.0, abs=1e-6)
        assert r["wcrps_right"] == pytest.approx(0.0, abs=1e-6)
        assert r["wcrps_center"] == pytest.approx(0.0, abs=1e-6)

    def test_coverage(self, delta):
        r = compute_scoring_rules(*delta)
        assert r["coverage_90"] == pytest.approx(1.0)
        assert r["coverage_95"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 3  Uniform distribution — analytical values
# ---------------------------------------------------------------------------


class TestUniformDistribution:
    @pytest.fixture
    def uniform(self):
        edges = np.arange(5, dtype=float)
        probas = np.array([[0.25, 0.25, 0.25, 0.25]])
        y = np.array([2.0])
        return _make_dist(probas, edges), y

    def test_crps(self, uniform):
        r = compute_scoring_rules(*uniform)
        assert r["crps"] == pytest.approx(0.375, abs=1e-5)

    def test_log_score(self, uniform):
        r = compute_scoring_rules(*uniform)
        assert r["log_score"] == pytest.approx(np.log(4), abs=1e-5)

    def test_sharpness(self, uniform):
        r = compute_scoring_rules(*uniform)
        assert r["sharpness"] == pytest.approx(np.sqrt(1.25), abs=1e-5)

    def test_crls(self, uniform):
        expected = -np.log(0.75) + np.log(2) + (-np.log(0.75))
        r = compute_scoring_rules(*uniform)
        assert r["crls"] == pytest.approx(expected, abs=1e-3)

    def test_coverage(self, uniform):
        r = compute_scoring_rules(*uniform)
        assert r["coverage_90"] == 1.0
        assert r["coverage_95"] == 1.0

    def test_interval_scores(self, uniform):
        r = compute_scoring_rules(*uniform)
        assert r["interval_score_90"] == pytest.approx(4.0, abs=1e-5)
        assert r["interval_score_95"] == pytest.approx(4.0, abs=1e-5)

    def test_energy_beta1_equals_crps(self, uniform):
        r = compute_scoring_rules(*uniform)
        assert r["energy_score_beta_1.0"] == pytest.approx(r["crps"], rel=1e-4)

    def test_wcrps_left(self, uniform):
        expected = 0.0625 * 1 + 0.25 * 4 / 9 + 0.0625 / 9
        r = compute_scoring_rules(*uniform)
        assert r["wcrps_left"] == pytest.approx(expected, abs=1e-5)

    def test_wcrps_right(self, uniform):
        expected = 0.25 / 9 + 0.0625 * 4 / 9
        r = compute_scoring_rules(*uniform)
        assert r["wcrps_right"] == pytest.approx(expected, abs=1e-5)

    def test_wcrps_center(self, uniform):
        expected = 0.25 * 8 / 9 + 0.0625 * 8 / 9
        r = compute_scoring_rules(*uniform)
        assert r["wcrps_center"] == pytest.approx(expected, abs=1e-5)


# ---------------------------------------------------------------------------
# 4  Asymmetric distribution — verify off-center behaviour
# ---------------------------------------------------------------------------


class TestAsymmetricDistribution:
    @pytest.fixture
    def asym(self):
        edges = np.array([0.0, 1.0, 2.0, 3.0])
        probas = np.array([[0.7, 0.2, 0.1]])
        y = np.array([0.5])
        return _make_dist(probas, edges), y

    def test_crps(self, asym):
        r = compute_scoring_rules(*asym)
        assert r["crps"] == pytest.approx(0.10, abs=1e-5)

    def test_log_score(self, asym):
        r = compute_scoring_rules(*asym)
        assert r["log_score"] == pytest.approx(-np.log(0.7), abs=1e-5)

    def test_sharpness(self, asym):
        r = compute_scoring_rules(*asym)
        assert r["sharpness"] == pytest.approx(np.sqrt(0.44), abs=1e-5)


# ---------------------------------------------------------------------------
# 5  Per-sample (2-D) grids
# ---------------------------------------------------------------------------


class TestPerSampleGrid:
    @pytest.fixture
    def persample(self):
        edges = np.array([[0.0, 1.0, 2.0, 3.0],
                          [1.0, 2.0, 3.0, 4.0]])
        probas = np.array([[0.5, 0.3, 0.2],
                           [0.1, 0.2, 0.7]])
        y = np.array([1.5, 3.0])
        return _make_dist(probas, edges), y

    def test_crps(self, persample):
        r = compute_scoring_rules(*persample)
        assert r["crps"] == pytest.approx(0.195, abs=1e-5)

    def test_returns_all_keys(self, persample):
        r = compute_scoring_rules(*persample)
        expected_keys = {
            "crps", "log_score", "sharpness",
            "coverage_90", "interval_score_90",
            "coverage_95", "interval_score_95",
            "crls",
            "wcrps_left", "wcrps_right", "wcrps_center",
            "dispersion",
        }
        expected_keys |= {f"energy_score_beta_{b}" for b in ENERGY_BETAS}
        assert expected_keys == set(r.keys())


# ---------------------------------------------------------------------------
# 7  compute_metrics combines point + scoring rules
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    def test_contains_all_keys(self):
        edges = np.arange(6, dtype=float)
        probas = np.array([[0.0, 0.0, 1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.0, 0.0]])
        y = np.array([2.5, 2.5])
        dist = _make_dist(probas, edges)
        m = compute_metrics(dist, y)
        assert "mae" in m and "rmse" in m and "r2" in m
        assert "crps" in m and "log_score" in m

    def test_mean_used_for_point_metrics(self):
        edges = np.arange(5, dtype=float)
        probas = np.array([[0.25, 0.25, 0.25, 0.25],
                           [0.25, 0.25, 0.25, 0.25]])
        y = np.array([2.0, 2.0])
        dist = _make_dist(probas, edges)
        m = compute_metrics(dist, y)
        assert m["mae"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 8  Sanity / invariant checks
# ---------------------------------------------------------------------------


class TestSanityChecks:
    @pytest.fixture
    def random_dist(self):
        rng = np.random.default_rng(99)
        n, k = 30, 15
        edges = np.linspace(-5, 5, k + 1)
        raw = rng.exponential(size=(n, k))
        probas = raw / raw.sum(axis=1, keepdims=True)
        y = rng.uniform(-4, 4, size=n)
        return _make_dist(probas, edges), y

    def test_crps_nonnegative(self, random_dist):
        r = compute_scoring_rules(*random_dist)
        assert r["crps"] >= -1e-10

    def test_sharpness_nonnegative(self, random_dist):
        r = compute_scoring_rules(*random_dist)
        assert r["sharpness"] >= -1e-10

    def test_coverage_in_range(self, random_dist):
        r = compute_scoring_rules(*random_dist)
        assert 0 <= r["coverage_90"] <= 1
        assert 0 <= r["coverage_95"] <= 1

    def test_energy_scores_nonnegative(self, random_dist):
        r = compute_scoring_rules(*random_dist)
        for b in ENERGY_BETAS:
            assert r[f"energy_score_beta_{b}"] >= -1e-5

    def test_wider_interval_geq_coverage(self, random_dist):
        r = compute_scoring_rules(*random_dist)
        assert r["coverage_95"] >= r["coverage_90"] - 1e-10


# ---------------------------------------------------------------------------
# 9  DistributionPrediction construction
# ---------------------------------------------------------------------------


class TestDistributionPredictionConstruction:
    def test_shapes_shared_grid(self):
        dist = _make_dist(
            probas=np.ones((5, 10)) / 10,
            bin_edges=np.linspace(0, 1, 11),
        )
        assert dist.probas.shape == (5, 10)
        assert dist.bin_edges.shape == (11,)
        assert dist.bin_midpoints.shape == (10,)
        assert dist.mean.shape == (5,)

    def test_shapes_persample_grid(self):
        n, k = 3, 8
        edges = np.tile(np.linspace(0, 1, k + 1), (n, 1))
        dist = _make_dist(
            probas=np.ones((n, k)) / k,
            bin_edges=edges,
        )
        assert dist.probas.shape == (n, k)
        assert dist.bin_edges.shape == (n, k + 1)
        assert dist.bin_midpoints.shape == (n, k)
        assert dist.mean.shape == (n,)

    def test_probas_sum_to_one(self):
        raw = np.random.default_rng(0).exponential(size=(4, 6))
        probas = raw / raw.sum(axis=1, keepdims=True)
        dist = _make_dist(probas, np.linspace(0, 1, 7))
        assert_allclose(dist.probas.sum(axis=1), 1.0, atol=1e-12)

    def test_mean_equals_weighted_midpoints(self):
        rng = np.random.default_rng(1)
        probas = rng.dirichlet(np.ones(5), size=3)
        edges = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
        dist = _make_dist(probas, edges)
        mids = (edges[:-1] + edges[1:]) / 2
        expected_mean = (probas * mids[None, :]).sum(axis=1)
        assert_allclose(dist.mean, expected_mean, atol=1e-12)


# ---------------------------------------------------------------------------
# 10  TabPFN wrapper translation
# ---------------------------------------------------------------------------


class TestTabPFNTranslation:
    def test_predict_distribution(self):
        torch = pytest.importorskip("torch")
        from scoringbench.wrappers.tabpfn import TabPFNWrapper

        wrapper = TabPFNWrapper.__new__(TabPFNWrapper)
        wrapper._torch = torch
        wrapper._device = "cpu"
        wrapper._model = MagicMock()

        logits = torch.tensor([[1.0, 2.0, 3.0, 0.5]])
        borders = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
        criterion = MagicMock()
        criterion.borders = borders
        wrapper._model.predict.return_value = {
            "logits": logits, "criterion": criterion,
        }

        dist = wrapper.predict_distribution(np.zeros((1, 2)))

        expected_probas = torch.softmax(logits, dim=-1).numpy()
        expected_edges = borders.numpy()
        expected_mids = (expected_edges[:-1] + expected_edges[1:]) / 2
        expected_mean = (expected_probas * expected_mids[None, :]).sum(axis=-1)

        assert_allclose(dist.probas, expected_probas, atol=1e-6)
        assert_allclose(dist.bin_edges, expected_edges)
        assert_allclose(dist.bin_midpoints, expected_mids)
        assert_allclose(dist.mean, expected_mean, atol=1e-6)

    def test_probas_sum_to_one(self):
        torch = pytest.importorskip("torch")
        from scoringbench.wrappers.tabpfn import TabPFNWrapper

        wrapper = TabPFNWrapper.__new__(TabPFNWrapper)
        wrapper._torch = torch
        wrapper._device = "cpu"
        wrapper._model = MagicMock()

        logits = torch.randn(5, 10)
        borders = torch.linspace(0, 1, 11)
        criterion = MagicMock()
        criterion.borders = borders
        wrapper._model.predict.return_value = {
            "logits": logits, "criterion": criterion,
        }

        dist = wrapper.predict_distribution(np.zeros((5, 3)))
        assert_allclose(dist.probas.sum(axis=1), 1.0, atol=1e-6)

    def test_shapes_multi_sample(self):
        torch = pytest.importorskip("torch")
        from scoringbench.wrappers.tabpfn import TabPFNWrapper

        wrapper = TabPFNWrapper.__new__(TabPFNWrapper)
        wrapper._torch = torch
        wrapper._device = "cpu"
        wrapper._model = MagicMock()

        n_samples, n_bins = 10, 20
        logits = torch.randn(n_samples, n_bins)
        borders = torch.linspace(-5, 5, n_bins + 1)
        criterion = MagicMock()
        criterion.borders = borders
        wrapper._model.predict.return_value = {
            "logits": logits, "criterion": criterion,
        }

        dist = wrapper.predict_distribution(np.zeros((n_samples, 4)))

        assert dist.probas.shape == (n_samples, n_bins)
        assert dist.bin_edges.shape == (n_bins + 1,)
        assert dist.bin_midpoints.shape == (n_bins,)
        assert dist.mean.shape == (n_samples,)
        assert (dist.probas >= 0).all()
        assert_allclose(dist.probas.sum(axis=1), 1.0, atol=1e-6)

    def test_negative_logits_still_valid(self):
        torch = pytest.importorskip("torch")
        from scoringbench.wrappers.tabpfn import TabPFNWrapper

        wrapper = TabPFNWrapper.__new__(TabPFNWrapper)
        wrapper._torch = torch
        wrapper._device = "cpu"
        wrapper._model = MagicMock()

        logits = torch.tensor([[-100.0, -200.0, 0.0, -50.0]])
        borders = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
        criterion = MagicMock()
        criterion.borders = borders
        wrapper._model.predict.return_value = {
            "logits": logits, "criterion": criterion,
        }

        dist = wrapper.predict_distribution(np.zeros((1, 1)))
        assert not np.any(np.isnan(dist.probas))
        assert_allclose(dist.probas.sum(axis=1), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# 11  FinetuneTabPFN wrapper translation (same logic as TabPFN)
# ---------------------------------------------------------------------------


class TestFinetuneTabPFNTranslation:
    def test_predict_distribution(self):
        torch = pytest.importorskip("torch")
        from scoringbench.wrappers.tabpfn import FinetuneTabPFNWrapper

        wrapper = FinetuneTabPFNWrapper.__new__(FinetuneTabPFNWrapper)
        wrapper._torch = torch
        wrapper._device = "cpu"
        wrapper._model = MagicMock()

        logits = torch.tensor([[0.0, 1.0, 2.0]])
        borders = torch.tensor([0.0, 1.0, 2.0, 3.0])
        criterion = MagicMock()
        criterion.borders = borders
        wrapper._model.predict.return_value = {
            "logits": logits, "criterion": criterion,
        }

        dist = wrapper.predict_distribution(np.zeros((1, 2)))

        expected_probas = torch.softmax(logits, dim=-1).numpy()
        expected_edges = borders.numpy()
        expected_mids = (expected_edges[:-1] + expected_edges[1:]) / 2
        expected_mean = (expected_probas * expected_mids[None, :]).sum(axis=-1)

        assert_allclose(dist.probas, expected_probas, atol=1e-6)
        assert_allclose(dist.bin_edges, expected_edges)
        assert_allclose(dist.bin_midpoints, expected_mids)
        assert_allclose(dist.mean, expected_mean, atol=1e-6)


# ---------------------------------------------------------------------------
# 12  XGBVector wrapper translation
# ---------------------------------------------------------------------------


class TestXGBTranslation:
    def test_predict_distribution(self):
        from scoringbench.wrappers.xgb_vector import XGBVectorWrapper

        wrapper = XGBVectorWrapper.__new__(XGBVectorWrapper)
        wrapper.n_bins = 4
        wrapper._bin_edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        wrapper._bin_midpoints = np.array([0.5, 1.5, 2.5, 3.5])

        known_probas = np.array([
            [0.1, 0.5, 0.3, 0.1],
            [0.25, 0.25, 0.25, 0.25],
        ])
        wrapper._get_probs = lambda X: known_probas

        dist = wrapper.predict_distribution(np.zeros((2, 3)))

        expected_mean = (known_probas * wrapper._bin_midpoints[None, :]).sum(axis=-1)

        assert_allclose(dist.probas, known_probas)
        assert_allclose(dist.bin_edges, wrapper._bin_edges)
        assert_allclose(dist.bin_midpoints, wrapper._bin_midpoints)
        assert_allclose(dist.mean, expected_mean, atol=1e-12)

    def test_uniform_logits_give_uniform_probas(self):
        from scoringbench.wrappers.xgb_vector import XGBVectorWrapper

        wrapper = XGBVectorWrapper.__new__(XGBVectorWrapper)
        wrapper.n_bins = 5
        wrapper._bin_edges = np.linspace(0, 10, 6)
        wrapper._bin_midpoints = np.linspace(1, 9, 5)

        wrapper._get_probs = lambda X: np.full((3, 5), 0.2)

        dist = wrapper.predict_distribution(np.zeros((3, 2)))
        assert_allclose(dist.probas, 0.2, atol=1e-6)

    def test_softmax_in_get_probs(self):
        xgb = pytest.importorskip("xgboost")
        from unittest.mock import patch
        from scoringbench.wrappers.xgb_vector import XGBVectorWrapper

        wrapper = XGBVectorWrapper.__new__(XGBVectorWrapper)
        wrapper.n_bins = 3
        wrapper._bin_edges = np.linspace(0, 3, 4)
        wrapper._bin_midpoints = np.array([0.5, 1.5, 2.5])

        logits = np.array([[1.0, 2.0, 3.0]])
        mock_booster = MagicMock()
        mock_booster.predict.return_value = logits
        wrapper._model = mock_booster

        with patch.object(xgb, "DMatrix", return_value=MagicMock()):
            probs = wrapper._get_probs(np.zeros((1, 2)))

        shift = logits - logits.max(axis=1, keepdims=True)
        exps = np.exp(shift)
        expected = exps / exps.sum(axis=1, keepdims=True)
        assert_allclose(probs, expected, atol=1e-6)

    def test_fit_creates_correct_bins(self):
        pytest.importorskip("xgboost")
        from scoringbench.wrappers.xgb_vector import XGBVectorWrapper

        wrapper = XGBVectorWrapper(n_bins=4, num_boost_round=1)
        rng = np.random.default_rng(0)
        X = rng.normal(size=(20, 2))
        y = np.tile(np.arange(10, dtype=float), 2)
        wrapper.fit(X, y)

        assert_allclose(wrapper._bin_edges, np.linspace(0, 9, 5))
        expected_mids = (wrapper._bin_edges[:-1] + wrapper._bin_edges[1:]) / 2
        assert_allclose(wrapper._bin_midpoints, expected_mids)
