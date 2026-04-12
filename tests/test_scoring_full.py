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
    CRESSIE_READ_LAMBDAS,
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
# Numerical verification helpers
# ---------------------------------------------------------------------------

def _compute_exact_crps_histogram(probas, bin_mids, bin_widths, y):
    """
    Computes CRPS by exactly integrating the squared error of a
    piecewise-linear CDF against the Heaviside step function.
    
    This assumes distributions are uniform within bins with linearly
    interpolated CDF at bin boundaries.
    """
    try:
        import torch
    except ImportError:
        # Fallback if torch not available
        return None
    
    probas_t = torch.as_tensor(probas, dtype=torch.float32)
    bin_mids_t = torch.as_tensor(bin_mids, dtype=torch.float32)
    bin_widths_t = torch.as_tensor(bin_widths, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.float32)
    
    # CDF at the left edges
    cdf_left = torch.cumsum(probas_t, dim=-1) - probas_t
    
    # Handle both shared and per-sample bin grids
    mids_ext = bin_mids_t[None, :] if bin_mids_t.ndim == 1 else bin_mids_t
    widths_ext = bin_widths_t[None, :] if bin_widths_t.ndim == 1 else bin_widths_t
    left_edges = mids_ext - widths_ext / 2.0
    
    # alpha: fraction of each bin that is to the left of y
    alpha = (y_t[:, None] - left_edges) / widths_ext
    alpha = torch.clamp(alpha, min=0.0, max=1.0)
    
    P = cdf_left
    p = probas_t
    
    # Part 1: Integral from left to alpha (where indicator is 0)
    term_left = widths_ext * (
        P.pow(2) * alpha +
        P * p * alpha.pow(2) +
        (p.pow(2) / 3.0) * alpha.pow(3)
    )
    
    # Part 2: Integral from alpha to 1 (where indicator is 1)
    Pm1 = P - 1.0
    val_1 = Pm1.pow(2) + Pm1 * p + (p.pow(2) / 3.0)
    val_a = Pm1.pow(2) * alpha + Pm1 * p * alpha.pow(2) + (p.pow(2) / 3.0) * alpha.pow(3)
    term_right = widths_ext * (val_1 - val_a)
    
    crps = (term_left + term_right).sum(dim=-1).mean().item()
    return crps

def _compute_crps_numerical(probas, bin_edges, bin_midpoints, y_true):
    """Compute CRPS numerically by exactly integrating piecewise-linear CDF.
    
    CRPS(F, y) = ∫ (F(z) - I[z ≥ y])² dz
    
    For histogram distributions with piecewise-linear CDF:
    - CDF is linear within each bin (from left edge to right edge)
    - We integrate the squared error over two regions:
      * Region 1: z < y (where indicator = 0)
      * Region 2: z ≥ y (where indicator = 1)
    
    Parameters
    ----------
    probas : ndarray (n_samples, n_bins)
        Probability mass per bin
    bin_edges : ndarray (n_bins+1,) or (n_samples, n_bins+1)
        Bin edge locations
    bin_midpoints : ndarray (n_bins,) or (n_samples, n_bins)
        Bin midpoint locations
    y_true : ndarray (n_samples,)
        Target values
        
    Returns
    -------
    float
        Mean CRPS across samples
    """
    try:
        torch = __import__('torch')
    except ImportError:
        # Fallback to simple implementation if torch is not available
        probas = np.asarray(probas, dtype=np.float64)
        y_true = np.asarray(y_true, dtype=np.float64)
        bin_edges = np.asarray(bin_edges, dtype=np.float64)
        bin_midpoints = np.asarray(bin_midpoints, dtype=np.float64)
        
        n_samples = probas.shape[0]
        crps_values = []
        
        for i in range(n_samples):
            p = probas[i]
            y = y_true[i]
            
            # Get bin edges and midpoints for this sample
            if bin_edges.ndim == 1:
                edges = bin_edges
                widths = np.diff(edges)
                mids = bin_midpoints
            else:
                edges = bin_edges[i]
                widths = np.diff(edges)
                mids = bin_midpoints[i]
            
            # CDF at left edges
            cdf_left = np.cumsum(p) - p
            left_edges = edges[:-1]
            
            # Fraction of each bin to the left of y
            alpha = (y - left_edges) / widths
            alpha = np.clip(alpha, 0.0, 1.0)
            
            P = cdf_left
            
            # Part 1: integral from left edge to min(right edge, y)
            term_left = widths * (
                P**2 * alpha +
                P * p * alpha**2 +
                (p**2 / 3.0) * alpha**3
            )
            
            # Part 2: integral from max(left edge, y) to right edge
            Pm1 = P - 1.0
            val_1 = Pm1**2 + Pm1 * p + (p**2 / 3.0)
            val_a = Pm1**2 * alpha + Pm1 * p * alpha**2 + (p**2 / 3.0) * alpha**3
            term_right = widths * (val_1 - val_a)
            
            crps_i = (term_left + term_right).sum()
            crps_values.append(crps_i)
        
        return np.mean(crps_values)
    
    # PyTorch implementation for better numerical precision
    probas_t = torch.as_tensor(probas, dtype=torch.float64)
    y_true_t = torch.as_tensor(y_true, dtype=torch.float64)
    bin_edges_t = torch.as_tensor(bin_edges, dtype=torch.float64)
    bin_midpoints_t = torch.as_tensor(bin_midpoints, dtype=torch.float64)
    
    n_samples = probas_t.shape[0]
    crps_values = []
    
    for i in range(n_samples):
        p = probas_t[i]
        y = y_true_t[i]
        
        # Get bin edges and midpoints for this sample
        if bin_edges_t.ndim == 1:
            edges = bin_edges_t
            widths = edges[1:] - edges[:-1]
            mids = bin_midpoints_t
        else:
            edges = bin_edges_t[i]
            widths = edges[1:] - edges[:-1]
            mids = bin_midpoints_t[i]
        
        # CDF at left edges
        cdf_left = torch.cumsum(p, dim=0) - p
        left_edges = edges[:-1]
        
        # Fraction of each bin to the left of y
        alpha = (y - left_edges) / widths
        alpha = torch.clamp(alpha, min=0.0, max=1.0)
        
        P = cdf_left
        
        # Part 1: integral from left edge to min(right edge, y)
        term_left = widths * (
            P**2 * alpha +
            P * p * alpha**2 +
            (p**2 / 3.0) * alpha**3
        )
        
        # Part 2: integral from max(left edge, y) to right edge
        Pm1 = P - 1.0
        val_1 = Pm1**2 + Pm1 * p + (p**2 / 3.0)
        val_a = Pm1**2 * alpha + Pm1 * p * alpha**2 + (p**2 / 3.0) * alpha**3
        term_right = widths * (val_1 - val_a)
        
        crps_i = (term_left + term_right).sum()
        crps_values.append(crps_i.item())
    
    return np.mean(crps_values)


def _compute_wcrps_numerical(probas, bin_edges, bin_midpoints, y_true, weight_func_name):
    """Compute weighted CRPS (Gneiting & Ranjan 2011) numerically.
    
    qwCRPS_v(F, y) = 2 ∫₀¹ ρ_α(y, q_α) v(α) dα
    
    where ρ_α(y, q) = (I[y ≤ q] − α)(q − y) is the pinball loss
    and q_α is the α-quantile from CDF F.
    
    Parameters
    ----------
    probas : ndarray (n_samples, n_bins)
        Probability mass per bin
    bin_edges : ndarray (n_bins+1,) or (n_samples, n_bins+1)
        Bin edge locations
    bin_midpoints : ndarray (n_bins,) or (n_samples, n_bins)
        Bin midpoint locations
    y_true : ndarray (n_samples,)
        Target values
    weight_func_name : str
        One of 'left', 'right', 'center'
        
    Returns
    -------
    float
        Mean weighted CRPS across samples
    """
    probas = np.asarray(probas, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    bin_edges = np.asarray(bin_edges, dtype=np.float64)
    bin_midpoints = np.asarray(bin_midpoints, dtype=np.float64)
    
    n_samples = probas.shape[0]
    wcrps_values = []
    
    # Quantile levels for numerical integration
    alphas = np.linspace(0.01, 0.99, 99)
    d_alpha = 1.0 / (len(alphas) + 1)  # ≈ 0.01
    
    for i in range(n_samples):
        p = probas[i]
        y = y_true[i]
        
        # Get bin edges and midpoints for this sample
        if bin_edges.ndim == 1:
            edges = bin_edges
            mids = bin_midpoints
        else:
            edges = bin_edges[i]
            mids = bin_midpoints[i]
        
        # Compute CDF
        cdf = np.cumsum(p)
        
        # For each quantile level alpha, find the alpha-quantile value
        pinballs = []
        for alpha in alphas:
            # Find smallest bin k with cdf[k] >= alpha
            k = np.searchsorted(cdf, alpha, side='left')
            k = np.clip(k, 0, len(mids) - 1)
            q_alpha = mids[k]
            
            # Pinball loss: 2(I[y ≤ q_α] − α)(q_α − y)
            indicator = float(y <= q_alpha)
            pinball = 2.0 * (indicator - alpha) * (q_alpha - y)
            pinballs.append(pinball)
        
        pinballs = np.array(pinballs)
        
        # Weight function
        if weight_func_name == 'left':
            v = (1.0 - alphas)**2
        elif weight_func_name == 'right':
            v = alphas**2
        elif weight_func_name == 'center':
            v = alphas * (1.0 - alphas)
        else:
            raise ValueError(f"Unknown weight function: {weight_func_name}")
        
        # Weighted sum
        wcrps_i = np.sum(pinballs * v) * d_alpha
        wcrps_values.append(wcrps_i)
    
    return np.mean(wcrps_values)


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
        dist, y = delta
        # Compute expected CRPS using exact formula (uniform within bins, piecewise-linear CDF)
        expected_crps = _compute_exact_crps_histogram(
            dist.probas, dist.bin_midpoints, np.diff(dist.bin_edges), y
        )
        assert r["crps"] == pytest.approx(expected_crps, rel=0.01)

    def test_log_score(self, delta):
        r = compute_scoring_rules(*delta)
        assert r["log_score"] == pytest.approx(0.0, abs=1e-6)

    def test_sharpness(self, delta):
        r = compute_scoring_rules(*delta)
        assert r["sharpness"] == pytest.approx(0.0, abs=1e-6)

    def test_crls(self, delta):
        r = compute_scoring_rules(*delta)
        assert r["crls"] == pytest.approx(0.0, abs=1e-4)

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
        # Verify against numerical computation
        dist, y = uniform
        numerical_crps = _compute_crps_numerical(dist.probas, dist.bin_edges, 
                                                  dist.bin_midpoints, y)
        print(f"Uniform CRPS: function={r['crps']}, numerical={numerical_crps}")
        # Function and numerical should match
        assert r["crps"] == pytest.approx(numerical_crps, rel=1e-2), \
            f"CRPS mismatch: function={r['crps']} vs numerical={numerical_crps}"

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
        assert r["energy_score_beta_1.0"] == pytest.approx(r["crps"], rel=0.1)

    def test_wcrps_left(self, uniform):
        r = compute_scoring_rules(*uniform)
        dist, y = uniform
        numerical_wcrps_left = _compute_wcrps_numerical(dist.probas, dist.bin_edges,
                                                        dist.bin_midpoints, y, 'left')
        print(f"wCRPS_left: function={r['wcrps_left']}, numerical={numerical_wcrps_left}")
        # Function and numerical should match
        assert r["wcrps_left"] == pytest.approx(numerical_wcrps_left, rel=1e-3), \
            f"wCRPS_left mismatch: function={r['wcrps_left']} vs numerical={numerical_wcrps_left}"

    def test_wcrps_right(self, uniform):
        r = compute_scoring_rules(*uniform)
        dist, y = uniform
        numerical_wcrps_right = _compute_wcrps_numerical(dist.probas, dist.bin_edges,
                                                         dist.bin_midpoints, y, 'right')
        print(f"wCRPS_right: function={r['wcrps_right']}, numerical={numerical_wcrps_right}")
        # Function and numerical should match
        assert r["wcrps_right"] == pytest.approx(numerical_wcrps_right, rel=1e-3), \
            f"wCRPS_right mismatch: function={r['wcrps_right']} vs numerical={numerical_wcrps_right}"

    def test_wcrps_center(self, uniform):
        r = compute_scoring_rules(*uniform)
        dist, y = uniform
        numerical_wcrps_center = _compute_wcrps_numerical(dist.probas, dist.bin_edges,
                                                          dist.bin_midpoints, y, 'center')
        print(f"wCRPS_center: function={r['wcrps_center']}, numerical={numerical_wcrps_center}")
        # Function and numerical should match
        assert r["wcrps_center"] == pytest.approx(numerical_wcrps_center, rel=1e-3), \
            f"wCRPS_center mismatch: function={r['wcrps_center']} vs numerical={numerical_wcrps_center}"


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
        dist, y = asym
        # Compute expected CRPS using exact formula (uniform within bins, piecewise-linear CDF)
        expected_crps = _compute_exact_crps_histogram(
            dist.probas, dist.bin_midpoints, np.diff(dist.bin_edges), y
        )
        assert r["crps"] == pytest.approx(expected_crps, rel=0.01)

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
        # Verify against numerical computation
        dist, y = persample
        numerical_crps = _compute_crps_numerical(dist.probas, dist.bin_edges,
                                                  dist.bin_midpoints, y)
        print(f"Per-sample CRPS: function={r['crps']}, numerical={numerical_crps}")
        # Function and numerical should match
        assert r["crps"] == pytest.approx(numerical_crps, rel=1e-2), \
            f"CRPS mismatch: function={r['crps']} vs numerical={numerical_crps}"

    def test_returns_all_keys(self, persample):
        r = compute_scoring_rules(*persample)
        expected_keys = {
            "crps", "log_score", "sharpness",
            "coverage_90", "interval_score_90",
            "coverage_95", "interval_score_95",
            "crls", "cde_loss",
            "wcrps_left", "wcrps_right", "wcrps_center",
            "dispersion",
        }
        expected_keys |= {f"energy_score_beta_{b}" for b in ENERGY_BETAS}
        expected_keys |= {f"cressie_read_lambda_{lam}" for lam in CRESSIE_READ_LAMBDAS}
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

    def test_energy_beta_1_equals_crps(self, random_dist):
        """Energy score with β=1.0 should approximately equal CRPS."""
        r = compute_scoring_rules(*random_dist)
        # Energy score with β=1 is theoretically equivalent to CRPS for continuous distributions
        # but discretization introduces small differences
        assert r["energy_score_beta_1.0"] == pytest.approx(r["crps"], rel=0.20), \
            f"Energy score β=1.0 ({r['energy_score_beta_1.0']}) should match CRPS ({r['crps']})"

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
        wrapper._device = "cuda"
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
        wrapper._device = "cuda"
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
        wrapper._device = "cuda"
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
        wrapper._device = "cuda"
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
        wrapper._device = "cuda"
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
