"""Tests for helper functions extracted from _compute_scoring_rules_torch."""

import math
import numpy as np
import pytest
import torch

from scoringbench.univariate.metrics import (
    _interval,
    compute_quantile_wcrps,
    compute_crts,
    compute_cde_loss,
    unified_bin_density,
    compute_energy_score_histogram_corrected,
)


def _g_y(probas, bin_widths, y_bin, shared):
    """Pointwise predictive density f(y), matching the production path.

    ``compute_cde_loss`` takes the pre-computed pointwise density ``g_y``
    (the same estimate the log score / DPD scores use) instead of recomputing
    it internally, so tests build it via the real estimator.
    """
    eps = 100 * torch.finfo(torch.float64).eps
    f_bins, _ = unified_bin_density(probas, bin_widths, shared, eps)
    return f_bins.gather(1, y_bin.unsqueeze(1)).squeeze(1)

# Force CPU
torch.cuda.is_available = lambda: False


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_shared_grid():
    """Create a simple shared (1-D) grid."""
    device = torch.device("cpu")
    
    # Grid: [0, 1, 2, 3]
    bin_edges = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32, device=device)
    bin_mids = torch.tensor([0.5, 1.5, 2.5], dtype=torch.float32, device=device)
    bin_widths = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)
    
    return {
        "device": device,
        "bin_edges": bin_edges,
        "bin_mids": bin_mids,
        "bin_widths": bin_widths,
        "shared": True,
    }


@pytest.fixture
def simple_pmf_and_targets(simple_shared_grid):
    """Create simple PMF and target values."""
    device = simple_shared_grid["device"]
    n_samples = 4
    n_bins = 3
    
    # All probability on middle bin
    probas = torch.zeros((n_samples, n_bins), dtype=torch.float32, device=device)
    probas[:, 1] = 1.0
    
    # One target in each bin region
    y = torch.tensor([0.5, 1.5, 2.5, 1.0], dtype=torch.float32, device=device)
    
    # Bin indices
    bin_edges = simple_shared_grid["bin_edges"]
    y_bin = torch.searchsorted(bin_edges[1:].contiguous(), y).clamp(0, n_bins - 1)
    
    # Utilities
    ns_idx = torch.arange(n_samples, device=device)
    cdf = torch.cumsum(probas, dim=-1)
    
    return {
        "probas": probas,
        "y": y,
        "y_bin": y_bin,
        "n_samples": n_samples,
        "n_bins": n_bins,
        "cdf": cdf,
        "ns_idx": ns_idx,
    }


# ============================================================================
# Test _interval function
# ============================================================================

def test_interval_basic(simple_shared_grid, simple_pmf_and_targets):
    """Test basic interval score and coverage computation."""
    device = simple_shared_grid["device"]
    bin_edges = simple_shared_grid["bin_edges"]
    
    cdf = simple_pmf_and_targets["cdf"]
    y = simple_pmf_and_targets["y"]
    y_bin = simple_pmf_and_targets["y_bin"]
    n_samples = simple_pmf_and_targets["n_samples"]
    n_bins = simple_pmf_and_targets["n_bins"]
    ns_idx = simple_pmf_and_targets["ns_idx"]
    
    alpha = 0.05  # 95% confidence
    is_score, coverage = _interval(
        alpha, cdf, bin_edges, y, n_samples, n_bins, device, 
        shared=True, y_bin=y_bin, ns_idx=ns_idx
    )
    
    # For uniform grid with all probability on middle bin and alpha=0.05,
    # we expect wide intervals and near 100% coverage
    assert isinstance(is_score, float)
    assert isinstance(coverage, float)
    assert 0 <= coverage <= 1
    assert is_score >= 0  # Interval score should be non-negative


def test_interval_alpha_dependency(simple_shared_grid, simple_pmf_and_targets):
    """Test that interval scores are well-defined for different alpha levels."""
    device = simple_shared_grid["device"]
    bin_edges = simple_shared_grid["bin_edges"]
    
    cdf = simple_pmf_and_targets["cdf"]
    y = simple_pmf_and_targets["y"]
    y_bin = simple_pmf_and_targets["y_bin"]
    n_samples = simple_pmf_and_targets["n_samples"]
    n_bins = simple_pmf_and_targets["n_bins"]
    ns_idx = simple_pmf_and_targets["ns_idx"]
    
    is_10, cov_10 = _interval(
        0.10, cdf, bin_edges, y, n_samples, n_bins, device,
        shared=True, y_bin=y_bin, ns_idx=ns_idx
    )
    is_05, cov_05 = _interval(
        0.05, cdf, bin_edges, y, n_samples, n_bins, device,
        shared=True, y_bin=y_bin, ns_idx=ns_idx
    )
    
    # Both should produce finite, non-negative results
    assert isinstance(is_10, float)
    assert isinstance(is_05, float)
    assert is_10 >= 0
    assert is_05 >= 0
    assert 0 <= cov_10 <= 1
    assert 0 <= cov_05 <= 1


# ============================================================================
# Test compute_quantile_wcrps function
# ============================================================================

def test_quantile_wcrps_basic(simple_shared_grid, simple_pmf_and_targets):
    """Test basic quantile-weighted CRPS computation."""
    device = simple_shared_grid["device"]
    bin_mids = simple_shared_grid["bin_mids"]
    
    cdf = simple_pmf_and_targets["cdf"]
    y = simple_pmf_and_targets["y"]
    n_samples = simple_pmf_and_targets["n_samples"]
    n_bins = simple_pmf_and_targets["n_bins"]
    
    result = compute_quantile_wcrps(cdf, bin_mids, y, n_samples, n_bins, device, shared=True)
    
    assert "wcrps_left" in result
    assert "wcrps_right" in result
    assert "wcrps_center" in result
    
    for key in ["wcrps_left", "wcrps_right", "wcrps_center"]:
        assert isinstance(result[key], float)
        assert result[key] >= 0  # CRPS components should be non-negative


def test_quantile_wcrps_weights_sum(simple_shared_grid, simple_pmf_and_targets):
    """Test that different weight schemes produce different results."""
    device = simple_shared_grid["device"]
    bin_mids = simple_shared_grid["bin_mids"]
    
    cdf = simple_pmf_and_targets["cdf"]
    y = simple_pmf_and_targets["y"]
    n_samples = simple_pmf_and_targets["n_samples"]
    n_bins = simple_pmf_and_targets["n_bins"]
    
    result = compute_quantile_wcrps(cdf, bin_mids, y, n_samples, n_bins, device, shared=True)
    
    # Different weighting schemes should produce different values
    # (not all the same)
    values = [result["wcrps_left"], result["wcrps_right"], result["wcrps_center"]]
    assert len(set(values)) > 1  # Not all values should be identical


# ============================================================================
# Test compute_crts function
# ============================================================================

def test_crts_basic(simple_shared_grid, simple_pmf_and_targets):
    """Test basic CRTS computation."""
    bin_edges = simple_shared_grid["bin_edges"]

    cdf = simple_pmf_and_targets["cdf"]
    y = simple_pmf_and_targets["y"]
    y_bin = simple_pmf_and_targets["y_bin"]

    results = compute_crts(cdf, bin_edges, y, y_bin, shared=True)

    assert isinstance(results, dict)
    assert set(results.keys()) == {"crts_alpha_1.01", "crts_alpha_1.2", "crts_alpha_1.5", "crts_alpha_2.0"}
    for v in results.values():
        assert isinstance(v, float)


def test_crts_perfect_prediction(device=torch.device("cpu")):
    """Test CRTS with perfect prediction (point mass at target)."""
    n_samples = 2
    n_bins = 3

    # Point mass at middle bin
    probas = torch.zeros((n_samples, n_bins), dtype=torch.float32, device=device)
    probas[:, 1] = 1.0

    cdf = torch.cumsum(probas, dim=-1)
    bin_edges = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32, device=device)
    y = torch.tensor([1.5, 1.5], dtype=torch.float32, device=device)
    y_bin = torch.tensor([1, 1], dtype=torch.int64, device=device)

    results = compute_crts(cdf, bin_edges, y, y_bin, shared=True)

    # Perfect prediction should give a finite value for all alphas
    for key, val in results.items():
        assert isinstance(val, float), f"{key} should be float"


def test_crts_invalid_alpha_raises():
    """compute_crts should raise ValueError for alpha <= 1 + 1e-4."""
    n_bins = 4
    cdf = torch.linspace(0.25, 1.0, n_bins).unsqueeze(0)
    bin_edges = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    y = torch.tensor([2.5])
    y_bin = torch.tensor([2])

    import pytest
    with pytest.raises(ValueError, match="1e-4"):
        compute_crts(cdf, bin_edges, y, y_bin, shared=True, alphas=[1.0])


# ============================================================================
# Test compute_cde_loss function
# ============================================================================

def test_cde_loss_basic(simple_shared_grid, simple_pmf_and_targets):
    """Test basic CDE loss computation."""
    device = simple_shared_grid["device"]
    bin_edges = simple_shared_grid["bin_edges"]
    bin_widths = simple_shared_grid["bin_widths"]
    
    probas = simple_pmf_and_targets["probas"]
    y = simple_pmf_and_targets["y"]
    y_bin = simple_pmf_and_targets["y_bin"]
    
    bw = bin_widths[None, :]
    g_y = _g_y(probas, bin_widths, y_bin, shared=True)
    
    cde = compute_cde_loss(probas, bin_widths, g_y, bw, shared=True)
    
    assert isinstance(cde, float)
    assert math.isfinite(cde)  # CDE loss is a finite proper-scoring value


def test_cde_loss_zero_prediction(device=torch.device("cpu")):
    """Test CDE loss with zero (impossible) prediction."""
    n_samples = 1
    n_bins = 3
    
    # All probability on wrong bin (2), target in bin 0
    probas = torch.zeros((n_samples, n_bins), dtype=torch.float32, device=device)
    probas[:, 2] = 1.0
    
    y = torch.tensor([0.5], dtype=torch.float32, device=device)
    y_bin = torch.tensor([0], dtype=torch.int64, device=device)
    
    bin_edges = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32, device=device)
    bin_widths = torch.ones(n_bins, dtype=torch.float32, device=device)
    bw = bin_widths[None, :]
    g_y = _g_y(probas, bin_widths, y_bin, shared=True)
    
    cde = compute_cde_loss(probas, bin_widths, g_y, bw, shared=True)
    
    # Zero probability at target is a legitimate input: g(y)=0 gives a finite
    # score.  Finiteness comes from the strictly positive bin widths (all 1.0
    # here), not from any density clamp -- the density path now *requires*
    # positive widths and raises otherwise.
    assert isinstance(cde, float)
    assert not math.isinf(cde)


# ============================================================================
# Test consistency with legacy behavior
# ============================================================================

def test_all_helpers_with_energy_score(simple_shared_grid, simple_pmf_and_targets):
    """Test that helper functions work together with energy score computation."""
    device = simple_shared_grid["device"]
    bin_edges = simple_shared_grid["bin_edges"]
    
    probas = simple_pmf_and_targets["probas"]
    y = simple_pmf_and_targets["y"]
    
    # Compute energy score
    energy_result = compute_energy_score_histogram_corrected(
        probas, bin_edges, y, betas=[0.5, 1.0, 1.5]
    )
    
    assert "energy_score_beta_0.5" in energy_result
    assert "energy_score_beta_1.0" in energy_result
    assert "energy_score_beta_1.5" in energy_result
    
    # All should be non-negative floats
    for beta in [0.5, 1.0, 1.5]:
        key = f"energy_score_beta_{beta}"
        assert isinstance(energy_result[key], float)
        assert energy_result[key] >= 0


def test_helpers_with_several_betas(simple_shared_grid, simple_pmf_and_targets):
    """Test helper functions work with multiple beta values."""
    device = simple_shared_grid["device"]
    bin_edges = simple_shared_grid["bin_edges"]
    
    probas = simple_pmf_and_targets["probas"]
    y = simple_pmf_and_targets["y"]
    
    betas_list = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 1.7, 1.8, 1.9]
    energy_result = compute_energy_score_histogram_corrected(
        probas, bin_edges, y, betas=betas_list
    )
    
    # All beta values should be present
    for beta in betas_list:
        key = f"energy_score_beta_{beta}"
        assert key in energy_result
        assert isinstance(energy_result[key], float)
