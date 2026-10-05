"""Tests for helper functions extracted from _compute_scoring_rules_torch."""

import math
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from scipy import stats

from scoringbench.univariate.metrics import (
    _interval,
    compute_pit_ks,
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

@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("target, expected_score, expected_coverage", [
    (0.5, 0.9, 1.0), (0.02, 1.5, 0.0), (0.98, 1.5, 0.0),
    (-1.0, 21.9, 0.0), (2.0, 21.9, 0.0),
])
def test_interval_interpolates_uniform_quantiles(shared, target, expected_score, expected_coverage):
    edges = torch.tensor([0.0, 1.0], dtype=torch.float64)
    if not shared:
        edges = edges[None, :]
    score, coverage = _interval(
        0.1, torch.ones((1, 1), dtype=torch.float64), edges,
        torch.tensor([target], dtype=torch.float64), 1, 1, edges.device,
        shared, torch.zeros(1, dtype=torch.long), torch.arange(1),
    )
    assert score == pytest.approx(expected_score, abs=1e-12)
    assert coverage == expected_coverage


def test_interval_interpolation_reports_nominal_coverage_for_uniform_forecast():
    edges = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    probas = torch.full((10, 2), 0.5, dtype=torch.float64)
    cdf = torch.cumsum(probas, dim=-1)
    targets = torch.arange(10, dtype=torch.float64) * 0.1 + 0.05

    _, coverage = _interval(
        0.2, cdf, edges, targets, 10, 2, edges.device,
        True, torch.zeros(10, dtype=torch.long), torch.arange(10),
    )

    # The pre-interpolation code rounded the 10th and 90th percentiles
    # outward to their containing bin edges, making this interval [0, 1].
    lower_index = (cdf >= 0.1).to(torch.uint8).argmax(dim=1)
    upper_index = (cdf >= 0.9).to(torch.uint8).argmax(dim=1) + 1
    old_lows = edges[lower_index]
    old_highs = edges[upper_index]
    old_coverage = ((targets >= old_lows) & (targets <= old_highs)).double().mean().item()

    assert coverage == pytest.approx(0.8)
    assert old_coverage == 1.0


def test_interval_interpolates_per_row_grids():
    edges = torch.tensor([[0.0, 1.0], [10.0, 12.0]], dtype=torch.float64)
    score, coverage = _interval(
        0.1, torch.ones((2, 1), dtype=torch.float64), edges,
        torch.tensor([0.02, 10.04], dtype=torch.float64), 2, 1, edges.device,
        False, torch.zeros(2, dtype=torch.long), torch.arange(2),
    )
    assert score == pytest.approx(2.25, abs=1e-12)
    assert coverage == 0.0


def test_interval_preserves_atoms_and_skips_empty_bins():
    edges = torch.tensor([-1.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float64)
    cdf = torch.tensor([[0.0, 0.4, 1.0, 1.0]], dtype=torch.float64)
    score, coverage = _interval(
        0.1, cdf, edges, torch.zeros(1, dtype=torch.float64), 1, 4, edges.device,
        True, torch.ones(1, dtype=torch.long), torch.arange(1),
    )
    assert score == 0.0
    assert coverage == 1.0


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

@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("edges, masses, targets, expected_pit", [
    ([-1.0, 0.0, 0.0, 1.0], [0.25, 0.5, 0.25],
     [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], [0.0, 0.0, 0.125, 0.5, 0.875, 1.0, 1.0]),
    ([-1.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.25, 0.2, 0.0, 0.3, 0.25],
     [-0.5, 0.0, 0.5], [0.125, 0.5, 0.875]),
    ([0.0, 0.0, 1.0, 1.0], [0.3, 0.4, 0.3],
     [-1.0, 0.0, 0.5, 1.0, 2.0], [0.0, 0.15, 0.5, 0.85, 1.0]),
    ([0.0, 0.0, 0.0], [0.3, 0.7], [-1.0, 0.0, 1.0], [0.0, 0.5, 1.0]),
    ([0.0, 1.0], [1.0], [-1.0, 0.0, 0.3, 0.5, 1.0, 2.0], [0.0, 0.0, 0.3, 0.5, 1.0, 1.0]),
], ids=["mixed-atom", "split-atom", "boundary-atoms", "all-atoms", "continuous"])
def test_pit_uses_both_cdf_limits(shared, edges, masses, targets, expected_pit, monkeypatch):
    n_samples = len(targets)
    edges = torch.tensor(edges, dtype=torch.float64)
    probas = torch.tensor(masses, dtype=torch.float64).repeat(n_samples, 1)
    targets = torch.tensor(targets, dtype=torch.float64)
    if shared:
        y_bin = torch.searchsorted(edges[1:].contiguous(), targets)
    else:
        shifts = torch.arange(n_samples, dtype=torch.float64)
        edges = edges[None, :] + shifts[:, None]
        targets = targets + shifts
        y_bin = torch.searchsorted(edges[:, 1:].contiguous(), targets[:, None]).squeeze(1)
    y_bin = y_bin.clamp(0, len(masses) - 1)
    expected_ks = stats.kstest(expected_pit, "uniform")
    kstest = Mock(wraps=stats.kstest)
    monkeypatch.setattr("scoringbench.univariate.metrics.stats.kstest", kstest)

    result = compute_pit_ks(
        probas, probas.cumsum(dim=1), edges, edges.diff(dim=-1), y_bin,
        targets, shared, torch.arange(n_samples),
    )
    np.testing.assert_allclose(kstest.call_args.args[0], expected_pit, atol=1e-14)
    assert result["pit_ks_stat"] == pytest.approx(expected_ks.statistic, abs=1e-14)
    assert result["pit_ks_pvalue"] == pytest.approx(expected_ks.pvalue, abs=1e-14)


def test_quantile_wcrps_basic(simple_shared_grid, simple_pmf_and_targets):
    """Test basic quantile-weighted CRPS computation."""
    device = simple_shared_grid["device"]
    bin_edges = simple_shared_grid["bin_edges"]
    
    cdf = simple_pmf_and_targets["cdf"]
    y = simple_pmf_and_targets["y"]
    n_samples = simple_pmf_and_targets["n_samples"]
    n_bins = simple_pmf_and_targets["n_bins"]
    
    result = compute_quantile_wcrps(cdf, bin_edges, y, n_samples, n_bins, device, shared=True)
    
    assert "wcrps_left" in result
    assert "wcrps_right" in result
    assert "wcrps_center" in result
    
    for key in ["wcrps_left", "wcrps_right", "wcrps_center"]:
        assert isinstance(result[key], float)
        assert result[key] >= 0  # CRPS components should be non-negative


def test_quantile_wcrps_weights_sum(simple_shared_grid, simple_pmf_and_targets):
    """Test that different weight schemes produce different results."""
    device = simple_shared_grid["device"]
    bin_edges = simple_shared_grid["bin_edges"]
    
    cdf = simple_pmf_and_targets["cdf"]
    y = simple_pmf_and_targets["y"]
    n_samples = simple_pmf_and_targets["n_samples"]
    n_bins = simple_pmf_and_targets["n_bins"]
    
    result = compute_quantile_wcrps(cdf, bin_edges, y, n_samples, n_bins, device, shared=True)
    
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
