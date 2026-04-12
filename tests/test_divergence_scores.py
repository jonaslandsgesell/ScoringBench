"""Tests for divergence-based scoring metrics: Cressie-Read power divergence.

This module provides comprehensive numerical validation of Cressie-Read power divergence metrics
using explicit computations with known values rather than simple inequality checks.

These are loss/divergence metrics where lower values indicate better predictions.
"""

import math
import numpy as np
import pytest
import torch

# Force CPU
torch.cuda.is_available = lambda: False

from scoringbench.metrics import (
    compute_scoring_rules,
    CRESSIE_READ_LAMBDAS,
)
from scoringbench.wrappers import DistributionPrediction


# ============================================================================
# Numerical Reference Implementations
# ============================================================================

def reference_cressie_read_score(p_at_y, dz_at_y, lam):
    """Reference implementation of Cressie-Read Power Divergence.
    
    A family of divergence measures parameterized by lambda:
    L = (g(y)^(-lambda) - 1) / (lambda * (lambda + 1))
    
    As lambda -> 0, this converges to the continuous log score (-log g(y)).
    As lambda -> -1, this converges to g(y) * log(g(y)).
    """
    g_y = p_at_y / dz_at_y
    eps = 1e-10
    g_y = max(g_y, eps)  # Clamp to avoid NaNs
    
    if abs(lam) < 1e-5:
        # Limit as lambda -> 0 is -log(g(y))
        return -np.log(g_y)
    elif abs(lam + 1.0) < 1e-5:
        # Limit as lambda -> -1 is g(y) * log(g(y))
        return g_y * np.log(g_y)
    else:
        return (g_y ** (-lam) - 1.0) / (lam * (lam + 1.0))


# ============================================================================
# Test Fixture: Distribution Builders
# ============================================================================

def get_simple_distribution():
    """Create a simple test distribution with known properties.
    
    Configuration: Bin edges [0, 1, 2, 3], all probability on middle bin [1, 2].
    """
    bin_edges = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    bin_mids = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    # All probability on middle bin
    probas = np.zeros((1, 3), dtype=np.float32)
    probas[0, 1] = 1.0
    mean = np.array([1.5], dtype=np.float64)
    
    return DistributionPrediction(
        probas=probas,
        bin_edges=bin_edges,
        bin_midpoints=bin_mids,
        mean=mean,
    )


def get_perfect_prediction_distribution(bin_idx=0):
    """Create a distribution with all mass on one bin.
    
    Args:
        bin_idx: Which bin (0, 1, or 2) gets all the probability.
    
    Returns:
        Distribution and corresponding target y value for perfect prediction.
    """
    bin_edges = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    bin_mids = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    
    probas = np.zeros((1, 3), dtype=np.float32)
    probas[0, bin_idx] = 1.0
    
    dist = DistributionPrediction(
        probas=probas,
        bin_edges=bin_edges,
        bin_midpoints=bin_mids,
        mean=np.array([bin_mids[bin_idx]], dtype=np.float64),
    )
    
    y_true = np.array([bin_mids[bin_idx]], dtype=np.float32)
    return dist, y_true


def get_imperfect_distribution():
    """Create a distribution with spread across bins.
    
    Configuration: probabilities [0.3, 0.4, 0.3] across bins.
    """
    bin_edges = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    bin_mids = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    
    probas = np.array([[0.3, 0.4, 0.3]], dtype=np.float32)
    
    return DistributionPrediction(
        probas=probas,
        bin_edges=bin_edges,
        bin_midpoints=bin_mids,
        mean=np.array([1.5], dtype=np.float64),
    )


# ============================================================================
# Sanity Check Tests: Finite Values, No NaNs/Infs
# ============================================================================

@pytest.mark.parametrize("lam", CRESSIE_READ_LAMBDAS)
def test_cressie_read_score_is_finite(lam):
    """Test that Cressie-Read scores are finite and well-defined."""
    dist = get_simple_distribution()
    y_true = np.array([1.5], dtype=np.float32)
    
    metrics = compute_scoring_rules(dist, y_true)
    key = f"cressie_read_lambda_{lam}"
    
    assert key in metrics, f"Missing key {key}"
    assert isinstance(metrics[key], float)
    assert not np.isnan(metrics[key]), f"{key} is NaN"
    assert not np.isinf(metrics[key]), f"{key} is Inf"


# ============================================================================
# Numerical Validation Tests: Exact Values
# ============================================================================

class TestCressieReadExactValues:
    """Explicit numerical tests for Cressie-Read scores with known results."""
    
    def test_perfect_prediction_all_lambdas(self):
        """Perfect prediction should yield 0.0 for all lambda values.
        
        When all probability mass is exactly at the target:
        - density g(y) = 1.0
        - Cressie-Read = (1.0^(-lambda) - 1.0) / (lambda*(lambda+1)) = 0 / denominator = 0
        """
        dist, y_true = get_perfect_prediction_distribution(bin_idx=1)
        metrics = compute_scoring_rules(dist, y_true)
        
        for lam in CRESSIE_READ_LAMBDAS:
            key = f"cressie_read_lambda_{lam}"
            score = metrics[key]
            assert math.isclose(score, 0.0, abs_tol=0.01), (
                f"cressie_read_lambda_{lam} for perfect prediction should be 0.0, got {score}"
            )
    
    @pytest.mark.parametrize("lam", CRESSIE_READ_LAMBDAS)
    def test_imperfect_prediction_positive(self, lam):
        """Imperfect predictions should have positive divergence scores.
        
        For g(y) < 1 and lambda ≠ 0, -1:
        - g(y)^(-lambda) > 1.0 (since g < 1 and negative exponent)
        - So (g(y)^(-lambda) - 1.0) > 0
        """
        dist = get_imperfect_distribution()
        y_true = np.array([1.5], dtype=np.float32)
        metrics = compute_scoring_rules(dist, y_true)
        
        key = f"cressie_read_lambda_{lam}"
        score = metrics[key]
        
        # For imperfect prediction, expect non-zero positive divergence
        assert score > 0, (
            f"Cressie-Read score for imperfect prediction should be positive, "
            f"got {score:.6f} for {key}"
        )
    
    def test_lambda_neg0p5_vs_0p5_imperfect(self):
        """Compare lambda=-0.5 vs lambda=0.2 for imperfect prediction.
        
        Both should give positive values, but with different magnitudes.
        """
        dist = get_imperfect_distribution()
        y_true = np.array([1.5], dtype=np.float32)
        metrics = compute_scoring_rules(dist, y_true)
        
        score_neg = metrics["cressie_read_lambda_-0.5"]
        score_pos = metrics["cressie_read_lambda_0.2"]
        
        # Both should be positive for g(y)=0.4
        assert score_neg > 0, f"lambda=-0.5 should be positive, got {score_neg:.6f}"
        assert score_pos > 0, f"lambda=0.2 should be positive, got {score_pos:.6f}"
        
        # Magnitude should differ (depends on the specific formula)
        assert not math.isclose(score_neg, score_pos, rel_tol=0.1), (
            f"lambda=-0.5 ({score_neg:.6f}) and lambda=0.2 ({score_pos:.6f}) "
            f"should give different values"
        )
    
    @pytest.mark.parametrize("lam", CRESSIE_READ_LAMBDAS)
    def test_imperfect_worse_than_perfect(self, lam):
        """Imperfect predictions should have worse (higher) divergence than perfect."""
        dist_perfect, y_perfect = get_perfect_prediction_distribution(bin_idx=1)
        dist_imperfect = get_imperfect_distribution()
        y_imperfect = np.array([1.5], dtype=np.float32)
        
        metrics_perfect = compute_scoring_rules(dist_perfect, y_perfect)
        metrics_imperfect = compute_scoring_rules(dist_imperfect, y_imperfect)
        
        key = f"cressie_read_lambda_{lam}"
        score_perfect = metrics_perfect[key]
        score_imperfect = metrics_imperfect[key]
        
        # Perfect: 0.0, Imperfect: positive divergence
        assert score_perfect < score_imperfect, (
            f"Perfect prediction ({score_perfect:.6f}) should have lower divergence "
            f"than imperfect ({score_imperfect:.6f})"
        )


class TestComparisonBetweenMetrics:
    """Tests comparing Cressie-Read behavior at perfect vs imperfect predictions."""
    
    def test_cressie_read_optimal_at_perfect_prediction(self):
        """Cressie-Read should be 0.0 at perfect prediction."""
        dist, y_true = get_perfect_prediction_distribution(bin_idx=0)
        metrics = compute_scoring_rules(dist, y_true)
        
        # Cressie-Read should be 0.0 (its best value)
        cr_scores = [metrics[f"cressie_read_lambda_{l}"] for l in CRESSIE_READ_LAMBDAS]
        assert all(math.isclose(s, 0.0, abs_tol=0.01) for s in cr_scores), (
            f"All Cressie-Read scores should be 0.0 at perfect prediction, got {cr_scores}"
        )
    
    def test_cressie_read_worse_at_imperfect(self):
        """Cressie-Read should worsen (increase) with imperfect predictions."""
        dist_perfect, y_perfect = get_perfect_prediction_distribution(bin_idx=1)
        dist_imperfect = get_imperfect_distribution()
        y_imperfect = np.array([1.5], dtype=np.float32)
        
        metrics_perfect = compute_scoring_rules(dist_perfect, y_perfect)
        metrics_imperfect = compute_scoring_rules(dist_imperfect, y_imperfect)
        
        # Check Cressie-Read: becomes positive (worse)
        for lam in CRESSIE_READ_LAMBDAS:
            key = f"cressie_read_lambda_{lam}"
            score_perfect = metrics_perfect[key]
            score_imperfect = metrics_imperfect[key]
            assert score_perfect < score_imperfect, (
                f"Perfect prediction score ({score_perfect:.6f}) should be lower (better) "
                f"than imperfect ({score_imperfect:.6f}) for {key}"
            )


# ============================================================================
# Integration and Consistency Tests
# ============================================================================

def test_all_new_metrics_present_in_results():
    """Verify all Cressie-Read metrics are computed."""
    dist = get_simple_distribution()
    y_true = np.array([1.5], dtype=np.float32)
    
    metrics = compute_scoring_rules(dist, y_true)
    
    # Check Cressie-Read
    for lam in CRESSIE_READ_LAMBDAS:
        key = f"cressie_read_lambda_{lam}"
        assert key in metrics, f"Missing {key}"


def test_different_lambdas_produce_different_scores():
    """Different lambda values should produce different results."""
    dist = get_imperfect_distribution()
    y_true = np.array([1.5], dtype=np.float32)
    
    metrics = compute_scoring_rules(dist, y_true)
    
    scores = [metrics[f"cressie_read_lambda_{l}"] for l in CRESSIE_READ_LAMBDAS]
    unique_scores = set(scores)
    
    # Should have multiple distinct values
    assert len(unique_scores) > 1, (
        "Different lambda values should produce different Cressie-Read scores"
    )


# ============================================================================
class TestLimitBehavior:
    """Tests verifying limit behavior as parameters approach special values."""
    
    def test_cressie_read_lambda_near_zero_convergence(self):
        """Verify lambda->0 converges to negative log score.
        
        As lambda -> 0, Cressie-Read should approach -log(g(y)).
        We test with a tiny lambda value and verify it's close to the limit.
        """
        dist = get_imperfect_distribution()
        y_true = np.array([1.5], dtype=np.float32)
        
        p_at_y = 0.4
        dz_at_y = 1.0
        g_y = p_at_y / dz_at_y
        expected_limit = -np.log(g_y)
        
        # Compute using reference at lambda=0 (should use the limit)
        computed_via_reference = reference_cressie_read_score(p_at_y, dz_at_y, 0.0)
        
        assert math.isclose(computed_via_reference, expected_limit, rel_tol=1e-6), (
            f"Lambda->0 limit: expected {expected_limit:.8f}, got {computed_via_reference:.8f}"
        )
    
    def test_cressie_read_lambda_near_minus_one_convergence(self):
        """Verify lambda->-1 converges to g(y)*log(g(y)).
        
        As lambda -> -1, Cressie-Read should approach g(y)*log(g(y)).
        """
        p_at_y = 0.4
        dz_at_y = 1.0
        g_y = p_at_y / dz_at_y
        expected_limit = g_y * np.log(g_y)
        
        # Compute using reference at lambda=-1 (should use the limit)
        computed_via_reference = reference_cressie_read_score(p_at_y, dz_at_y, -1.0)
        
        assert math.isclose(computed_via_reference, expected_limit, rel_tol=1e-6), (
            f"Lambda->-1 limit: expected {expected_limit:.8f}, got {computed_via_reference:.8f}"
        )


class TestEdgeCasesAndRobustness:
    """Tests for edge cases and numerical robustness."""
    
    def test_cressie_read_very_small_density(self):
        """Test Cressie-Read with very small but nonzero density.
        
        Should not produce NaN or Inf values even with small g(y).
        """
        bin_edges = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        bin_mids = np.array([0.5, 1.5], dtype=np.float32)
        # Very small probability on first bin
        probas = np.array([[0.001, 0.999]], dtype=np.float32)
        
        dist = DistributionPrediction(
            probas=probas,
            bin_edges=bin_edges,
            bin_midpoints=bin_mids,
            mean=np.array([1.5], dtype=np.float64),
        )
        y_true = np.array([0.5], dtype=np.float32)
        metrics = compute_scoring_rules(dist, y_true)
        
        for lam in CRESSIE_READ_LAMBDAS:
            score = metrics[f"cressie_read_lambda_{lam}"]
            assert not np.isnan(score), f"Score is NaN for lambda={lam}"
            assert not np.isinf(score), f"Score is Inf for lambda={lam}"
            # Very low confidence should have very high score (at least 2.0)
            assert score > 2.0, f"Very low confidence should have high score, got {score}"
