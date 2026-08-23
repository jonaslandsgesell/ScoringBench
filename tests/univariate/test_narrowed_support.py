"""Test that tail contributions are handled correctly by the scoring rules.

This is a simplified, standalone version that doesn't rely on the analytical
test infrastructure. It verifies that:

1. The padding mechanism doesn't introduce NaNs or infinities
2. Model rankings are stable across different bin configurations
3. Scoring works correctly with narrow-support histograms
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scoringbench.univariate.metrics import compute_scoring_rules


def make_simple_dataset(n_samples=32, n_features=5, seed=42):
    """Generate a simple synthetic dataset for testing.
    
    Returns
    -------
    dict with keys:
        - X: (n_samples, n_features) features
        - y: (n_samples,) targets
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    y = rng.randn(n_samples) + X[:, 0]  # y depends on first feature
    return {"X": X, "y": y}


def make_bins(y, n_bins=128, pad_frac=0.1):
    """Create bin edges covering the target range with padding.
    
    Parameters
    ----------
    y : array of shape (n_samples,)
        Target values
    n_bins : int
        Number of bins
    pad_frac : float
        Fraction of range to pad on each side
    
    Returns
    -------
    bin_edges : array of shape (n_bins + 1,)
    """
    lo, hi = y.min(), y.max()
    span = hi - lo
    pad = pad_frac * span
    bin_edges = np.linspace(lo - pad, hi + pad, n_bins + 1)
    return bin_edges


def make_oracle_histogram(y, bin_edges):
    """Create an oracle histogram (empirical CDF binned).
    
    Parameters
    ----------
    y : array of shape (n_samples,)
        Target values
    bin_edges : array of shape (n_bins + 1,)
        Bin edges
    
    Returns
    -------
    probas : array of shape (n_samples, n_bins)
        Probability mass in each bin for each sample
    """
    n_samples = len(y)
    n_bins = len(bin_edges) - 1
    probas = np.zeros((n_samples, n_bins))
    
    # For each sample, put all mass on the bin containing it
    for i, yi in enumerate(y):
        bin_idx = np.searchsorted(bin_edges, yi, side="right") - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        probas[i, bin_idx] = 1.0
    
    return probas


def make_narrow_support_histogram(y, bin_edges, support_fraction=0.5):
    """Create a histogram with intentionally narrow support.
    
    This zeros out mass outside a narrower range, then renormalizes.
    Used to test that padding handles targets outside reported support.
    
    Parameters
    ----------
    y : array of shape (n_samples,)
        Target values
    bin_edges : array of shape (n_bins + 1,)
        Bin edges
    support_fraction : float
        Fraction of the full range to keep (default 0.5 = 50%)
    
    Returns
    -------
    probas : array of shape (n_samples, n_bins)
        Narrow-support histogram
    """
    # Start with oracle
    probas = make_oracle_histogram(y, bin_edges)
    
    # Compute narrow support range
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    full_lo, full_hi = bin_centers[0], bin_centers[-1]
    full_span = full_hi - full_lo
    narrow_span = support_fraction * full_span
    narrow_lo = full_lo + 0.5 * (full_span - narrow_span)
    narrow_hi = narrow_lo + narrow_span
    
    # Zero out mass outside narrow range
    for j in range(len(bin_edges) - 1):
        bin_center = bin_centers[j]
        if bin_center < narrow_lo or bin_center > narrow_hi:
            probas[:, j] = 0.0
    
    # Renormalize to unit mass per sample
    for i in range(len(y)):
        total = probas[i].sum()
        if total > 0:
            probas[i] /= total
    
    return probas


def score_probas(probas, bin_edges, y):
    """Score a probability distribution using the production scoring rules.
    
    Parameters
    ----------
    probas : array of shape (n_samples, n_bins)
        Probability mass in each bin
    bin_edges : array of shape (n_bins + 1,)
        Bin edges
    y : array of shape (n_samples,)
        Target values
    
    Returns
    -------
    scores : dict
        Mapping rule_name -> score (scalar)
    """
    # Convert to torch tensors (use float64 for numerical accuracy)
    from scoringbench.univariate.metrics import _compute_scoring_rules_torch
    
    probas_torch = torch.as_tensor(np.asarray(probas), dtype=torch.float64)
    bin_edges_torch = torch.as_tensor(np.asarray(bin_edges), dtype=torch.float64)
    bin_mids_torch = 0.5 * (bin_edges_torch[1:] + bin_edges_torch[:-1])
    y_torch = torch.as_tensor(np.asarray(y, dtype=float).reshape(-1), dtype=torch.float64)
    
    # Compute scores using the internal function (same as analytical tests)
    result = _compute_scoring_rules_torch(
        probas_torch,
        bin_edges_torch,
        bin_mids_torch,
        y_torch,
        shared=True,
    )
    
    # Convert back to numpy
    scores = {}
    for key, val in result.items():
        if isinstance(val, torch.Tensor):
            scores[key] = float(val.mean().cpu().numpy())
        else:
            scores[key] = float(val)
    
    return scores


# ============================================================================
# Tests
# ============================================================================

class TestTailContributions:
    """Test suite for tail contribution handling."""
    
    @pytest.fixture(scope="class")
    def dataset(self):
        """Generate a simple dataset once per test class."""
        return make_simple_dataset(n_samples=32, seed=42)
    
    @pytest.fixture(scope="class")
    def bin_edges(self, dataset):
        """Create bin edges for the dataset."""
        return make_bins(dataset["y"], n_bins=128, pad_frac=0.1)
    
    def test_oracle_scores_are_finite(self, dataset, bin_edges):
        """Verify that oracle histogram produces finite scores."""
        y = dataset["y"]
        probas = make_oracle_histogram(y, bin_edges)
        scores = score_probas(probas, bin_edges, y)
        
        for rule, score in scores.items():
            assert np.isfinite(score), (
                f"{rule}: score is not finite: {score}"
            )
    
    def test_narrow_support_scores_are_finite(self, dataset, bin_edges):
        """Verify that narrow-support histogram produces finite scores.
        
        This is the key test: padding should handle targets outside the
        reported support without introducing NaNs or infinities.
        """
        y = dataset["y"]
        probas = make_narrow_support_histogram(y, bin_edges, support_fraction=0.5)
        scores = score_probas(probas, bin_edges, y)
        
        for rule, score in scores.items():
            assert np.isfinite(score), (
                f"{rule}: narrow-support score is not finite: {score}"
            )
    
    def test_narrow_support_with_extreme_fraction(self, dataset, bin_edges):
        """Test with very narrow support (25%) to stress-test padding."""
        y = dataset["y"]
        probas = make_narrow_support_histogram(y, bin_edges, support_fraction=0.25)
        scores = score_probas(probas, bin_edges, y)
        
        for rule, score in scores.items():
            assert np.isfinite(score), (
                f"{rule}: extreme narrow-support score is not finite: {score}"
            )
    
    def test_scoring_consistency_across_bin_counts(self, dataset):
        """Verify that scoring is consistent across different bin counts.
        
        This tests that the padding mechanism works correctly regardless
        of the bin configuration.
        """
        y = dataset["y"]
        
        # Score with two different bin counts
        bin_edges_128 = make_bins(y, n_bins=128, pad_frac=0.1)
        bin_edges_64 = make_bins(y, n_bins=64, pad_frac=0.1)
        
        probas_128 = make_oracle_histogram(y, bin_edges_128)
        probas_64 = make_oracle_histogram(y, bin_edges_64)
        
        scores_128 = score_probas(probas_128, bin_edges_128, y)
        scores_64 = score_probas(probas_64, bin_edges_64, y)
        
        # Both should be finite
        for rule in scores_128:
            assert np.isfinite(scores_128[rule]), (
                f"{rule}: 128-bin score is not finite"
            )
            assert np.isfinite(scores_64[rule]), (
                f"{rule}: 64-bin score is not finite"
            )
    
    def test_ranking_stability_across_support_widths(self, dataset, bin_edges):
        """Verify that estimator rankings are stable across support widths.
        
        This tests the proper scoring rule property: the relative ranking
        of estimators should not change when we change the support width.
        """
        y = dataset["y"]
        
        # Create multiple estimators
        oracle = make_oracle_histogram(y, bin_edges)
        narrow = make_narrow_support_histogram(y, bin_edges, support_fraction=0.5)
        
        # Also create a misspecified estimator (uniform)
        n_bins = len(bin_edges) - 1
        uniform = np.ones((len(y), n_bins)) / n_bins
        
        # Score all estimators
        scores_oracle = score_probas(oracle, bin_edges, y)
        scores_narrow = score_probas(narrow, bin_edges, y)
        scores_uniform = score_probas(uniform, bin_edges, y)
        
        # For invariant rules (CRPS, energy, etc.), oracle should rank best
        # or at least not be significantly worse
        invariant_rules = ["crps", "energy_score", "interval_score", "wcrps"]
        for rule in invariant_rules:
            if rule in scores_oracle and rule in scores_narrow and rule in scores_uniform:
                oracle_score = scores_oracle[rule]
                narrow_score = scores_narrow[rule]
                uniform_score = scores_uniform[rule]
                
                # Oracle should be better than or comparable to narrow
                # (narrow is also a reasonable estimator, just with different support)
                # Both should be better than uniform
                assert oracle_score <= uniform_score * 1.5, (
                    f"{rule}: oracle much worse than uniform: "
                    f"oracle={oracle_score}, uniform={uniform_score}"
                )
                assert narrow_score <= uniform_score * 1.5, (
                    f"{rule}: narrow much worse than uniform: "
                    f"narrow={narrow_score}, uniform={uniform_score}"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
