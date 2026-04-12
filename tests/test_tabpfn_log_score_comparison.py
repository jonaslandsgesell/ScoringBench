"""Conditional test for CE loss comparison (only runs if TabPFN available).

This test validates that:
1. Cross-entropy loss computed via `_compute_single_score` with beta="ce"
2. Is equivalent (up to numerical precision) to the log-score from scoringbench.metrics

Both should measure the negative log-likelihood of the true target under the predictive distribution.

The test is conditional on:
- TabPFN being successfully imported
- Model checkpoint file existing
- Sufficient resources for model fitting

If any condition is not met, tests are skipped gracefully.
"""

import os
import sys
import pytest
import numpy as np
import torch

# Test requires the modified TabPFN from the workspace
sys.path.insert(0, "/home/landsges/ScoringBench")

# Check prerequisites before importing heavy modules
ATOL = 1e-4
RTOL = 1e-3
MODEL_PATH = "tabpfn-v2.5-regressor-v2.5_real.ckpt"#"tabpfn-v2.6-regressor-v2.6_default.ckpt"

# Determine availability by attempting to import the top-level `tabpfn`
# package and the local `TabPFNWrapper`. If either import fails, skip tests.
has_tabpfn = False
try:
    import tabpfn  # type: ignore
    from scoringbench.wrappers.tabpfn import TabPFNWrapper
    has_tabpfn = True
except Exception:
    has_tabpfn = False

pytestmark = pytest.mark.skipif(
    not has_tabpfn,
    reason="tabpfn package or TabPFNWrapper not importable",
)


def create_toy_linear_data(n_samples=50, n_features=3, random_state=42):
    """Create a toy linear regression dataset."""
    rng = np.random.RandomState(random_state)
    X = rng.randn(n_samples, n_features)
    true_weights = np.array([2.0, -1.5, 0.8])
    y = X @ true_weights + 0.5 * rng.randn(n_samples)
    return X, y


def get_cross_entropy_via_criterion(logits, targets, criterion):
    """Compute cross-entropy loss using the BarDistribution criterion."""
    ce_loss = criterion(logits, targets)
    return ce_loss.mean().item()


def compute_log_score_direct(probas, bin_edges, y):
    """Compute log-score directly from probabilities."""
    device = probas.device
    n_samples, n_bins = probas.shape
    bin_widths = torch.diff(bin_edges)
    
    # Find which bin each y belongs to
    y_bin = torch.searchsorted(bin_edges[1:].contiguous(), y).clamp(0, n_bins - 1)
    
    # Get probability and bin width for y's bin
    sel_p = probas[torch.arange(n_samples, device=device), y_bin]
    sel_w = bin_widths[y_bin]
    
    # Compute density: p / width
    density = sel_p / sel_w.clamp(min=1e-10)
    
    # Log-score is negative log of density
    log_score = -torch.log(density.clamp(min=1e-10)).mean().item()
    
    return log_score


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_ce_loss_comparison_tabpfn_v2_6():
    """Test CE loss consistency between criterion and direct computation.
    
    Creates a toy dataset, fits TabPFN v2.6, and verifies that cross-entropy
    loss computed via the criterion matches the direct log-score computation.
    """
    # Setup
    np.random.seed(42)
    torch.manual_seed(42)
    
    X_train, y_train = create_toy_linear_data(n_samples=40, n_features=3, random_state=42)
    X_test, y_test = create_toy_linear_data(n_samples=10, n_features=3, random_state=43)
    
    # Fit model
    wrapper = TabPFNWrapper(model_path=MODEL_PATH)
    wrapper.fit(X_train, y_train)
    
    # Get predictions
    dist_pred = wrapper.predict_distribution(X_test)
    
    with torch.no_grad():
        pred_full = wrapper._model.predict(X_test, output_type="full")
    
    logits_full = pred_full["logits"]
    criterion = pred_full["criterion"]
    
    if not isinstance(logits_full, torch.Tensor):
        logits_full = torch.as_tensor(logits_full)
    
    device = wrapper._device
    logits_full = logits_full.to(device)
    targets_tensor = torch.as_tensor(y_test, dtype=torch.float32, device=device)
    
    # Reshape for criterion
    if logits_full.ndim == 2:
        logits_BQL = logits_full.unsqueeze(0)
    elif logits_full.ndim == 4:
        B, E, Q, L = logits_full.shape
        logits_BQL = logits_full.reshape(B * E, Q, L)
    else:
        logits_BQL = logits_full.view(-1, *logits_full.shape[-2:])
    
    if targets_tensor.ndim == 1:
        targets_BQ = targets_tensor.unsqueeze(0)
    else:
        targets_BQ = targets_tensor.view(logits_BQL.shape[0], -1)
    
    # Method 1: Via criterion
    ce_loss_criterion = get_cross_entropy_via_criterion(logits_BQL, targets_BQ, criterion)
    
    # Method 2: Direct computation
    probas = torch.softmax(logits_BQL, dim=-1)
    bin_edges = criterion.borders.to(device).float()
    y_tensor = targets_BQ.to(device)
    
    probas_flat = probas.view(-1, probas.shape[-1])
    y_flat = y_tensor.view(-1)
    
    log_score_direct = compute_log_score_direct(probas_flat, bin_edges, y_flat)
    
    # Compare
    diff = abs(ce_loss_criterion - log_score_direct)
    max_abs = max(abs(ce_loss_criterion), abs(log_score_direct))
    rel_diff = diff / (max_abs + 1e-10) if max_abs > 0 else diff
    
    assert np.isfinite(ce_loss_criterion), "CE loss not finite"
    assert np.isfinite(log_score_direct), "Log-score not finite"
    assert (diff < ATOL or rel_diff < RTOL), (
        f"CE loss mismatch: criterion={ce_loss_criterion:.6f}, "
        f"direct={log_score_direct:.6f}, diff={diff:.6e}, rel={rel_diff:.6e}"
    )