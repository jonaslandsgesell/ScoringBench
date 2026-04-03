import math
import warnings

import numpy as np
import pytest

from scoringbench import __version__
from scoringbench.metrics import compute_point_metrics, compute_scoring_rules
from scoringbench.wrappers import DistributionPrediction

# Force CPU so the test runs on machines without a working CUDA device.
try:
    import torch as _torch
    _torch.cuda.is_available = lambda: False
except Exception:
    _torch = None


def test_scoringbench_imports():
    # basic smoke-test: package imports and exposes a version
    assert isinstance(__version__, str)


def test_compute_point_metrics_basic():
    y_true = np.array([0.0, 1.0])
    y_pred = np.array([0.0, 2.0])

    res = compute_point_metrics(y_true, y_pred)

    assert math.isclose(res["mae"], 0.5, rel_tol=1e-9)
    assert math.isclose(res["rmse"], math.sqrt(0.5), rel_tol=1e-9)
    # For this simple example R^2 is -1.0
    assert math.isclose(res["r2"], -1.0, rel_tol=1e-9)


def test_distribution_prediction_container():
    # Ensure DistributionPrediction dataclass accepts arrays and exposes fields
    probas = np.array([[0.5, 0.5], [0.2, 0.8]])
    bin_edges = np.array([0.0, 0.5, 1.0])
    bin_mids = np.array([0.25, 0.75])
    mean = np.array([0.25, 0.8])

    dist = DistributionPrediction(probas=probas, bin_edges=bin_edges, bin_midpoints=bin_mids, mean=mean)

    assert dist.probas.shape == (2, 2)
    assert dist.bin_edges.shape[0] == 3
    assert dist.bin_midpoints.shape == (2,)
    assert dist.mean.shape == (2,)


# ---------------------------------------------------------------------------
# Quantile-weighted CRPS tests (Gneiting & Ranjan 2011, Eq. 17)
# ---------------------------------------------------------------------------
# Weight functions:
#   wCRPS_left   v(α) = (1-α)²  — emphasises small α (lower quantiles)
#   wCRPS_right  v(α) = α²      — emphasises large α (upper quantiles)
#   wCRPS_center v(α) = α(1-α)  — emphasises α ≈ 0.5
#
# Key identity: when distribution mass is concentrated to the RIGHT of y,
# the lower quantiles (small α) deviate most from the truth, so
# wCRPS_left > wCRPS_right.  The symmetric argument holds for the mirror case.


def _make_dist_shared(probas_row, bin_edges, n_samples):
    """Helper: replicate a single probability row into a shared-grid DistributionPrediction."""
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    probas = np.tile(probas_row, (n_samples, 1)).astype(np.float32)
    mean = (probas @ bin_mids).astype(np.float64)
    return DistributionPrediction(
        probas=probas,
        bin_edges=bin_edges.astype(np.float32),
        bin_midpoints=bin_mids.astype(np.float32),
        mean=mean,
    )


def test_wcrps_left_dominated_when_distribution_right_of_truth_shared():
    """Shared grid: distribution concentrated right ⟹ wCRPS_left > wCRPS_right.

    mass at right (bin 2), truth at left (bin 0):
      lower quantiles deviate most → left-tail weight penalises most.
    """
    # 3 bins: [0,1), [1,2), [2,3]  midpoints 0.5, 1.5, 2.5
    bin_edges = np.array([0.0, 1.0, 2.0, 3.0])
    probas_row = np.array([0.01, 0.01, 0.98], dtype=np.float32)
    dist = _make_dist_shared(probas_row, bin_edges, n_samples=20)
    y_true = np.full(20, 0.5, dtype=np.float32)  # truth in leftmost bin

    res = compute_scoring_rules(dist, y_true)

    assert "wcrps_left" in res and "wcrps_right" in res and "wcrps_center" in res
    assert res["wcrps_left"] > res["wcrps_right"], (
        f"expected wcrps_left ({res['wcrps_left']:.4f}) > "
        f"wcrps_right ({res['wcrps_right']:.4f}) "
        "when distribution mass is right of truth"
    )
    # wCRPS_center should also be smaller than wCRPS_left in this regime
    assert res["wcrps_left"] > res["wcrps_center"], (
        "wcrps_left should dominate wcrps_center when distribution is right of truth"
    )


def test_wcrps_right_dominated_when_distribution_left_of_truth_shared():
    """Shared grid: distribution concentrated left ⟹ wCRPS_right > wCRPS_left.

    mass at left (bin 0), truth at right (bin 2):
      upper quantiles deviate most → right-tail weight penalises most.
    """
    bin_edges = np.array([0.0, 1.0, 2.0, 3.0])
    probas_row = np.array([0.98, 0.01, 0.01], dtype=np.float32)
    dist = _make_dist_shared(probas_row, bin_edges, n_samples=20)
    y_true = np.full(20, 2.5, dtype=np.float32)  # truth in rightmost bin

    res = compute_scoring_rules(dist, y_true)

    assert res["wcrps_right"] > res["wcrps_left"], (
        f"expected wcrps_right ({res['wcrps_right']:.4f}) > "
        f"wcrps_left ({res['wcrps_left']:.4f}) "
        "when distribution mass is left of truth"
    )
    assert res["wcrps_right"] > res["wcrps_center"], (
        "wcrps_right should dominate wcrps_center when distribution is left of truth"
    )


def test_wcrps_left_dominated_when_distribution_right_of_truth_nonshared():
    """Non-shared (per-sample) grid: same directional property holds.

    Uses unequal bin widths to exercise the torch.gather path.
    """
    # Two different bin-edge layouts, each with 3 bins
    #   Sample 0–4: [0, 0.5, 1.5, 3.0]  — narrow left bin, wide right bin
    #   Sample 5–9: [0, 1.0, 2.0, 3.0]  — equal widths
    n_samples = 10
    edges_a = np.array([0.0, 0.5, 1.5, 3.0], dtype=np.float32)
    edges_b = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    bin_edges = np.vstack([np.tile(edges_a, (5, 1)), np.tile(edges_b, (5, 1))])  # (10, 4)
    bin_mids = (bin_edges[:, :-1] + bin_edges[:, 1:]) / 2.0                      # (10, 3)

    # All samples: almost all mass in the rightmost bin
    probas = np.tile(np.array([0.01, 0.01, 0.98], dtype=np.float32), (n_samples, 1))
    mean = (probas * bin_mids).sum(axis=1)

    # Truth: in the leftmost bin of each sample
    y_true = bin_mids[:, 0].astype(np.float32)

    dist = DistributionPrediction(
        probas=probas,
        bin_edges=bin_edges,
        bin_midpoints=bin_mids.astype(np.float32),
        mean=mean,
    )
    res = compute_scoring_rules(dist, y_true)

    assert res["wcrps_left"] > res["wcrps_right"], (
        f"Non-shared grid: expected wcrps_left ({res['wcrps_left']:.4f}) > "
        f"wcrps_right ({res['wcrps_right']:.4f})"
    )


def test_wcrps_symmetric_distribution_left_right_equal():
    """Perfectly symmetric setup ⟹ wCRPS_left ≈ wCRPS_right."""
    # Single bin carrying all mass exactly at the mean; truth symmetric around it.
    # 4 bins, uniform distribution, y_true = midrange → left/right equally wrong.
    bin_edges = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    probas_row = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    dist = _make_dist_shared(probas_row, bin_edges, n_samples=100)
    rng = np.random.default_rng(42)
    # Draw y uniformly from [-2, 2] so the distribution is unbiased in expectation
    y_true = rng.uniform(-2.0, 2.0, size=100).astype(np.float32)

    res = compute_scoring_rules(dist, y_true)

    # With large n and symmetric y, left and right should be very close
    assert abs(res["wcrps_left"] - res["wcrps_right"]) < 0.15, (
        f"Symmetric case: wcrps_left={res['wcrps_left']:.4f}, "
        f"wcrps_right={res['wcrps_right']:.4f} — difference too large"
    )


def test_wcrps_analytical_uniform_distribution():
    """Analytical test: uniform distribution across 5 bins with truth at different positions.

    Setup:
      - 5 equal bins: [0, 1), [1, 2), [2, 3), [3, 4), [4, 5]
      - Uniform probability: 0.2 in each bin
      - Midpoints: [0.5, 1.5, 2.5, 3.5, 4.5]
      - y_true = 0.2 (well below median 2.5, in first bin)

    Expected behavior:
      - Median of distribution: 2.5
      - Truth at 0.2 is well below median, so lower quantiles have smaller deviations
      - wCRPS_right should dominate (upper quantiles are far from y)
      - wCRPS_left penalizes lower quantiles that are close to y
      - wCRPS_center is intermediate

    Test verifies the directional properties and that all three metrics are positive.
    """
    bin_edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    bin_mids = np.array([0.5, 1.5, 2.5, 3.5, 4.5], dtype=np.float32)
    probas = np.array([[0.2, 0.2, 0.2, 0.2, 0.2]], dtype=np.float32)  # Uniform
    mean = np.array([2.5], dtype=np.float64)
    y_true = np.array([0.2], dtype=np.float32)  # Truth well below median (in first bin)

    dist = DistributionPrediction(
        probas=probas,
        bin_edges=bin_edges,
        bin_midpoints=bin_mids,
        mean=mean,
    )

    res = compute_scoring_rules(dist, y_true)

    # Verify all three metrics are positive and reasonable
    assert res["wcrps_left"] > 0, "wcrps_left should be positive"
    assert res["wcrps_right"] > 0, "wcrps_right should be positive"
    assert res["wcrps_center"] > 0, "wcrps_center should be positive"

    # Verify ordering: truth well below distribution median
    # When truth is well below median, upper quantiles deviate much more, so right-tail weighting dominates
    assert res["wcrps_right"] > res["wcrps_left"], (
        f"Uniform distribution: expected wcrps_right ({res['wcrps_right']:.4f}) > "
        f"wcrps_left ({res['wcrps_left']:.4f}) when truth is well below mean"
    )
    
    # Center weighting should be smaller than right in this extreme case
    assert res["wcrps_center"] < res["wcrps_right"], (
        f"Uniform distribution: expected wcrps_center ({res['wcrps_center']:.4f}) < "
        f"wcrps_right ({res['wcrps_right']:.4f}) when truth is well below mean"
    )


def test_wcrps_analytical_single_bin():
    """Analytical test: single-bin Dirac-like distribution with known closed-form wCRPS.

    Setup:
      - 1 bin [0, 1] with midpoint 0.5, all mass concentrated here
      - y_true = 0.3 (below the quantiles q_α = 0.5)
      - For all α: pinball(α) = 2(I[0.3 ≤ 0.5] - α)(0.5 - 0.3) = 2(1 - α)(0.2) = 0.4(1 - α)

    Expected wCRPS (via numerical integration):
      wCRPS_v = ∫₀¹ 0.4(1-α) v(α) dα

    For v_left(α) = (1-α)²:    wCRPS_left ≈ 0.1333
    For v_right(α) = α²:        wCRPS_right ≈ 0.0333
    For v_center(α) = α(1-α):   wCRPS_center ≈ 0.0667
    """
    # Create single-bin distribution: [0, 1] with all mass at midpoint 0.5
    bin_edges = np.array([0.0, 1.0], dtype=np.float32)
    bin_mids = np.array([0.5], dtype=np.float32)
    probas = np.array([[1.0]], dtype=np.float32)  # Single sample, all mass in bin 0
    mean = np.array([0.5], dtype=np.float64)
    y_true = np.array([0.3], dtype=np.float32)  # Truth below quantile

    dist = DistributionPrediction(
        probas=probas,
        bin_edges=bin_edges,
        bin_midpoints=bin_mids,
        mean=mean,
    )

    res = compute_scoring_rules(dist, y_true)

    # Analytically compute expected wCRPS values via numerical integration
    # For pinball(α) = 0.4(1-α), integrate with weight functions
    alphas = np.linspace(0.001, 0.999, 10000)
    d_alpha = 1.0 / (len(alphas) + 1)
    pinball_vals = 0.4 * (1.0 - alphas)  # All samples have y=0.3 < q=0.5
    
    v_left = (1.0 - alphas) ** 2
    v_right = alphas ** 2
    v_center = alphas * (1.0 - alphas)

    expected_wcrps_left = np.sum(pinball_vals * v_left) * d_alpha
    expected_wcrps_right = np.sum(pinball_vals * v_right) * d_alpha
    expected_wcrps_center = np.sum(pinball_vals * v_center) * d_alpha

    # Verify computed values against analytical expectations (within 2.5% tolerance for numerical precision)
    assert math.isclose(res["wcrps_left"], expected_wcrps_left, rel_tol=0.025), (
        f"wcrps_left: expected {expected_wcrps_left:.6f}, got {res['wcrps_left']:.6f}"
    )
    assert math.isclose(res["wcrps_right"], expected_wcrps_right, rel_tol=0.025), (
        f"wcrps_right: expected {expected_wcrps_right:.6f}, got {res['wcrps_right']:.6f}"
    )
    assert math.isclose(res["wcrps_center"], expected_wcrps_center, rel_tol=0.025), (
        f"wcrps_center: expected {expected_wcrps_center:.6f}, got {res['wcrps_center']:.6f}"
    )

    # Verify ordering: when truth is below all quantiles,
    # pinball increases with α, so left-tail weighting should give largest value
    assert res["wcrps_left"] > res["wcrps_right"], (
        f"Expected wcrps_left ({res['wcrps_left']:.4f}) > "
        f"wcrps_right ({res['wcrps_right']:.4f}) "
        "when truth is below distribution"
    )


def test_wcrps_exact_values_with_epsilon():
    """Exact value test: verify specific wCRPS values match analytical expectations within epsilon.

    Setup:
      - 1 bin [0, 1] with all mass at midpoint 0.5
      - y_true = 0.3 (below quantile)
      - Closed-form analytical solution possible

    Analytical Solution:
      For all α: pinball(α) = 2(I[0.3 ≤ 0.5] - α)(0.5 - 0.3) = 0.4(1 - α)

      wCRPS_left = ∫₀¹ 0.4(1-α)(1-α)² dα = 0.4 * 1/4 = 0.1
      wCRPS_right = ∫₀¹ 0.4(1-α)α² dα = 0.4 * 1/12 ≈ 0.0333...
      wCRPS_center = ∫₀¹ 0.4(1-α)α(1-α) dα = 0.4 * 1/12 ≈ 0.0333...

    The discrete implementation uses 99 quantile levels, so there's ~1% discretization error.
    """
    bin_edges = np.array([0.0, 1.0], dtype=np.float32)
    bin_mids = np.array([0.5], dtype=np.float32)
    probas = np.array([[1.0]], dtype=np.float32)
    mean = np.array([0.5], dtype=np.float64)
    y_true = np.array([0.3], dtype=np.float32)

    dist = DistributionPrediction(
        probas=probas,
        bin_edges=bin_edges,
        bin_midpoints=bin_mids,
        mean=mean,
    )

    res = compute_scoring_rules(dist, y_true)

    # Analytical values computed from closed-form integrals
    expected_wcrps_left = 0.1
    expected_wcrps_right = 0.4 / 12.0  # ≈ 0.0333...
    expected_wcrps_center = 0.4 / 12.0  # ≈ 0.0333...

    # Tolerance accounts for:
    # 1. Discretization: 99 quantile levels instead of continuous integration
    # 2. Numerical precision: finite precision arithmetic in torch
    epsilon = 0.01  # 1% tolerance for discretization + numerical error

    assert abs(res["wcrps_left"] - expected_wcrps_left) < epsilon, (
        f"wcrps_left exact value test failed: "
        f"expected {expected_wcrps_left:.6f}, got {res['wcrps_left']:.6f}, "
        f"error {abs(res['wcrps_left'] - expected_wcrps_left):.6f}, epsilon {epsilon}"
    )
    assert abs(res["wcrps_right"] - expected_wcrps_right) < epsilon, (
        f"wcrps_right exact value test failed: "
        f"expected {expected_wcrps_right:.6f}, got {res['wcrps_right']:.6f}, "
        f"error {abs(res['wcrps_right'] - expected_wcrps_right):.6f}, epsilon {epsilon}"
    )
    assert abs(res["wcrps_center"] - expected_wcrps_center) < epsilon, (
        f"wcrps_center exact value test failed: "
        f"expected {expected_wcrps_center:.6f}, got {res['wcrps_center']:.6f}, "
        f"error {abs(res['wcrps_center'] - expected_wcrps_center):.6f}, epsilon {epsilon}"
    )

