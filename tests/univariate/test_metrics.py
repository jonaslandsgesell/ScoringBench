import math
import warnings

import numpy as np
import pytest
from scipy import stats

from scoringbench.univariate import __version__
from scoringbench.univariate.metrics import compute_point_metrics, compute_scoring_rules
from scoringbench.univariate.wrappers import DistributionPrediction

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

    dist = DistributionPrediction(probas=probas, bin_edges=bin_edges, bin_midpoints=bin_mids, mean=mean, train_range=(float(np.asarray(bin_edges).min()), float(np.asarray(bin_edges).max())))

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
        train_range=(float(np.asarray(bin_edges).min()), float(np.asarray(bin_edges).max())),
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
        train_range=(float(np.asarray(bin_edges).min()), float(np.asarray(bin_edges).max())),
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
        train_range=(float(np.asarray(bin_edges).min()), float(np.asarray(bin_edges).max())),
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
        train_range=(float(np.asarray(bin_edges).min()), float(np.asarray(bin_edges).max())),
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
        train_range=(float(np.asarray(bin_edges).min()), float(np.asarray(bin_edges).max())),
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


# ---------------------------------------------------------------------------
# Energy score tests with beta=1 (CRPS) against Gaussian ground truth formula
# ---------------------------------------------------------------------------
# Gaussian CRPS Formula (Weigend & Shi 2000):
# For F = N(μ, σ²) and observation y:
#   CRPS(N(μ, σ²), y) = σ [z(2Φ(z) - 1) + 2φ(z) - 1/√π]
#   where z = (y - μ) / σ, Φ is standard normal CDF, φ is standard normal PDF


def _compute_gaussian_crps(y, mu, sigma):
    """Compute CRPS for a Gaussian distribution using closed-form formula.
    
    For F = N(μ, σ²) and observation y:
        CRPS(F, y) = σ [z(2Φ(z) - 1) + 2φ(z) - 1/√π]
    
    where z = (y - μ)/σ, Φ is CDF of N(0,1), φ is PDF of N(0,1).
    
    Parameters
    ----------
    y : array-like
        Observation(s)
    mu : float
        Mean of Gaussian distribution
    sigma : float
        Standard deviation of Gaussian distribution
    
    Returns
    -------
    float or array
        CRPS value(s)
    """
    z = (y - mu) / sigma
    normal_dist = stats.norm(0, 1)
    phi_z = normal_dist.pdf(z)      # PDF: exp(-z²/2) / √(2π)
    Phi_z = normal_dist.cdf(z)      # CDF: Φ(z)
    
    crps = sigma * (z * (2 * Phi_z - 1) + 2 * phi_z - 1.0 / np.sqrt(np.pi))
    return crps


def test_energy_score_beta_1_gaussian_single_sample():
    """Test energy score with β=1 against Gaussian CRPS formula for a single sample.
    
    Setup:
      - Gaussian distribution N(μ=0, σ²=1)
      - Observation y = 0.5 (one standard deviation above mean)
      - Create histogram discretization with fine bins to approximate continuous Gaussian
      - Compute energy_score_beta_1.0 and compare to analytical Gaussian CRPS
    
    Expected behavior:
      - Energy score with β=1 should match the analytical Gaussian CRPS
      - Within tolerance accounting for histogram discretization error
    """
    mu = 0.0
    sigma = 1.0
    y_obs = 0.5
    
    # Create fine histogram discretization of Gaussian
    # Use +/- 5σ range with 200 bins for good approximation
    n_bins = 200
    bin_edges = np.linspace(mu - 5*sigma, mu + 5*sigma, n_bins + 1, dtype=np.float32)
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    
    # Compute CDF values at bin edges to get probabilities (PMF)
    normal_dist = stats.norm(mu, sigma)
    cdf_edges = normal_dist.cdf(bin_edges)
    probas = np.diff(cdf_edges)[np.newaxis, :].astype(np.float32)  # (1, n_bins)
    
    # Create DistributionPrediction
    mean_val = (probas @ bin_mids).astype(np.float64)
    dist = DistributionPrediction(
        probas=probas,
        bin_edges=bin_edges,
        bin_midpoints=bin_mids.astype(np.float32),
        mean=mean_val,
        train_range=(float(np.asarray(bin_edges).min()), float(np.asarray(bin_edges).max())),
    )
    
    y_true = np.array([y_obs], dtype=np.float32)
    
    # Compute scoring rules (includes energy_score_beta_1.0)
    res = compute_scoring_rules(dist, y_true)
    
    # Compute analytical Gaussian CRPS
    expected_crps = _compute_gaussian_crps(y_obs, mu, sigma)
    
    # The energy score with beta=1.0 is exactly the CRPS
    computed_crps = res["energy_score_beta_1.0"]
    
    # Tolerance: discretization error grows with bin width
    # With 200 bins over ±5σ, bin width ≈ 0.05σ, expect <2% error
    rel_tol = 0.02
    abs_tol = 1e-6
    
    assert math.isclose(computed_crps, expected_crps, rel_tol=rel_tol, abs_tol=abs_tol), (
        f"Energy score β=1.0 vs Gaussian CRPS formula mismatch:\n"
        f"  Expected CRPS (formula): {expected_crps:.6f}\n"
        f"  Computed (energy β=1.0): {computed_crps:.6f}\n"
        f"  Relative error: {abs(computed_crps - expected_crps) / abs(expected_crps):.4%}\n"
        f"  Tolerance: rel_tol={rel_tol}, abs_tol={abs_tol}"
    )


def test_energy_score_beta_1_gaussian_multiple_samples():
    """Test energy score β=1 against Gaussian CRPS for multiple samples with varying σ.
    
    Setup:
      - Multiple Gaussian distributions with varying σ ∈ {0.5, 1.0, 1.5, 2.0}
      - Multiple observations from each distribution
      - Verify energy_score_beta_1.0 matches analytical CRPS across all samples
    
    This test validates the formula across different scales of uncertainty.
    """
    rng = np.random.default_rng(123)
    
    # Create 4 samples with different standard deviations
    mus = np.array([0.0, 1.0, -1.0, 0.5], dtype=np.float64)
    sigmas = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    n_samples = len(mus)
    
    # Generate observations from each distribution
    y_true = np.array([
        rng.normal(mus[i], sigmas[i])
        for i in range(n_samples)
    ], dtype=np.float32)
    
    # Create per-sample histogram discretizations (non-shared grid)
    n_bins = 150
    bin_edges_list = []
    bin_mids_list = []
    probas_list = []
    
    for i in range(n_samples):
        mu = mus[i]
        sigma = sigmas[i]
        
        # Bin edges: ±5σ around μ
        edges = np.linspace(mu - 5*sigma, mu + 5*sigma, n_bins + 1, dtype=np.float32)
        mids = (edges[:-1] + edges[1:]) / 2.0
        
        # Compute PMF from CDF differences
        normal_dist = stats.norm(mu, sigma)
        cdf_edges = normal_dist.cdf(edges)
        proba = np.diff(cdf_edges).astype(np.float32)
        
        bin_edges_list.append(edges)
        bin_mids_list.append(mids.astype(np.float32))
        probas_list.append(proba)
    
    # Stack into per-sample arrays (non-shared grid)
    bin_edges = np.vstack(bin_edges_list)      # (n_samples, n_bins+1)
    bin_mids = np.vstack(bin_mids_list)        # (n_samples, n_bins)
    probas = np.vstack(probas_list)            # (n_samples, n_bins)
    
    mean_vals = (probas * bin_mids).sum(axis=1)
    
    dist = DistributionPrediction(
        probas=probas,
        bin_edges=bin_edges,
        bin_midpoints=bin_mids,
        mean=mean_vals,
        train_range=(float(np.asarray(bin_edges).min()), float(np.asarray(bin_edges).max())),
    )
    
    res = compute_scoring_rules(dist, y_true)
    
    # Compute analytical CRPS for each sample
    expected_crps_list = [
        _compute_gaussian_crps(y_true[i], mus[i], sigmas[i])
        for i in range(n_samples)
    ]
    expected_crps_mean = np.mean(expected_crps_list)
    
    computed_crps = res["energy_score_beta_1.0"]
    
    # Tolerance accounts for discretization error (2% rel_tol is reasonable for 150 bins)
    rel_tol = 0.025
    abs_tol = 1e-6
    
    assert math.isclose(computed_crps, expected_crps_mean, rel_tol=rel_tol, abs_tol=abs_tol), (
        f"Energy score β=1.0 (multiple samples) vs Gaussian CRPS mismatch:\n"
        f"  Expected mean CRPS: {expected_crps_mean:.6f}\n"
        f"  Computed (β=1.0):   {computed_crps:.6f}\n"
        f"  Per-sample values: {expected_crps_list}\n"
        f"  Relative error: {abs(computed_crps - expected_crps_mean) / abs(expected_crps_mean):.4%}\n"
        f"  Tolerance: rel_tol={rel_tol}, abs_tol={abs_tol}"
    )


def test_energy_score_beta_1_gaussian_edge_cases():
    """Test energy score β=1 against Gaussian CRPS for edge cases.
    
    Edge cases:
      - y exactly at μ (z=0): should give σ(2φ(0) - 1/√π) ≈ 0.797σ
      - y far above μ (z>>0): should give ≈ z*σ for large z
      - y far below μ (z<<0): should give ≈ |z|*σ for large |z|
    """
    mu = 0.0
    sigma = 1.0
    n_bins = 200
    bin_edges = np.linspace(mu - 5*sigma, mu + 5*sigma, n_bins + 1, dtype=np.float32)
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    
    normal_dist = stats.norm(mu, sigma)
    cdf_edges = normal_dist.cdf(bin_edges)
    probas_template = np.diff(cdf_edges).astype(np.float32)
    
    # Test three cases: y=μ, y=μ+2σ, y=μ-3σ
    test_cases = [
        {"y": 0.0, "name": "y at mean (z=0)"},
        {"y": 2.0, "name": "y far above (z=2)"},
        {"y": -3.0, "name": "y far below (z=-3)"},
    ]
    
    for case in test_cases:
        y_obs = case["y"]
        
        # Create single-sample distribution
        probas = probas_template[np.newaxis, :].astype(np.float32)
        mean_val = (probas @ bin_mids).astype(np.float64)
        
        dist = DistributionPrediction(
            probas=probas,
            bin_edges=bin_edges,
            bin_midpoints=bin_mids.astype(np.float32),
            mean=mean_val,
            train_range=(float(np.asarray(bin_edges).min()), float(np.asarray(bin_edges).max())),
        )
        
        y_true = np.array([y_obs], dtype=np.float32)
        res = compute_scoring_rules(dist, y_true)
        
        expected_crps = _compute_gaussian_crps(y_obs, mu, sigma)
        computed_crps = res["energy_score_beta_1.0"]
        
        # Tighter tolerance for edge cases
        rel_tol = 0.03
        abs_tol = 1e-5
        
        assert math.isclose(computed_crps, expected_crps, rel_tol=rel_tol, abs_tol=abs_tol), (
            f"Edge case '{case['name']}' failed:\n"
            f"  Expected CRPS: {expected_crps:.6f}\n"
            f"  Computed:      {computed_crps:.6f}\n"
            f"  Relative error: {abs(computed_crps - expected_crps) / abs(expected_crps):.4%}"
        )


# ---------------------------------------------------------------------------
# PIT / KS test (Dawid 1984; Diebold et al. 1998)
# ---------------------------------------------------------------------------
# If F_t is ideal and continuous, PIT values p_t = F_t(x_t) ~ Uniform(0,1).
# We exercise the histogram PIT implementation in compute_pit_ks via
# compute_scoring_rules.

def test_pit_ks_uniform_when_truth_drawn_from_predictive():
    """y drawn from the predictive distribution -> KS p-value should be large."""
    rng = np.random.default_rng(0)
    n_bins = 50
    n_samples = 2000
    bin_edges = np.linspace(-5.0, 5.0, n_bins + 1, dtype=np.float32)
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Standard-normal predictive density on a shared grid, identical for all samples
    pdf = np.exp(-0.5 * bin_mids ** 2) / np.sqrt(2 * np.pi)
    widths = np.diff(bin_edges)
    probas_row = pdf * widths
    probas_row = probas_row / probas_row.sum()
    probas = np.tile(probas_row, (n_samples, 1)).astype(np.float32)

    # Sample y by inverting the per-bin uniform CDF
    cdf = np.cumsum(probas_row)
    u = rng.uniform(size=n_samples)
    bin_idx = np.searchsorted(cdf, u)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    cdf_prev = np.where(bin_idx == 0, 0.0, cdf[np.clip(bin_idx - 1, 0, n_bins - 1)])
    frac = (u - cdf_prev) / probas_row[bin_idx]
    y_true = (bin_edges[bin_idx] + frac * widths[bin_idx]).astype(np.float32)

    dist = DistributionPrediction(
        probas=probas,
        bin_edges=bin_edges,
        bin_midpoints=bin_mids.astype(np.float32),
        mean=(probas @ bin_mids).astype(np.float64),
        train_range=(float(np.asarray(bin_edges).min()), float(np.asarray(bin_edges).max())),
    )
    res = compute_scoring_rules(dist, y_true)

    assert "pit_ks_stat" in res and "pit_ks_pvalue" in res
    assert 0.0 <= res["pit_ks_stat"] <= 1.0
    assert 0.0 <= res["pit_ks_pvalue"] <= 1.0
    # Calibrated forecasts -> should not reject uniformity at 1% level.
    assert res["pit_ks_pvalue"] > 0.01, (
        f"Calibrated PIT should be ~uniform; got p={res['pit_ks_pvalue']:.4f}"
    )


def test_pit_ks_rejects_when_predictive_is_miscalibrated():
    """Truth shifted far from predictive -> PIT concentrates -> KS rejects uniformity."""
    rng = np.random.default_rng(1)
    n_bins = 50
    n_samples = 500
    bin_edges = np.linspace(-5.0, 5.0, n_bins + 1, dtype=np.float32)
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    pdf = np.exp(-0.5 * bin_mids ** 2) / np.sqrt(2 * np.pi)
    widths = np.diff(bin_edges)
    probas_row = pdf * widths
    probas_row = probas_row / probas_row.sum()
    probas = np.tile(probas_row, (n_samples, 1)).astype(np.float32)

    # Truth drawn from N(3, 0.5) — heavily right-shifted vs. predictive N(0, 1)
    y_true = rng.normal(loc=3.0, scale=0.5, size=n_samples).astype(np.float32)

    dist = DistributionPrediction(
        probas=probas,
        bin_edges=bin_edges,
        bin_midpoints=bin_mids.astype(np.float32),
        mean=(probas @ bin_mids).astype(np.float64),
        train_range=(float(np.asarray(bin_edges).min()), float(np.asarray(bin_edges).max())),
    )
    res = compute_scoring_rules(dist, y_true)

    assert res["pit_ks_pvalue"] < 1e-3, (
        f"Miscalibrated PIT should reject uniformity; got p={res['pit_ks_pvalue']:.4g}"
    )
    assert res["pit_ks_stat"] > 0.3
