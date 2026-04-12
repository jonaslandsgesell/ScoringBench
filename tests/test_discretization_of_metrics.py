"""Test stability of metrics under discretization.

Tests verify that all distributional metrics in ScoringBench are robust
to discretization level changes. Metrics should be reasonably stable when
increasing grid resolution from N=50 to N=100 points.

Optimized with PyTorch for batch processing of synthetic distributions.
"""

import numpy as np
import pytest
import torch

# Force CPU
torch.cuda.is_available = lambda: False

from scoringbench.metrics import compute_scoring_rules, ENERGY_BETAS, CRESSIE_READ_LAMBDAS
from scoringbench.wrappers import DistributionPrediction


# ---------------------------------------------------------------------------
# Test Configuration Constants (Magic Numbers)
# ---------------------------------------------------------------------------

# Grid range for synthetic distributions (covers roughly ±4σ for unit normal)
_GRID_MIN, _GRID_MAX = -4.0, 4.0

# Distribution parameters for testing
# f (true/forecaster): mean=0, std=1 (standard normal)
# g (model/ground-truth): mean=0.7, std=1.2 (shifted and scaled normal)
_MU_F, _SIGMA_F = 0.0, 1.0
_MU_G, _SIGMA_G = 0.7, 1.2

# Per-sample variation factors for creating realistic diversity
# (keep variation small to maintain convergence properties across resolutions)
_MU_STD_FACTOR = 0.05  # Sample mus ~ N(base_mu, base_sigma * 0.05)
_SIGMA_LOWER_FACTOR = 0.95  # Sample sigmas ~ U(base_sigma * 0.95, base_sigma * 1.05)
_SIGMA_UPPER_FACTOR = 1.05


# ---------------------------------------------------------------------------
# Torch-based Synthetic Distribution Helpers
# ---------------------------------------------------------------------------

def normal_pdf_torch(x, mu, sigma):
    """Probability density function of normal distribution (torch)."""
    return (1.0 / (sigma * torch.sqrt(torch.tensor(2.0 * np.pi)))) * torch.exp(-0.5 * ((x - mu) / sigma) ** 2)


def make_discretized_distributions_batch(x_grids, mus, sigmas, n_samples_list):
    """Vectorized creation of multiple DistributionPredictions.
    
    Parameters
    ----------
    x_grids : list of np.ndarray
        Grid points for each discretization level.
    mus : list of float
        Means for distributions (true and model).
    sigmas : list of float
        Stds for distributions (true and model).
    n_samples_list : list of int
        Number of samples for each test.
        
    Yields
    ------
    (grid_name, n_samples, DistributionPrediction)
        Tuples for easy unpacking.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for grid_name, x_grid in x_grids.items():
        x_grid_torch = torch.as_tensor(x_grid, dtype=torch.float32, device=device)
        x_grid_np = np.asarray(x_grid, dtype=np.float32)
        
        # Get bin midpoints
        bin_mids = (x_grid_np[:-1] + x_grid_np[1:]) / 2.0
        bin_widths = np.diff(x_grid_np)
        bin_mids_torch = torch.as_tensor(bin_mids, dtype=torch.float32, device=device)
        bin_widths_torch = torch.as_tensor(bin_widths, dtype=torch.float32, device=device)
        
        for n_samples in n_samples_list:
            # Create distributions for both true (f) and model (g)
            for label, (base_mu, base_sigma) in [("f", (_MU_F, _SIGMA_F)), ("g", (_MU_G, _SIGMA_G))]:
                # Generate varied samples by varying mu slightly for each sample
                # This creates different distributions per sample to test dispersion
                # Use smaller variation to ensure stability across resolutions
                rng = np.random.RandomState(hash((grid_name, label, n_samples)) % 2**31)
                mus = rng.normal(base_mu, base_sigma * _MU_STD_FACTOR, size=n_samples)
                sigmas = rng.uniform(base_sigma * _SIGMA_LOWER_FACTOR, base_sigma * _SIGMA_UPPER_FACTOR, size=n_samples)
                
                probas_array = []
                mean_array = []
                
                for sample_idx in range(n_samples):
                    mu_torch = torch.tensor(mus[sample_idx], dtype=torch.float32, device=device)
                    sigma_torch = torch.tensor(sigmas[sample_idx], dtype=torch.float32, device=device)
                    
                    # Compute PDF at bin midpoints
                    pdf_vals = normal_pdf_torch(bin_mids_torch, mu_torch, sigma_torch)
                    probas = pdf_vals * bin_widths_torch
                    probas = probas / probas.sum()  # Normalize
                    
                    probas_array.append(probas.cpu().numpy())
                    mean_array.append(mus[sample_idx])
                
                probas_array = np.array(probas_array, dtype=np.float32)
                mean_array = np.array(mean_array, dtype=np.float32)
                
                yield (
                    f"{grid_name}_{label}_{n_samples}",
                    DistributionPrediction(
                        probas=probas_array,
                        bin_edges=x_grid_np.astype(np.float32),
                        bin_midpoints=bin_mids.astype(np.float32),
                        mean=mean_array,
                    ),
                )



def make_discretized_distribution(x_grid, mu, sigma, n_samples=10):
    """Create a DistributionPrediction from a normal distribution on a grid.
    
    Parameters
    ----------
    x_grid : np.ndarray
        Grid points (bin edges).
    mu : float
        Mean of the normal distribution.
    sigma : float
        Standard deviation of the normal distribution.
    n_samples : int
        Number of i.i.d. samples to create.
        
    Returns
    -------
    DistributionPrediction
        Distribution prediction with shared bin grid.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ensure grid is sorted
    x_grid = np.sort(np.asarray(x_grid, dtype=np.float32))
    x_grid_torch = torch.as_tensor(x_grid, dtype=torch.float32, device=device)
    
    # Get bin midpoints
    bin_mids = (x_grid[:-1] + x_grid[1:]) / 2.0
    bin_mids_torch = torch.as_tensor(bin_mids, dtype=torch.float32, device=device)
    bin_widths_torch = torch.diff(x_grid_torch, dim=-1)
    
    # Compute PDF at midpoints and normalize to get probabilities
    mu_t = torch.tensor(mu, dtype=torch.float32, device=device)
    sigma_t = torch.tensor(sigma, dtype=torch.float32, device=device)
    pdf_vals = normal_pdf_torch(bin_mids_torch, mu_t, sigma_t)
    probas = pdf_vals * bin_widths_torch
    probas = probas / probas.sum()  # Normalize to sum to 1
    
    # Create n_samples identical samples
    probas_np = probas.cpu().numpy()
    probas_array = np.tile(probas_np[np.newaxis, :], (n_samples, 1)).astype(np.float32)
    bin_edges_array = x_grid.astype(np.float32)
    bin_mids_array = bin_mids.astype(np.float32)
    
    # Compute mean
    mean_array = np.full(n_samples, mu, dtype=np.float32)
    
    return DistributionPrediction(
        probas=probas_array,
        bin_edges=bin_edges_array,
        bin_midpoints=bin_mids_array,
        mean=mean_array,
    )


# ---------------------------------------------------------------------------
# Fixtures and Common Setup
# ---------------------------------------------------------------------------

@pytest.fixture(params=[30, 40, 50])
def n_samples(request):
    """Parametrize number of samples."""
    return request.param


@pytest.fixture
def discretizations():
    """Create two discretization levels (N=50 and N=100)."""
    return {
        "x_50": np.linspace(_GRID_MIN, _GRID_MAX, 51),
        "x_100": np.linspace(_GRID_MIN, _GRID_MAX, 101),
    }


# Batch precomputed distributions for faster access
_batch_cache = {}

def get_batch_distributions(n_samples):
    """Get precomputed batch of all distribution configurations for n_samples."""
    if n_samples in _batch_cache:
        return _batch_cache[n_samples]
    
    discretizations = {
        "x_50": np.linspace(_GRID_MIN, _GRID_MAX, 51),
        "x_100": np.linspace(_GRID_MIN, _GRID_MAX, 101),
    }
    
    batch = {}
    for name, dist in make_discretized_distributions_batch(
        discretizations, [_MU_F, _MU_G], [_SIGMA_F, _SIGMA_G], [n_samples]
    ):
        batch[name] = dist
    
    _batch_cache[n_samples] = batch
    return batch



def assert_metric_stability(val_50, val_100, metric_name, threshold=None, is_coverage=False):
    """Helper to check metric stability between two discretization levels.
    
    Parameters
    ----------
    val_50 : float
        Metric value at N=50 discretization.
    val_100 : float
        Metric value at N=100 discretization.
    metric_name : str
        Name of the metric (for error messages).
    threshold : float, optional
        Threshold for stability check. Default depends on metric type.
    is_coverage : bool
        If True, use absolute difference; otherwise use relative difference.
    """
    if threshold is None:
        threshold = 0.10  # Default 10% relative difference
    
    if is_coverage:
        diff = abs(val_100 - val_50)
        check_passed = diff < threshold
        diff_str = f"abs_diff={diff:.4f}"
    else:
        diff = abs(val_100 - val_50) / (abs(val_50) + 1e-10)
        check_passed = diff < threshold
        diff_str = f"rel_diff={diff:.4f}"
    
    assert check_passed, (
        f"{metric_name} discretization instability: "
        f"N=50 → {val_50:.6f}, N=100 → {val_100:.6f}, "
        f"{diff_str} (threshold={threshold})"
    )


# ---------------------------------------------------------------------------
# Test Cases (Optimized with Batch Computation)
# ---------------------------------------------------------------------------

relative_threshold=0.15
@pytest.mark.parametrize("n_samples", [10, 20])
@pytest.mark.parametrize("metric_name,threshold,is_coverage", [
    ("crps", relative_threshold, False),
    ("log_score", relative_threshold, False),
    ("sharpness", relative_threshold, False),
    ("coverage_90", relative_threshold, True),
    ("coverage_95", relative_threshold, True),
    ("interval_score_90", relative_threshold, False),
    ("interval_score_95", relative_threshold, False),
    ("crls", relative_threshold, False),
    ("wcrps_left", relative_threshold, False),
    ("wcrps_right", relative_threshold, False),
    ("wcrps_center", relative_threshold, False),
    ("cde_loss", relative_threshold, False),
])
def test_metric_discretization_stability(n_samples, metric_name, threshold, is_coverage):
    """Parametrized test for all metric types with batch computation."""
    batch = get_batch_distributions(n_samples)
    
    # Compute metrics at both discretization levels
    metrics_50 = compute_scoring_rules(batch[f"x_50_g_{n_samples}"], batch[f"x_50_f_{n_samples}"].mean)
    metrics_100 = compute_scoring_rules(batch[f"x_100_g_{n_samples}"], batch[f"x_100_f_{n_samples}"].mean)
    
    val_50 = metrics_50[metric_name]
    val_100 = metrics_100[metric_name]
    
    # Print metric values for debugging
    print(f"\n{metric_name} (n_samples={n_samples}, threshold={threshold}): N=50 → {val_50:.6f}, N=100 → {val_100:.6f}")
    
    assert_metric_stability(val_50, val_100, metric_name, threshold, is_coverage)


@pytest.mark.parametrize("n_samples", [10, 20])
@pytest.mark.parametrize("beta", ENERGY_BETAS)
def test_energy_score_discretization_stability(n_samples, beta):
    """Parametrized test for energy scores with batch computation."""
    batch = get_batch_distributions(n_samples)
    
    metrics_50 = compute_scoring_rules(batch[f"x_50_g_{n_samples}"], batch[f"x_50_f_{n_samples}"].mean)
    metrics_100 = compute_scoring_rules(batch[f"x_100_g_{n_samples}"], batch[f"x_100_f_{n_samples}"].mean)
    
    key = f"energy_score_beta_{beta}"
    assert_metric_stability(
        metrics_50[key],
        metrics_100[key],
        key,
        threshold=0.10
    )


def test_all_metrics_discretization_stability_summary():
    """Comprehensive test: verify all metrics are reasonably stable when changing grid resolution.
    
    Uses batch computation for efficiency.
    """
    batch = get_batch_distributions(40)
    
    metrics_50 = compute_scoring_rules(batch["x_50_g_40"], batch["x_50_f_40"].mean)
    metrics_100 = compute_scoring_rules(batch["x_100_g_40"], batch["x_100_f_40"].mean)
    
    # Thresholds for different metric types
    thresholds = {
        "crps": 0.10,
        "log_score": 0.15,
        "sharpness": 0.05,
        "dispersion": 0.2,
        "crls": 0.10,
        "cde_loss": 0.10,
        "wcrps_left": 0.10,
        "wcrps_right": 0.10,
        "wcrps_center": 0.10,
    }
    # Add Cressie-Read thresholds
    for lam in CRESSIE_READ_LAMBDAS:
        thresholds[f"cressie_read_lambda_{lam}"] = 0.15
    # Coverage metrics use absolute difference
    coverage_thresholds = {
        "coverage_90": 0.10,
        "coverage_95": 0.10,
    }
    # Interval scores use relative difference
    interval_thresholds = {
        "interval_score_90": 0.10,
        "interval_score_95": 0.10,
    }
    
    failed_metrics = []
    
    # Check standard metrics (relative difference)
    for metric_name, threshold in thresholds.items():
        val_50 = metrics_50[metric_name]
        val_100 = metrics_100[metric_name]
        rel_diff = abs(val_100 - val_50) / (abs(val_50) + 1e-10)
        if rel_diff >= threshold:
            failed_metrics.append(
                f"{metric_name}: N=50→{val_50:.6f}, N=100→{val_100:.6f}, "
                f"rel_diff={rel_diff:.4f} (threshold={threshold})"
            )
    
    # Check coverage metrics (absolute difference)
    for metric_name, threshold in coverage_thresholds.items():
        val_50 = metrics_50[metric_name]
        val_100 = metrics_100[metric_name]
        abs_diff = abs(val_100 - val_50)
        if abs_diff >= threshold:
            failed_metrics.append(
                f"{metric_name}: N=50→{val_50:.6f}, N=100→{val_100:.6f}, "
                f"abs_diff={abs_diff:.4f} (threshold={threshold})"
            )
    
    # Check interval score metrics (relative difference)
    for metric_name, threshold in interval_thresholds.items():
        val_50 = metrics_50[metric_name]
        val_100 = metrics_100[metric_name]
        rel_diff = abs(val_100 - val_50) / (abs(val_50) + 1e-10)
        if rel_diff >= threshold:
            failed_metrics.append(
                f"{metric_name}: N=50→{val_50:.6f}, N=100→{val_100:.6f}, "
                f"rel_diff={rel_diff:.4f} (threshold={threshold})"
            )
    
    # Check energy scores (relative difference)
    for beta in ENERGY_BETAS:
        key = f"energy_score_beta_{beta}"
        val_50 = metrics_50[key]
        val_100 = metrics_100[key]
        rel_diff = abs(val_100 - val_50) / (abs(val_50) + 1e-10)
        threshold = 0.10
        if rel_diff >= threshold:
            failed_metrics.append(
                f"{key}: N=50→{val_50:.6f}, N=100→{val_100:.6f}, "
                f"rel_diff={rel_diff:.4f} (threshold={threshold})"
            )
    
    if failed_metrics:
        msg = "Discretization instability detected in the following metrics:\n"
        msg += "\n".join(f"  - {m}" for m in failed_metrics)
        pytest.fail(msg)


# ---------------------------------------------------------------------------
# Multi-level convergence tests (N = 25, 50, 100, 200)
# ---------------------------------------------------------------------------

# All scalar metrics returned by compute_scoring_rules (excluding energy_score_*)
_SCALAR_METRICS = [
    "crps",
    "log_score",
    "sharpness",
    "dispersion",
    "coverage_90",
    "coverage_95",
    "interval_score_90",
    "interval_score_95",
    "crls",
    "cde_loss",
    "wcrps_left",
    "wcrps_right",
    "wcrps_center",
]
_COVERAGE_METRICS = {"coverage_90", "coverage_95"}

_GRID_SIZES = [50, 100, 200]


def _compute_metrics_at_grid(n_pts, n_samples=30):
    """Return compute_scoring_rules dict for N-point grid.
    
    Parameters
    ----------
    n_pts : int
        Number of bins for discretization.
    n_samples : int
        Number of samples. Must be large enough to compute meaningful dispersion.
    
    Returns
    -------
    dict
        Metrics dictionary from compute_scoring_rules.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_grid = np.sort(np.asarray(np.linspace(_GRID_MIN, _GRID_MAX, n_pts + 1), dtype=np.float32))
    x_grid_torch = torch.as_tensor(x_grid, dtype=torch.float32, device=device)
    bin_mids = (x_grid[:-1] + x_grid[1:]) / 2.0
    bin_widths = np.diff(x_grid)
    bin_mids_torch = torch.as_tensor(bin_mids, dtype=torch.float32, device=device)
    bin_widths_torch = torch.as_tensor(bin_widths, dtype=torch.float32, device=device)
    
    def _create_varied_dist(base_mu, base_sigma):
        """Create a distribution with varied samples for meaningful dispersion."""
        rng = np.random.RandomState(hash((n_pts, base_mu, base_sigma)) % 2**31)
        mus = rng.normal(base_mu, base_sigma * _MU_STD_FACTOR, size=n_samples)
        sigmas = rng.uniform(base_sigma * _SIGMA_LOWER_FACTOR, base_sigma * _SIGMA_UPPER_FACTOR, size=n_samples)
        
        probas_array = []
        mean_array = []
        
        for sample_idx in range(n_samples):
            mu_t = torch.tensor(mus[sample_idx], dtype=torch.float32, device=device)
            sigma_t = torch.tensor(sigmas[sample_idx], dtype=torch.float32, device=device)
            pdf_vals = normal_pdf_torch(bin_mids_torch, mu_t, sigma_t)
            probas = pdf_vals * bin_widths_torch
            probas = probas / probas.sum()
            probas_array.append(probas.cpu().numpy())
            mean_array.append(mus[sample_idx])
        
        probas_array = np.array(probas_array, dtype=np.float32)
        mean_array = np.array(mean_array, dtype=np.float32)
        
        return DistributionPrediction(
            probas=probas_array,
            bin_edges=x_grid.astype(np.float32),
            bin_midpoints=bin_mids.astype(np.float32),
            mean=mean_array,
        )
    
    dist_g = _create_varied_dist(base_mu=_MU_G, base_sigma=_SIGMA_G)
    dist_f = _create_varied_dist(base_mu=_MU_F, base_sigma=_SIGMA_F)
    return compute_scoring_rules(dist_g, dist_f.mean)


@pytest.mark.parametrize("metric_name", _SCALAR_METRICS)
def test_metric_convergence_across_resolutions(metric_name):
    """Verify that each metric converges (stays within 15 % relative or 0.15 absolute)
    as grid resolution doubles from N=50 → N=100 → N=200.

    Convergence is assessed by checking that consecutive doublings all satisfy
    the threshold, i.e. the value does not jump between any two adjacent levels.
    """
    threshold_rel = 0.15
    threshold_abs = 0.15  # for coverage metrics

    values = {n: _compute_metrics_at_grid(n)[metric_name] for n in _GRID_SIZES}

    failures = []
    for n_lo, n_hi in zip(_GRID_SIZES[:-1], _GRID_SIZES[1:]):
        v_lo = values[n_lo]
        v_hi = values[n_hi]
        if metric_name in _COVERAGE_METRICS:
            diff = abs(v_hi - v_lo)
            if diff >= threshold_abs:
                failures.append(
                    f"N={n_lo}→N={n_hi}: {v_lo:.6f}→{v_hi:.6f}, "
                    f"abs_diff={diff:.4f} (threshold={threshold_abs})"
                )
        else:
            rel = abs(v_hi - v_lo) / (abs(v_lo) + 1e-10)
            if rel >= threshold_rel:
                failures.append(
                    f"N={n_lo}→N={n_hi}: {v_lo:.6f}→{v_hi:.6f}, "
                    f"rel_diff={rel:.4f} (threshold={threshold_rel})"
                )

    if failures:
        pytest.fail(
            f"{metric_name} convergence failures across resolutions "
            f"{_GRID_SIZES}:\n" + "\n".join(f"  - {f}" for f in failures)
        )

@pytest.mark.parametrize("n_samples", [10, 20])
@pytest.mark.parametrize("lam", CRESSIE_READ_LAMBDAS)
def test_cressie_read_score_discretization_stability(n_samples, lam):
    """Parametrized test for Cressie-Read scores with batch computation."""
    batch = get_batch_distributions(n_samples)
    
    metrics_50 = compute_scoring_rules(batch[f"x_50_g_{n_samples}"], batch[f"x_50_f_{n_samples}"].mean)
    metrics_100 = compute_scoring_rules(batch[f"x_100_g_{n_samples}"], batch[f"x_100_f_{n_samples}"].mean)
    
    key = f"cressie_read_lambda_{lam}"
    assert_metric_stability(
        metrics_50[key],
        metrics_100[key],
        key,
        threshold=0.15
    )


@pytest.mark.parametrize("lam", CRESSIE_READ_LAMBDAS)
def test_cressie_read_score_convergence_across_resolutions(lam):
    """Verify Cressie-Read score convergence as grid resolution doubles."""
    threshold_rel = 0.15
    key = f"cressie_read_lambda_{lam}"
    values = {n: _compute_metrics_at_grid(n)[key] for n in _GRID_SIZES}

    failures = []
    for n_lo, n_hi in zip(_GRID_SIZES[:-1], _GRID_SIZES[1:]):
        v_lo, v_hi = values[n_lo], values[n_hi]
        rel = abs(v_hi - v_lo) / (abs(v_lo) + 1e-10)
        if rel >= threshold_rel:
            failures.append(
                f"N={n_lo}→N={n_hi}: {v_lo:.6f}→{v_hi:.6f}, "
                f"rel_diff={rel:.4f} (threshold={threshold_rel})"
            )

    if failures:
        pytest.fail(
            f"{key} convergence failures across resolutions {_GRID_SIZES}:\n"
            + "\n".join(f"  - {f}" for f in failures)
        )

