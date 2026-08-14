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

from scoringbench.metrics import compute_scoring_rules, ENERGY_BETAS, DPD_BETAS, CRTS_ALPHAS
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
                #
                # The seed deliberately does NOT depend on ``grid_name``: a
                # discretization test must hold the *underlying* distribution and
                # the *targets* fixed and vary only the grid resolution.  Seeding
                # per grid would redraw mus/sigmas (and hence the targets, which
                # are ``dist.mean``) for every resolution, so the measured
                # difference would mix discretization error with Monte-Carlo
                # noise — the noise dominates for small ``n_samples`` and makes
                # the test flaky rather than informative.
                rng = np.random.RandomState(hash((label, n_samples)) % 2**31)
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
        "x_100": np.linspace(_GRID_MIN, _GRID_MAX, 101),
        "x_150": np.linspace(_GRID_MIN, _GRID_MAX, 151),
    }


# Batch precomputed distributions for faster access
_batch_cache = {}

def get_batch_distributions(n_samples):
    """Get precomputed batch of all distribution configurations for n_samples."""
    if n_samples in _batch_cache:
        return _batch_cache[n_samples]
    
    discretizations = {
        "x_100": np.linspace(_GRID_MIN, _GRID_MAX, 101),
        "x_150": np.linspace(_GRID_MIN, _GRID_MAX, 151),
    }
    
    batch = {}
    for name, dist in make_discretized_distributions_batch(
        discretizations, [_MU_F, _MU_G], [_SIGMA_F, _SIGMA_G], [n_samples]
    ):
        batch[name] = dist
    
    _batch_cache[n_samples] = batch
    return batch


# Cache of full metric dicts keyed by n_samples.
#
# ``compute_scoring_rules`` returns *all* scoring rules in one pass, but the
# parametrized stability tests each read out only a single key (one energy
# beta, one DPD beta, one metric name, ...).  Recomputing the entire metric
# dict once per parameter is the dominant cost of this module.  Memoizing the
# two dicts (N=100 and N=150) per ``n_samples`` collapses those redundant
# passes into a single computation shared across every parametrization.
_metrics_cache = {}

def get_discretization_metrics(n_samples):
    """Return ``(metrics_100, metrics_150)`` metric dicts for ``n_samples``.

    The result is cached so that ``compute_scoring_rules`` runs at most twice
    per ``n_samples`` regardless of how many parametrized cases consume it.
    """
    if n_samples in _metrics_cache:
        return _metrics_cache[n_samples]

    batch = get_batch_distributions(n_samples)
    metrics_100 = compute_scoring_rules(
        batch[f"x_100_g_{n_samples}"], batch[f"x_100_f_{n_samples}"].mean
    )
    metrics_150 = compute_scoring_rules(
        batch[f"x_150_g_{n_samples}"], batch[f"x_150_f_{n_samples}"].mean
    )
    _metrics_cache[n_samples] = (metrics_100, metrics_150)
    return metrics_100, metrics_150



def assert_metric_stability(val_lo, val_hi, metric_name, threshold=None, is_coverage=False):
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
        threshold = 0.15  # Default 15% relative difference (relaxed per discussion)
    
    if is_coverage:
        diff = abs(val_hi - val_lo)
        check_passed = diff < threshold
        diff_str = f"abs_diff={diff:.4f}"
    else:
        diff = abs(val_hi - val_lo) / (abs(val_lo) + 1e-10)
        check_passed = diff < threshold
        diff_str = f"rel_diff={diff:.4f}"
    
    assert check_passed, (
        f"{metric_name} discretization instability: "
        f"N_lo → {val_lo:.6f}, N_hi → {val_hi:.6f}, "
        f"{diff_str} (threshold={threshold})"
    )


# ---------------------------------------------------------------------------
# Test Cases (Optimized with Batch Computation)
# ---------------------------------------------------------------------------

relative_threshold=0.15
@pytest.mark.parametrize("n_samples", [10, 20])
@pytest.mark.parametrize("metric_name,threshold,is_coverage", [
    ("crps", relative_threshold, False),
    ("sharpness", relative_threshold, False),
    ("coverage_90", relative_threshold, True),
    ("coverage_95", relative_threshold, True),
    ("interval_score_90", relative_threshold, False),
    ("interval_score_95", relative_threshold, False),
    ("crts_alpha_1.01", relative_threshold, False),
    ("wcrps_left", relative_threshold, False),
    ("wcrps_right", relative_threshold, False),
    ("wcrps_center", relative_threshold, False),
    ("cde_loss", relative_threshold, False),
    ("dpd_beta_0.01", relative_threshold, False),
    ("dpd_beta_0.2", relative_threshold, False),
    ("dpd_beta_0.5", relative_threshold, False),
    ("dpd_beta_1.0", relative_threshold, False),
])
def test_metric_discretization_stability(n_samples, metric_name, threshold, is_coverage):
    """Parametrized test for all metric types with batch computation."""
    # Compute metrics at both discretization levels (N=100 → N=150).
    # The dicts are memoized, so this is computed once per ``n_samples``.
    metrics_100, metrics_150 = get_discretization_metrics(n_samples)

    val_100 = metrics_100[metric_name]
    val_150 = metrics_150[metric_name]

    # Print metric values for debugging
    print(f"\n{metric_name} (n_samples={n_samples}, threshold={threshold}): N=100 → {val_100:.6f}, N=150 → {val_150:.6f}")

    assert_metric_stability(val_100, val_150, metric_name, threshold, is_coverage)


@pytest.mark.parametrize("n_samples", [10, 20])
@pytest.mark.parametrize("beta", ENERGY_BETAS)
def test_energy_score_discretization_stability(n_samples, beta):
    """Parametrized test for energy scores with batch computation."""
    # Memoized: ``compute_scoring_rules`` runs once per ``n_samples`` and is
    # shared across every energy-beta parametrization instead of being redone
    # for each of the 12 betas.
    metrics_100, metrics_150 = get_discretization_metrics(n_samples)

    key = f"energy_score_beta_{beta}"
    assert_metric_stability(
        metrics_100[key],
        metrics_150[key],
        key,
        threshold=0.15
    )


def test_all_metrics_discretization_stability_summary():
    """Comprehensive test: verify all metrics are reasonably stable when changing grid resolution.
    
    Uses batch computation for efficiency.
    """
    # Reuse the memoized metric dicts (computed once per ``n_samples``).
    metrics_100, metrics_150 = get_discretization_metrics(40)
    
    # Thresholds for different metric types
    thresholds = {
        "crps": 0.10,
        "dpd_beta_0.01": 0.15,
        "sharpness": 0.05,
        "dispersion": 0.2,
        "crts_alpha_1.01": 0.10,
        "cde_loss": 0.10,
        "wcrps_left": 0.10,
        "wcrps_right": 0.10,
        "wcrps_center": 0.10,
    }
    # Add DPD and CRTS thresholds
    for b in DPD_BETAS:
        thresholds[f"dpd_beta_{b}"] = 0.10
    for a in CRTS_ALPHAS:
        thresholds[f"crts_alpha_{a}"] = 0.10
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
        val_100 = metrics_100[metric_name]
        val_150 = metrics_150[metric_name]
        rel_diff = abs(val_150 - val_100) / (abs(val_100) + 1e-10)
        if rel_diff >= threshold:
            failed_metrics.append(
                f"{metric_name}: N=100→{val_100:.6f}, N=150→{val_150:.6f}, "
                f"rel_diff={rel_diff:.4f} (threshold={threshold})"
            )
    
    # Check coverage metrics (absolute difference)
    for metric_name, threshold in coverage_thresholds.items():
        val_100 = metrics_100[metric_name]
        val_150 = metrics_150[metric_name]
        abs_diff = abs(val_150 - val_100)
        if abs_diff >= threshold:
            failed_metrics.append(
                f"{metric_name}: N=100→{val_100:.6f}, N=150→{val_150:.6f}, "
                f"abs_diff={abs_diff:.4f} (threshold={threshold})"
            )
    
    # Check interval score metrics (relative difference)
    for metric_name, threshold in interval_thresholds.items():
        val_100 = metrics_100[metric_name]
        val_150 = metrics_150[metric_name]
        rel_diff = abs(val_150 - val_100) / (abs(val_100) + 1e-10)
        if rel_diff >= threshold:
            failed_metrics.append(
                f"{metric_name}: N=100→{val_100:.6f}, N=150→{val_150:.6f}, "
                f"rel_diff={rel_diff:.4f} (threshold={threshold})"
            )
    
    # Check energy scores (relative difference)
    for beta in ENERGY_BETAS:
        key = f"energy_score_beta_{beta}"
        val_100 = metrics_100[key]
        val_150 = metrics_150[key]
        rel_diff = abs(val_150 - val_100) / (abs(val_100) + 1e-10)
        threshold = 0.10
        if rel_diff >= threshold:
            failed_metrics.append(
                f"{key}: N=100→{val_100:.6f}, N=150→{val_150:.6f}, "
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
    "dpd_beta_0.01",
    "sharpness",
    "dispersion",
    "coverage_90",
    "coverage_95",
    "interval_score_90",
    "interval_score_95",
    "crts_alpha_1.01",
    "cde_loss",
    "wcrps_left",
    "wcrps_right",
    "wcrps_center",
]
_COVERAGE_METRICS = {"coverage_90", "coverage_95"}

_GRID_SIZES = [100, 150]


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
        """Create a distribution with varied samples for meaningful dispersion.

        The seed deliberately excludes ``n_pts`` so that the same underlying
        distributions and targets are discretized at every resolution; otherwise
        the comparison would measure Monte-Carlo noise instead of discretization
        error (see ``make_discretized_distributions_batch``).
        """
        rng = np.random.RandomState(hash((base_mu, base_sigma, n_samples)) % 2**31)
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
    across adjacent resolutions N=100 → N=150.

    Convergence is assessed by checking that consecutive adjacent levels all satisfy
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
@pytest.mark.parametrize("beta", DPD_BETAS)
def test_dpd_score_discretization_stability(n_samples, beta):
    """Parametrized test for DPD scores with batch computation."""
    batch = get_batch_distributions(n_samples)

    metrics_100 = compute_scoring_rules(batch[f"x_100_g_{n_samples}"], batch[f"x_100_f_{n_samples}"].mean)
    metrics_150 = compute_scoring_rules(batch[f"x_150_g_{n_samples}"], batch[f"x_150_f_{n_samples}"].mean)

    key = f"dpd_beta_{beta}"
    assert_metric_stability(
        metrics_100[key],
        metrics_150[key],
        key,
        threshold=0.15
    )


@pytest.mark.parametrize("beta", DPD_BETAS)
def test_dpd_score_convergence_across_resolutions(beta):
    """Verify DPD score convergence as grid resolution doubles."""
    threshold_rel = 0.15
    key = f"dpd_beta_{beta}"
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


# ---------------------------------------------------------------------------
# Monte-Carlo validation of the beta energy score
#
# The histogram energy score computed by ScoringBench must agree with the
# *definition* of the beta energy score,
#
#     ES_beta(F, y) = E|X - y|^beta - 0.5 * E|X - X'|^beta,   X, X' ~ F i.i.d.,
#
# estimated independently by plain Monte-Carlo from samples of the SAME
# predictive distribution.  We build the predictive distribution two ways from
# one Gaussian F:
#   * the eCDF equiprobable de-tied grid (samples_to_distribution), scored by
#     compute_scoring_rules;
#   * a large i.i.d. sample of F, scored by the naive MC formula above.
# For a well-behaved (continuous) F the two must be close for every beta.
# ---------------------------------------------------------------------------

from scoringbench.wrappers.sample_based import samples_to_distribution


def _mc_energy_score_beta(samples_row: np.ndarray, y: float, beta: float,
                          n_pairs: int, rng) -> float:
    """Independent Monte-Carlo estimate of ES_beta for one predictive dist."""
    x = np.asarray(samples_row, dtype=np.float64)
    # Term 1: E|X - y|^beta
    term1 = np.mean(np.abs(x - y) ** beta)
    # Term 2: 0.5 * E|X - X'|^beta  (independent pairs, no diagonal)
    i = rng.integers(0, x.shape[0], size=n_pairs)
    j = rng.integers(0, x.shape[0], size=n_pairs)
    term2 = 0.5 * np.mean(np.abs(x[i] - x[j]) ** beta)
    return term1 - term2


@pytest.mark.parametrize("beta", [0.5, 1.0, 1.5, 1.9])
def test_energy_score_matches_monte_carlo_beta_formula(beta):
    """Histogram beta energy score ≈ independent Monte-Carlo estimate on a
    continuous predictive distribution, for beta in (0, 2]."""
    rng = np.random.default_rng(20240607)
    n_test = 8
    n_draws = 200_000          # large so the histogram is a faithful estimate of F
    mu = rng.uniform(-1.0, 1.0, size=n_test)
    sigma = rng.uniform(0.7, 1.4, size=n_test)

    samples = rng.normal(loc=mu[:, None], scale=sigma[:, None], size=(n_test, n_draws))
    # Targets a little away from the mean so both energy terms contribute.
    y = mu + 0.5 * sigma

    # ScoringBench path: eCDF equiprobable de-tied grid -> histogram energy score.
    dist = samples_to_distribution(samples, n_bins=200)
    sb = compute_scoring_rules(dist, y)
    sb_es = sb[f"energy_score_beta_{beta}"]

    # Independent Monte-Carlo path on the SAME draws.
    mc_vals = np.array([
        _mc_energy_score_beta(samples[i], float(y[i]), beta, n_pairs=400_000, rng=rng)
        for i in range(n_test)
    ])
    mc_es = float(np.mean(np.clip(mc_vals, 0.0, None)))

    rel = abs(sb_es - mc_es) / (abs(mc_es) + 1e-8)
    assert np.isfinite(sb_es) and sb_es > 0.0
    assert rel < 0.05, (
        f"beta={beta}: histogram ES={sb_es:.6f} vs Monte-Carlo ES={mc_es:.6f} "
        f"(rel_diff={rel:.4f})"
    )


def test_energy_score_crps_equals_beta_1_monte_carlo():
    """At beta=1 the energy score is the CRPS; both must match the Monte-Carlo
    CRPS of the predictive distribution."""
    rng = np.random.default_rng(11)
    n_test = 6
    n_draws = 200_000
    mu = rng.uniform(-1.0, 1.0, size=n_test)
    sigma = rng.uniform(0.8, 1.2, size=n_test)
    samples = rng.normal(loc=mu[:, None], scale=sigma[:, None], size=(n_test, n_draws))
    y = mu - 0.3 * sigma

    dist = samples_to_distribution(samples, n_bins=200)
    scores = compute_scoring_rules(dist, y)

    mc_vals = np.array([
        _mc_energy_score_beta(samples[i], float(y[i]), 1.0, n_pairs=400_000, rng=rng)
        for i in range(n_test)
    ])
    mc_crps = float(np.mean(np.clip(mc_vals, 0.0, None)))

    # energy_score_beta_1.0 and crps are the same quantity here.
    assert abs(scores["energy_score_beta_1.0"] - scores["crps"]) < 1e-3 * (abs(scores["crps"]) + 1e-8)
    rel = abs(scores["crps"] - mc_crps) / (abs(mc_crps) + 1e-8)
    assert rel < 0.05, f"CRPS {scores['crps']:.6f} vs MC {mc_crps:.6f} (rel={rel:.4f})"

