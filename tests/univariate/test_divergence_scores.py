"""Tests for divergence-based scoring metrics: Density Power Divergence (DPD).

This module validates the DPD scoring rule implementation and basic consistency
properties: finiteness, ordering (perfect vs imperfect), and the β→0 limit
which recovers the negative log score up to a constant.
"""

import math
import numpy as np
import pytest
import torch

# Force CPU
torch.cuda.is_available = lambda: False

from scoringbench.univariate.metrics import (
    compute_scoring_rules,
    compute_pseudospherical_scores,
    DPD_BETAS,
    CRTS_ALPHAS,
    PSEUDOS_ALPHAS,
)  # noqa: E402
from scoringbench.univariate.wrappers import DistributionPrediction


# ============================================================================
# Numerical Reference Implementations
# ============================================================================

def _reference_unified_density(probas, bin_edges):
    """Independent NumPy re-derivation of the density the metrics module scores.

    The production density is piecewise constant on the bins and is the forward
    difference of the CDF::

        f_k = [F(x_{k+1}) - F(x_k)] / w_k = p_k / w_k

    with ``w_k`` the bin's own width.  ``w_k = 0`` would make that 0/0, but the
    grid never reaches the metrics in that state: density rules score on
    ``DistributionPrediction.resampled``, the grow-only grid built by
    ``resample_cdf_nodes_to_support_outer_hull_y_train_set_y_instance_prediction_grid``, so every width here is
    positive and the reference needs no special case.  Callers therefore pass
    ``dist.resampled.probas`` / ``dist.resampled.bin_edges`` here, not the raw
    native PMF grid.  The result is renormalised so that
    ``∑_k f_k w_k = 1``, which for an exact PMF is already true.

    Both terms of a two-term rule must be functionals of this *same* ``f`` for
    the rule to be proper, so the reference builds ``f`` once and returns it
    together with the effective widths that integrate it.

    Returns
    -------
    f : np.ndarray
        Per-bin density values, shape ``(n_bins,)``.
    w_eff : np.ndarray
        Per-bin effective widths, shape ``(n_bins,)``, with ``∑ f w_eff = 1``.
    """
    probas = np.asarray(probas, dtype=float)
    edges = np.asarray(bin_edges, dtype=float)
    widths = np.diff(edges)
    eps = 100 * np.finfo(np.float64).eps

    f = probas / np.maximum(widths, eps)
    w_eff = widths

    return f / max((f * w_eff).sum(), eps), w_eff


def reference_dpd_score(probas, bin_edges, y, beta):
    """Reference DPD score built from the unified bin density.

        S_β = ∫ f(t)^{1+β} dt - (1 + 1/β) f(y)^β    (β > 0)
        S_0 = -log f(y)                              (β = 0 limit)

    Because ``f`` is piecewise constant, ``∫ f^{1+β}`` has the closed form
    ``∑_k f_k^{1+β} w_k^eff`` — no quadrature is needed — and ``f(y)`` is the
    value of that *same* ``f`` on the bin containing ``y``, which is the
    propriety-preserving pairing.
    """
    f, w_eff = _reference_unified_density(probas, bin_edges)
    edges = np.asarray(bin_edges, dtype=float)
    y_bin = int(np.clip(np.searchsorted(edges[1:], y), 0, len(f) - 1))

    eps = 1e-10
    g_y = max(float(f[y_bin]), eps)

    if abs(beta) < 1e-12:
        return -math.log(g_y)

    integral = float((f ** (1.0 + beta) * w_eff).sum())
    point_term = (1.0 + 1.0 / beta) * (g_y ** beta)
    return integral - point_term


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
        train_range=(float(np.asarray(bin_edges).min()), float(np.asarray(bin_edges).max())),
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
        train_range=(float(np.asarray(bin_edges).min()), float(np.asarray(bin_edges).max())),
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
        train_range=(float(np.asarray(bin_edges).min()), float(np.asarray(bin_edges).max())),
    )


# ============================================================================
# Sanity Check Tests: Finite Values, No NaNs/Infs
# ============================================================================

@pytest.mark.parametrize("beta", DPD_BETAS)
def test_dpd_score_is_finite(beta):
    """Test that DPD scores are finite and well-defined."""
    dist = get_simple_distribution()
    y_true = np.array([1.5], dtype=np.float32)

    metrics = compute_scoring_rules(dist, y_true)
    key = f"dpd_beta_{beta}"

    assert key in metrics, f"Missing key {key}"
    assert isinstance(metrics[key], float)
    assert not np.isnan(metrics[key]), f"{key} is NaN"
    assert not np.isinf(metrics[key]), f"{key} is Inf"


# ============================================================================
# Numerical Validation Tests: Exact Values
# ============================================================================

class TestDPDExactValues:
    """Numerical tests for DPD scores and expected ordering."""

    @pytest.mark.parametrize("beta", DPD_BETAS)
    def test_perfect_vs_imperfect_ordering(self, beta):
        """Perfect prediction should score better (lower) than imperfect prediction.

        ``y`` is placed strictly inside a bin rather than on an edge: bins are
        read as half-open ``(left, right]``, so a target sitting exactly on
        ``edges[15]`` belongs to bin 14, and a spike put in bin 15 would then be
        scored as a *miss* rather than as a perfect prediction.
        """
        edges = np.linspace(0.0, 3.0, 31)
        mids = 0.5 * (edges[:-1] + edges[1:])
        y = np.array([1.55], dtype=np.float32)

        probas_perfect = np.zeros((1, 30))
        probas_perfect[0, 15] = 1.0                      # the bin containing y
        probas_imperfect = np.full((1, 30), 1.0 / 30.0)

        def _dist(probas):
            return DistributionPrediction(
                probas=probas, bin_edges=edges, bin_midpoints=mids,
                mean=(probas * mids).sum(axis=1),
                train_range=(float(np.asarray(edges).min()), float(np.asarray(edges).max())),
            )

        key = f"dpd_beta_{beta}"
        score_perfect = compute_scoring_rules(_dist(probas_perfect), y)[key]
        score_imperfect = compute_scoring_rules(_dist(probas_imperfect), y)[key]

        assert score_perfect < score_imperfect, (
            f"Perfect prediction ({score_perfect:.6f}) should score lower (better) "
            f"than imperfect ({score_imperfect:.6f}) for {key}"
        )

    def test_various_betas_differ(self):
        """Different β values should produce different numeric results for imperfect predictions."""
        dist = get_imperfect_distribution()
        y_true = np.array([1.5], dtype=np.float32)
        metrics = compute_scoring_rules(dist, y_true)

        scores = [metrics[f"dpd_beta_{b}"] for b in DPD_BETAS]
        assert len(set(scores)) > 1, "Different β values should produce different DPD scores"


class TestComparisonBetweenMetrics:
    """Tests comparing DPD behavior at perfect vs imperfect predictions."""
    
    def test_crts_keys_present(self):
        """All crts_alpha_* keys should be present in results."""
        dist = get_imperfect_distribution()
        y_true = np.array([1.5], dtype=np.float32)

        metrics = compute_scoring_rules(dist, y_true)

        for alpha in CRTS_ALPHAS:
            key = f"crts_alpha_{alpha}"
            assert key in metrics, f"Missing {key}"
            assert isinstance(metrics[key], float), f"{key} should be float"


# ============================================================================
# Integration and Consistency Tests
# ============================================================================

def test_all_new_metrics_present_in_results():
    """Verify all DPD and CRTS metrics are computed and present in results."""
    dist = get_simple_distribution()
    y_true = np.array([1.5], dtype=np.float32)

    metrics = compute_scoring_rules(dist, y_true)

    for beta in DPD_BETAS:
        assert f"dpd_beta_{beta}" in metrics, f"Missing dpd_beta_{beta}"
    for alpha in CRTS_ALPHAS:
        assert f"crts_alpha_{alpha}" in metrics, f"Missing crts_alpha_{alpha}"


def test_different_betas_produce_different_scores():
    """Different β values should produce different DPD results."""
    dist = get_imperfect_distribution()
    y_true = np.array([1.5], dtype=np.float32)

    metrics = compute_scoring_rules(dist, y_true)

    scores = [metrics[f"dpd_beta_{b}"] for b in DPD_BETAS]
    unique_scores = set(scores)

    assert len(unique_scores) > 1, "Different β values should produce different DPD scores"


# ============================================================================
class TestLimitBehavior:
    """Tests verifying limit behavior as parameters approach special values."""
    
    def test_reference_dpd_matches_implementation_for_simple_case(self):
        """Cross-check reference DPD formula against compute_scoring_rules for a simple case."""
        dist = get_simple_distribution()
        y_true = np.array([1.5], dtype=np.float32)

        metrics = compute_scoring_rules(dist, y_true)

        # Density rules score on the shared common grid, not the raw native
        # grid, so the reference must read the same resampled histogram.
        rg = dist.resampled
        probas = np.asarray(rg.probas)[0]
        y = float(y_true[0])

        for beta in DPD_BETAS:
            key = f"dpd_beta_{beta}"
            ref = reference_dpd_score(probas, rg.bin_edges, y, beta)
            # Both sides evaluate the same closed form, so only float summation
            # order separates them; the tolerance is deliberately generous.
            assert math.isclose(metrics[key], ref, rel_tol=1e-4, abs_tol=1e-6), (
                f"DPD implementation {key}={metrics[key]:.8f} differs from reference {ref:.8f}"
            )

    def test_reference_density_integrates_to_one(self):
        """The reference density must be a density (∫ f = 1).

        Guards the reference itself: if the renormalisation or the effective
        widths were wrong, every ∫ f^{1+β} term would be biased and the
        cross-check above would compare two equally wrong numbers.
        """
        dist = get_imperfect_distribution()
        f, w_eff = _reference_unified_density(dist.probas[0], dist.bin_edges)
        mass = float((f * w_eff).sum())
        assert math.isclose(mass, 1.0, rel_tol=1e-12), f"∫ f = {mass:.8f}, expected 1"


class TestEdgeCasesAndRobustness:
    """Tests for edge cases and numerical robustness."""
    
    def test_dpd_very_small_density(self):
        """Test DPD with very small but nonzero density — should be finite and large."""
        bin_edges = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        bin_mids = np.array([0.5, 1.5], dtype=np.float32)
        # Very small probability on first bin
        probas = np.array([[0.001, 0.999]], dtype=np.float32)

        dist = DistributionPrediction(
            probas=probas,
            bin_edges=bin_edges,
            bin_midpoints=bin_mids,
            mean=np.array([1.5], dtype=np.float64),
            train_range=(float(np.asarray(bin_edges).min()), float(np.asarray(bin_edges).max())),
        )
        y_true = np.array([0.5], dtype=np.float32)
        metrics = compute_scoring_rules(dist, y_true)

        for beta in DPD_BETAS:
            score = metrics[f"dpd_beta_{beta}"]
            assert not np.isnan(score), f"Score is NaN for beta={beta}"
            assert not np.isinf(score), f"Score is Inf for beta={beta}"
            # For β=0 (log-score) very low density should yield a large positive value.
            if abs(beta) < 1e-12:
                assert score > 0.1, f"Very low confidence should give large log-score, got {score}"
            # For other β values DPD can be negative depending on the integral term,
            # so we only require finiteness (already checked above).

    def test_cde_loss_equals_dpd_beta1(self):
        """CDE loss should match DPD with β=1 (∫ f^2 dt - 2 f(y))."""
        dist = get_imperfect_distribution()
        y_true = np.array([1.5], dtype=np.float32)

        metrics = compute_scoring_rules(dist, y_true)

        cde = metrics.get("cde_loss")
        dpd1 = metrics.get("dpd_beta_1.0")

        assert cde is not None and dpd1 is not None
        assert math.isclose(cde, dpd1, rel_tol=1e-6, abs_tol=1e-8), (
            f"cde_loss ({cde}) should equal dpd_beta_1.0 ({dpd1})"
        )


# ============================================================================
# CRTS propriety guards
#
# CRTS integrates the binary α-Tsallis loss of the threshold indicator against
# the predicted CDF.  At each bin the per-bin Bernoulli(p) is scored against the
# observed indicator with the *full* α-Tsallis polynomial, whose integral term
#
#     [p^α + (1-p)^α] / α
#
# is what makes the rule strictly proper.  A naive log-score → power-score text
# substitution that drops this normalisation term yields an IMPROPER rule whose
# expected score is no longer minimised at the true forecast.  The two tests
# below are the strongest regression guards against that specific mistake:
#
#   1.  α = 2 collapses CRTS to (a discretisation of) the CRPS.
#   2.  The expected CRTS is strictly minimised at the data-generating
#       forecast — i.e. the rule is empirically proper.
# ============================================================================

def _gaussian_histogram_distribution(mu, sigma, edges, n_rows):
    """Build a DistributionPrediction whose PMF is N(mu, sigma) on ``edges``."""
    from scipy import stats

    edges = np.asarray(edges, dtype=np.float64)
    mids = (edges[:-1] + edges[1:]) / 2.0
    cdf = stats.norm(mu, sigma).cdf(edges)
    p = np.diff(cdf)[None, :].repeat(n_rows, axis=0)
    return DistributionPrediction(
        probas=p.astype(np.float32),
        bin_edges=edges.astype(np.float32),
        bin_midpoints=mids.astype(np.float32),
        mean=(p @ mids),
        train_range=(float(np.asarray(edges).min()), float(np.asarray(edges).max())),
    )


def test_crts_alpha_2_matches_crps():
    """α = 2 CRTS collapses to the CRPS (Brier-divergence identity).

    The binary α-Tsallis loss at α = 2 is the Brier/quadratic score, so CRTS
    integrates ∫ (F - 1{y ≤ t})^2 dt, which is exactly the CRPS.  On a fine grid
    the two should agree up to discretisation error.  A missing normalisation
    term would break this identity.
    """
    rng = np.random.default_rng(0)
    y = rng.normal(0.0, 1.0, size=800).astype(np.float32)
    edges = np.linspace(-7.0, 7.0, 201, dtype=np.float64)
    dist = _gaussian_histogram_distribution(0.0, 1.0, edges, len(y))

    metrics = compute_scoring_rules(dist, y)
    crps = metrics["crps"]
    crts2 = metrics["crts_alpha_2.0"]

    assert math.isclose(crps, crts2, rel_tol=2e-2, abs_tol=2e-3), (
        f"crts_alpha_2.0 ({crts2:.6f}) should match crps ({crps:.6f}) "
        f"up to discretisation error"
    )


@pytest.mark.parametrize("alpha", [1.2, 1.5, 2.0])
def test_crts_is_empirically_proper(alpha):
    """The expected CRTS is strictly minimised at the true forecast.

    Data are drawn from N(0, 1).  Among a set of candidate forecasts, only the
    true one (μ=0, σ=1) should attain the lowest mean CRTS.  An improper rule
    (e.g. one missing the p^α + (1-p)^α normalisation term) would let a
    mis-specified forecast win — this test would then fail.

    The 300-sample / 61-bin grid over [-6, 6] is deliberately small: the
    propriety *margin* (runner-up minus true) converges by ~300 samples and is
    within <1% of the value obtained with 1500 samples / 161 bins, so a coarser
    grid keeps a comfortable margin while cutting the (quadratic-in-bins) energy
    score cost that dominates ``compute_scoring_rules``.
    """
    rng = np.random.default_rng(0)
    y = rng.normal(0.0, 1.0, size=300).astype(np.float32)
    edges = np.linspace(-6.0, 6.0, 61, dtype=np.float64)
    key = f"crts_alpha_{alpha}"

    candidates = {
        "true(0,1)": (0.0, 1.0),
        "shift(0.5,1)": (0.5, 1.0),
        "narrow(0,0.6)": (0.0, 0.6),
        "wide(0,1.6)": (0.0, 1.6),
    }

    scores = {}
    for name, (mu, sigma) in candidates.items():
        dist = _gaussian_histogram_distribution(mu, sigma, edges, len(y))
        scores[name] = compute_scoring_rules(dist, y)[key]

    best = min(scores, key=scores.get)
    assert best == "true(0,1)", (
        f"CRTS (α={alpha}) is not proper: expected the true forecast to win, "
        f"got {best}. Scores: "
        + ", ".join(f"{n}={v:.4f}" for n, v in scores.items())
    )
    # Also require a clear margin so a near-tie doesn't mask a subtle bias.
    runner_up = min(v for n, v in scores.items() if n != "true(0,1)")
    assert scores["true(0,1)"] < runner_up, (
        f"True forecast should strictly beat the runner-up (α={alpha}): "
        f"true={scores['true(0,1)']:.4f} vs runner-up={runner_up:.4f}"
    )


# ============================================================================
# Pseudospherical Score (Good, 1971)
# ============================================================================
#
# The pseudospherical score is the *ratio-form* member of the density-power
# family (DPD is the difference form).  We use the Gneiting & Raftery (2007,
# eq. 12) affine normalisation.  For order α > 1,
#
#     PseudoS_α(f, y) = 1/(α-1) · [ (f(y)/‖f‖_α)^{α-1} − 1 ],
#         ‖f‖_α = (∫ f^α dt)^{1/α},
#
# is positively oriented (larger = better), scale-invariant, and reduces (up to
# the affine map) to the spherical score at α = 2.  ScoringBench reports it
# negated so lower = better, under keys ``pseudospherical_alpha_{α}``.  The
# ``1/(α-1)`` factor and ``−1`` offset are order-preserving, so propriety and
# scale-invariance are unchanged.  These tests pin the properties that
# are *specific* to the ratio form; propriety and support-insensitivity on the
# production path are additionally covered by the support-sensitivity suite,
# which auto-classifies these keys as proper.


def _reference_pseudospherical(probas, bin_edges, y, alpha):
    """Independent NumPy re-derivation of the (negated) pseudospherical score.

        PseudoS_α = 1/(α-1) · [ f(y)^{α-1} / (∫ f^α dt)^{(α-1)/α} − 1 ]

    (Gneiting & Raftery 2007, eq. 12) built from the *same* unified bin density
    used by every two-term rule, so the point value ``f(y)`` and the norm
    ``∫ f^α`` are functionals of one ``f`` (the pairing that makes the rule
    proper).  Returned negated to match the module's lower-is-better convention.
    """
    f, w_eff = _reference_unified_density(probas, bin_edges)
    f = np.asarray(f, dtype=float).reshape(-1)
    edges = np.asarray(bin_edges, dtype=float)
    y_bin = int(np.clip(np.searchsorted(edges[1:], y), 0, len(f) - 1))

    eps = 1e-10
    g_y = max(float(f[y_bin]), 0.0)
    norm_alpha = max(float((f ** alpha * w_eff).sum()), eps)
    denom = norm_alpha ** ((alpha - 1.0) / alpha)
    ratio = g_y ** (alpha - 1.0) / denom
    return -((ratio - 1.0) / (alpha - 1.0))


@pytest.mark.parametrize("alpha", PSEUDOS_ALPHAS)
def test_pseudospherical_matches_reference(alpha):
    """The module's pseudospherical score matches an independent NumPy re-derivation.

    Guards the closed-form ``∫ f^α = ∑_k f_k^α w_k^eff`` and the exponent
    ``(α-1)/α`` on the norm — an off-by-one there would silently change the
    ranking without producing NaNs.
    """
    dist = get_imperfect_distribution()
    y = np.array([1.5], dtype=np.float32)

    key = f"pseudospherical_alpha_{alpha}"
    got = compute_scoring_rules(dist, y)[key]
    # Density rules score on the shared common grid; build the reference there.
    rg = dist.resampled
    ref = _reference_pseudospherical(rg.probas, rg.bin_edges, float(y[0]), alpha)

    assert math.isclose(got, ref, rel_tol=1e-6, abs_tol=1e-9), (
        f"pseudospherical (α={alpha}) = {got:.9f} does not match reference "
        f"{ref:.9f}"
    )


def test_pseudospherical_spherical_identity():
    """At α = 2 the score is the affine image of the spherical score f(y)/‖f‖₂.

    This is the defining special case: with the Gneiting–Raftery normalisation
    (1/(α-1)=1 and offset −1 at α=2) the negated reported value must equal
    ``f(y) / sqrt(∫ f² dt) − 1``.
    """
    dist = get_imperfect_distribution()
    y = np.array([1.5], dtype=np.float32)

    reported = compute_scoring_rules(dist, y)["pseudospherical_alpha_2.0"]

    # Density rules score on the shared common grid; build f there.
    rg = dist.resampled
    f, w_eff = _reference_unified_density(rg.probas, rg.bin_edges)
    f = np.asarray(f, dtype=float).reshape(-1)
    edges = np.asarray(rg.bin_edges, dtype=float)
    y_bin = int(np.clip(np.searchsorted(edges[1:], float(y[0])), 0, len(f) - 1))
    spherical = float(f[y_bin]) / math.sqrt(float((f ** 2 * w_eff).sum()))

    # reported is the negated (loss-form) score; at α=2 that is (spherical − 1).
    assert math.isclose(-reported, spherical - 1.0, rel_tol=1e-6, abs_tol=1e-9), (
        f"α=2 pseudospherical should equal spherical score minus one "
        f"{spherical - 1.0:.9f}, got {-reported:.9f}"
    )


@pytest.mark.parametrize("alpha", PSEUDOS_ALPHAS)
def test_pseudospherical_is_scale_invariant(alpha):
    """Scaling the forecast density f → c·f leaves the score unchanged.

    Scale invariance is the hallmark of the ratio form: the c^{α-1} in the
    numerator cancels the c^{α-1} the norm contributes.  We emulate an unnormalised
    forecast by scaling the raw probabilities; the metrics module renormalises
    internally, so an implementation that failed to cancel the scale would move
    the score.
    """
    dist = get_imperfect_distribution()
    y = np.array([1.5], dtype=np.float32)
    key = f"pseudospherical_alpha_{alpha}"

    base = compute_scoring_rules(dist, y)[key]

    scaled = DistributionPrediction(
        probas=(np.asarray(dist.probas, dtype=np.float32) * 5.0),
        bin_edges=dist.bin_edges,
        bin_midpoints=dist.bin_midpoints,
        mean=dist.mean,
        train_range=(float(np.asarray(dist.bin_edges).min()), float(np.asarray(dist.bin_edges).max())),
    )
    scaled_score = compute_scoring_rules(scaled, y)[key]

    assert math.isclose(base, scaled_score, rel_tol=1e-6, abs_tol=1e-9), (
        f"pseudospherical (α={alpha}) is not scale-invariant: "
        f"{base:.9f} vs {scaled_score:.9f}"
    )


def test_pseudospherical_is_empirically_proper():
    """The expected pseudospherical score is minimised at the true forecast.

    Data ~ N(0, 1).  Only the true (μ=0, σ=1) forecast should attain the lowest
    mean (negated) pseudospherical score among the candidates, and this must
    hold for *every* reported order α simultaneously.

    The candidate distributions and the targets do not depend on α, and a single
    ``compute_scoring_rules`` call already returns every ``pseudospherical_alpha_*``
    key at once.  So the full pipeline is run just **once per candidate** (not
    once per candidate *and* order) and every reported α is then checked against
    the same cached scores — this is the property-preserving speed-up over the
    old per-α parametrisation, which re-ran the whole pipeline for each order.

    Uses the same small 300-sample / 61-bin grid as
    ``test_crts_is_empirically_proper``: the propriety margin is stable well
    below 1500 samples, so the coarser grid keeps a clear margin while avoiding
    the expensive full-resolution ``compute_scoring_rules`` call.
    """
    rng = np.random.default_rng(0)
    y = rng.normal(0.0, 1.0, size=300).astype(np.float32)
    edges = np.linspace(-6.0, 6.0, 61, dtype=np.float64)

    candidates = {
        "true(0,1)": (0.0, 1.0),
        "shift(0.5,1)": (0.5, 1.0),
        "narrow(0,0.6)": (0.0, 0.6),
        "wide(0,1.6)": (0.0, 1.6),
    }

    # One full pipeline evaluation per candidate; all α keys come back together.
    all_scores = {}
    for name, (mu, sigma) in candidates.items():
        dist = _gaussian_histogram_distribution(mu, sigma, edges, len(y))
        all_scores[name] = compute_scoring_rules(dist, y)

    for alpha in PSEUDOS_ALPHAS:
        key = f"pseudospherical_alpha_{alpha}"
        scores = {name: all_scores[name][key] for name in candidates}

        best = min(scores, key=scores.get)
        assert best == "true(0,1)", (
            f"pseudospherical (α={alpha}) is not proper: expected the true "
            f"forecast to win, got {best}. Scores: "
            + ", ".join(f"{n}={v:.4f}" for n, v in scores.items())
        )
        runner_up = min(v for n, v in scores.items() if n != "true(0,1)")
        assert scores["true(0,1)"] < runner_up, (
            f"True forecast should strictly beat the runner-up (α={alpha}): "
            f"true={scores['true(0,1)']:.4f} vs runner-up={runner_up:.4f}"
        )


# ============================================================================
# Pseudospherical: direct-kernel sweep over a broad range of orders α
# ============================================================================
#
# The production path only exercises the three reported orders (1.5, 2.0, 3.0).
# The tests below drive ``compute_pseudospherical_scores`` *directly* on a
# controlled piecewise-constant density so the closed form and the structural
# properties (bounded loss, perfect-forecast optimum) are pinned across a much
# wider α grid — including orders close to 1 (where the ratio form is most
# fragile) and large orders — without depending on any histogram plumbing.

# Broad sweep: near-1, the reported values, and well above the spherical order.
_PSEUDOS_ALPHA_SWEEP = [1.01, 1.05, 1.2, 1.5, 2.0, 3.0, 5.0]


def _make_density_terms(f, w):
    """Build the ``(g_y, density_integral)`` pair the kernel consumes.

    ``f`` is a per-bin density (shape ``(1, n_bins)``), ``w`` the matching bin
    widths.  ``density_integral(power)`` returns ``∫ f^power dt = Σ_k f_k^power
    w_k`` per sample, exactly as the production ``_density_terms`` closure does.
    """
    f_t = torch.as_tensor(f, dtype=torch.float64).reshape(1, -1)
    w_t = torch.as_tensor(w, dtype=torch.float64).reshape(1, -1)

    def density_integral(power):
        return (f_t.clamp(min=0.0).pow(power) * w_t).sum(dim=-1)

    return f_t, density_integral


def _pseudospherical_kernel(f, w, y_bin, alpha, eps=1e-10):
    """Run the module kernel for one density and order, returning the scalar loss."""
    f_t, density_integral = _make_density_terms(f, w)
    g_y = f_t[:, y_bin]                                  # f(y): value in the y-bin
    out = compute_pseudospherical_scores.__wrapped__(
        g_y, [alpha], density_integral=density_integral, eps=eps,
    )
    return out[f"pseudospherical_alpha_{alpha}"]


@pytest.mark.parametrize("alpha", _PSEUDOS_ALPHA_SWEEP)
def test_pseudospherical_kernel_matches_closed_form(alpha):
    """Kernel equals ``−1/(α−1)·[ f(y)^{α−1} / (∫f^α)^{(α−1)/α} − 1 ]`` for any α.

    Independent scalar re-derivation on a fixed, deliberately non-uniform
    density and bin grid.  Sweeping α from 1.01 to 5.0 catches an exponent
    mistake — e.g. ``(α−1)/α`` vs ``1/α`` on the norm, or the ``1/(α−1)`` affine
    factor — that would stay hidden at the single spherical order α = 2.
    """
    f = np.array([0.2, 1.3, 0.5, 0.8, 0.1])
    w = np.array([0.5, 0.4, 0.7, 0.3, 1.1])
    y_bin = 1                                            # target lands in bin 1

    got = _pseudospherical_kernel(f, w, y_bin, alpha)

    g_y = f[y_bin]
    norm_alpha = float((f ** alpha * w).sum())
    denom = norm_alpha ** ((alpha - 1.0) / alpha)
    ratio = g_y ** (alpha - 1.0) / denom
    ref = -((ratio - 1.0) / (alpha - 1.0))               # negated -> loss

    assert math.isclose(got, ref, rel_tol=1e-9, abs_tol=1e-12), (
        f"pseudospherical kernel (α={alpha}) = {got:.12f} != closed form "
        f"{ref:.12f}"
    )


@pytest.mark.parametrize("alpha", _PSEUDOS_ALPHA_SWEEP)
def test_pseudospherical_loss_is_bounded_and_finite(alpha):
    """The reported loss is finite and worst at an out-of-support target.

    Unlike the log score, the ratio form has a *bounded* point term
    ``f(y)^{α−1}`` that → 0 as ``f(y) → 0``, so an out-of-support target yields
    the largest (worst) but still finite loss ``1/(α−1)`` rather than +∞.  Any
    in-support target places positive density at ``y`` and so scores strictly
    better (a strictly smaller loss).  This pins the boundedness and
    support-insensitivity of the ratio form across every order in the sweep.

    (Note the loss is *not* guaranteed ≤ 0 at the densest bin: the affine
    ``−1/(α−1)·[(f(y)/‖f‖_α)^{α−1} − 1]`` is only non-positive once
    ``f(y) ≥ ‖f‖_α``, which need not hold for a nearly-flat density at small α.
    Boundedness and the strict in-support advantage are the properties that
    hold universally, so those are what we assert.)
    """
    f = np.array([0.1, 0.9, 1.4, 0.6, 0.2])
    w = np.array([0.6, 0.5, 0.4, 0.5, 1.0])

    # Any in-support target has positive density -> finite loss.
    dense_bin = int(np.argmax(f))
    best_loss = _pseudospherical_kernel(f, w, dense_bin, alpha)
    assert np.isfinite(best_loss), (
        f"α={alpha}: in-support loss must be finite, got {best_loss}"
    )

    # Worst case: an out-of-support target (f(y)=0) -> finite, equals 1/(α−1).
    f_oos = np.concatenate([f, [0.0]])                   # extra empty catch-all bin
    w_oos = np.concatenate([w, [1.0]])
    worst_loss = _pseudospherical_kernel(f_oos, w_oos, len(f), alpha)
    assert np.isfinite(worst_loss), (
        f"α={alpha}: out-of-support loss must be finite, got {worst_loss}"
    )
    assert math.isclose(worst_loss, 1.0 / (alpha - 1.0), rel_tol=1e-9), (
        f"α={alpha}: out-of-support loss should be the maximum 1/(α−1)="
        f"{1.0/(alpha-1.0):.6f}, got {worst_loss:.6f}"
    )
    # The out-of-support target is strictly the worst: any positive density at
    # the target strictly lowers the loss below the 1/(α−1) ceiling.
    assert worst_loss > best_loss, (
        f"α={alpha}: out-of-support ({worst_loss:.4f}) must be strictly worse "
        f"than the densest-bin target ({best_loss:.4f})"
    )


@pytest.mark.parametrize("alpha", _PSEUDOS_ALPHA_SWEEP)
def test_pseudospherical_kernel_is_empirically_proper(alpha):
    """Averaged over draws from the true density, the true forecast wins.

    A minimal, plumbing-free propriety check straight on the kernel: sample bin
    indices from a reference PMF, then compare the mean loss of the true density
    against a shifted and an over-dispersed competitor.  Proper ⇒ the truth has
    the lowest mean loss at every order in the sweep.
    """
    rng = np.random.default_rng(1)
    w = np.full(21, 0.5)                                 # uniform grid, width 0.5
    centre = 10

    def gaussian_density(mu_bin, sigma_bins):
        idx = np.arange(len(w))
        d = np.exp(-0.5 * ((idx - mu_bin) / sigma_bins) ** 2)
        f = d / (d * w).sum()                            # normalise ∫ f w = 1
        return f

    true_f = gaussian_density(centre, 3.0)
    # Draw targets from the true PMF (mass = f_k w_k).
    pmf = true_f * w
    pmf = pmf / pmf.sum()
    y_bins = rng.choice(len(w), size=400, p=pmf)

    candidates = {
        "true": gaussian_density(centre, 3.0),
        "shifted": gaussian_density(centre + 2, 3.0),
        "over-dispersed": gaussian_density(centre, 5.0),
    }

    mean_loss = {}
    for name, f in candidates.items():
        losses = [_pseudospherical_kernel(f, w, int(k), alpha) for k in y_bins]
        mean_loss[name] = float(np.mean(losses))

    best = min(mean_loss, key=mean_loss.get)
    assert best == "true", (
        f"α={alpha}: pseudospherical kernel not proper — expected 'true' to "
        f"win, got '{best}'. Mean losses: "
        + ", ".join(f"{n}={v:.5f}" for n, v in mean_loss.items())
    )
