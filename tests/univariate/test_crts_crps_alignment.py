"""Tests for the CRTS (Continuous Ranked Tsallis Score) discretisation.

The central property under test: at Tsallis order ``α = 2`` the CRTS integrand
is the Brier divergence ``(F(t) − 1{t ≥ y})²``, whose integral is *exactly* the
continuous CRPS — which the benchmark computes as the energy score at ``β = 1``.
The two are the SAME continuous functional, so on a histogram they must agree
to numerical precision *provided both use the same discretisation*.

The production ``compute_crts`` uses a slab-aligned discretisation (CDF rises
linearly within each bin, the bin containing ``y`` is split at ``y`` exactly)
that mirrors the energy score's uniform-slab assumption.  These tests lock in
that alignment across a range of grids and target positions, plus the usual
finiteness / ordering / tail-correction sanity checks.

Reference (independent) CRPS is computed by a fine-grid numerical integral of
the Brier divergence, entirely separate from the production code path.
"""

import numpy as np
import pytest
import torch

# Force CPU so the tests are deterministic and independent of GPU availability.
torch.cuda.is_available = lambda: False

from scoringbench.univariate.metrics import (  # noqa: E402
    compute_scoring_rules,
    CRTS_ALPHAS,
    _crts_slab_integral,
)
from scoringbench.univariate.wrappers import DistributionPrediction  # noqa: E402


# ============================================================================
# Independent reference: continuous CRPS via fine-grid Brier integral
# ============================================================================

def _reference_crps(bin_edges, probas, y, pad=5.0, n=400_001):
    """Continuous CRPS of a piecewise-linear-CDF histogram, by fine quadrature.

    The histogram PMF ``probas`` over ``bin_edges`` defines a CDF that rises
    linearly within each bin (uniform slab).  The CRPS is

        CRPS(F, y) = ∫ (F(t) − 1{t ≥ y})² dt

    which is exactly the energy score at β = 1.  We evaluate it on a dense grid
    that is independent of the production discretisation.
    """
    edges = np.asarray(bin_edges, dtype=np.float64)
    p = np.asarray(probas, dtype=np.float64).ravel()
    cum = np.concatenate([[0.0], np.cumsum(p)])  # CDF at edges, F(edges[0]) = 0

    lo = min(edges[0], float(y)) - pad
    hi = max(edges[-1], float(y)) + pad
    t = np.linspace(lo, hi, n)

    # Piecewise-linear interpolation of the CDF between edge values; flat
    # (0 below the grid, 1 above) outside the support.
    F = np.interp(t, edges, cum, left=0.0, right=1.0)
    indicator = (t >= float(y)).astype(np.float64)
    integrand = (F - indicator) ** 2
    return float(np.trapezoid(integrand, t))


def _make_dist(bin_edges, probas):
    """Build a DistributionPrediction from edges + a single-row PMF."""
    edges = np.asarray(bin_edges, dtype=np.float32)
    mids = 0.5 * (edges[:-1] + edges[1:])
    probas = np.asarray(probas, dtype=np.float32).reshape(1, -1)
    mean = np.array([float((mids * probas.ravel()).sum())], dtype=np.float64)
    return DistributionPrediction(
        probas=probas,
        bin_edges=edges,
        bin_midpoints=mids.astype(np.float32),
        mean=mean,
    )


# ============================================================================
# Test cases: (name, bin_edges, probas, y)
# ============================================================================

def _gaussian_hist(n_bins, lo=-4.0, hi=4.0, mu=0.0, sigma=1.0):
    """Discretised standard-normal PMF on a uniform grid."""
    edges = np.linspace(lo, hi, n_bins + 1)
    from scipy.stats import norm
    cdf = norm.cdf(edges, loc=mu, scale=sigma)
    p = np.diff(cdf)
    p = p / p.sum()
    return edges, p


_CASES = {
    "coarse_symmetric_y_mid": (
        np.array([0.0, 1.0, 2.0, 3.0]),
        np.array([0.25, 0.5, 0.25]),
        1.5,
    ),
    "coarse_y_at_edge": (
        np.array([0.0, 1.0, 2.0, 3.0]),
        np.array([0.25, 0.5, 0.25]),
        1.0,
    ),
    "coarse_y_low_in_bin": (
        np.array([0.0, 1.0, 2.0, 3.0]),
        np.array([0.25, 0.5, 0.25]),
        0.3,
    ),
    "coarse_y_high_in_bin": (
        np.array([0.0, 1.0, 2.0, 3.0]),
        np.array([0.25, 0.5, 0.25]),
        0.9,
    ),
    "skewed": (
        np.array([-2.0, 0.0, 1.0, 1.5, 4.0]),
        np.array([0.1, 0.4, 0.3, 0.2]),
        0.7,
    ),
    "gauss_32bin": (*_gaussian_hist(32), 0.35),
    "gauss_256bin": (*_gaussian_hist(256), -1.2),
}


@pytest.mark.parametrize("name", list(_CASES))
def test_crts_alpha2_matches_energy_beta1(name):
    """crts_alpha_2.0 must equal energy_score_beta_1.0 (same functional)."""
    edges, probas, y = _CASES[name]
    dist = _make_dist(edges, probas)
    y_true = np.array([y], dtype=np.float32)

    m = compute_scoring_rules(dist, y_true)
    crts = m["crts_alpha_2.0"]
    es_b1 = m["energy_score_beta_1.0"]

    # The two are the SAME continuous functional evaluated on the SAME
    # discretisation, so they agree analytically.  The residual (~1e-8) is
    # pure float32-input round-off from building the DistributionPrediction,
    # not a discretisation gap, hence the 1e-7 bound rather than 0.
    assert abs(crts - es_b1) < 1e-7, (
        f"[{name}] crts_alpha_2.0={crts!r} vs energy_score_beta_1.0={es_b1!r}; "
        f"|Δ|={abs(crts - es_b1):.3e}"
    )


@pytest.mark.parametrize("name", list(_CASES))
def test_crts_alpha2_matches_independent_crps(name):
    """crts_alpha_2.0 must equal the independent fine-grid CRPS reference.

    ``DistributionPrediction`` resamples every input histogram onto its own
    regular grid (``regrid_to_uniform``); for coarse, non-uniform inputs that
    resampling *changes* the distribution.  The fine-grid reference must
    therefore integrate the SAME distribution the pipeline scores — i.e. the
    regridded ``dist.bin_edges``/``dist.probas`` — otherwise a coarse skewed
    grid produces a spurious mismatch that has nothing to do with the CRTS
    quadrature under test.
    """
    edges, probas, y = _CASES[name]
    dist = _make_dist(edges, probas)
    y_true = np.array([y], dtype=np.float32)

    m = compute_scoring_rules(dist, y_true)
    crts = m["crts_alpha_2.0"]
    ref = _reference_crps(
        np.asarray(dist.bin_edges).ravel(),
        np.asarray(dist.probas).ravel(),
        y,
    )

    # Fine-grid quadrature carries a small discretisation error of its own.
    assert abs(crts - ref) < 1e-4, (
        f"[{name}] crts_alpha_2.0={crts!r} vs reference CRPS={ref!r}; "
        f"|Δ|={abs(crts - ref):.3e}"
    )


@pytest.mark.parametrize("name", list(_CASES))
@pytest.mark.parametrize("alpha", CRTS_ALPHAS)
def test_crts_is_finite_and_nonnegative(name, alpha):
    """Every CRTS value must be finite and non-negative."""
    edges, probas, y = _CASES[name]
    dist = _make_dist(edges, probas)
    y_true = np.array([y], dtype=np.float32)

    m = compute_scoring_rules(dist, y_true)
    key = f"crts_alpha_{alpha}"
    assert key in m, f"missing {key}"
    val = m[key]
    assert np.isfinite(val), f"[{name}] {key} not finite: {val!r}"
    assert val >= -1e-9, f"[{name}] {key} negative: {val!r}"


def test_crts_alpha2_perfect_prediction_is_small():
    """A near-Dirac forecast at y gives a near-zero CRTS at α=2."""
    # Narrow mass around y = 0 on a fine grid.
    edges, p = _gaussian_hist(512, lo=-4.0, hi=4.0, mu=0.0, sigma=0.05)
    dist = _make_dist(edges, p)
    y_true = np.array([0.0], dtype=np.float32)

    m = compute_scoring_rules(dist, y_true)
    assert m["crts_alpha_2.0"] < 0.05
    # Still consistent with the energy score (1e-7 covers float32 round-off).
    assert abs(m["crts_alpha_2.0"] - m["energy_score_beta_1.0"]) < 1e-7


def test_crts_alpha2_ordering_sharp_vs_diffuse():
    """A sharper (correct) forecast scores lower than a diffuse one at α=2."""
    y = 0.0
    edges_s, p_s = _gaussian_hist(256, sigma=0.3)
    edges_d, p_d = _gaussian_hist(256, sigma=2.0)

    m_s = compute_scoring_rules(_make_dist(edges_s, p_s), np.array([y], np.float32))
    m_d = compute_scoring_rules(_make_dist(edges_d, p_d), np.array([y], np.float32))

    assert m_s["crts_alpha_2.0"] < m_d["crts_alpha_2.0"]


def test_crts_alpha2_tail_correction_y_outside_grid():
    """When y falls outside the grid the padded tail keeps α=2 ≙ β=1.

    ``pad_to_common_grid`` appends a single zero-mass catch-all bin whose
    flat-CDF slab integral reproduces the exact tail contribution gap/(α−1).
    There is no *separate* analytic term (adding one would double-count the
    tail — see ``test_crts_no_tail_double_count`` below), so α=2 must still
    equal the β=1 energy score.
    """
    edges = np.array([0.0, 1.0, 2.0, 3.0])
    probas = np.array([0.25, 0.5, 0.25])
    y_true = np.array([5.0], dtype=np.float32)

    m = compute_scoring_rules(_make_dist(edges, probas), y_true)
    assert abs(m["crts_alpha_2.0"] - m["energy_score_beta_1.0"]) < 1e-7


# ============================================================================
# Explicit regression: the out-of-support tail must be counted EXACTLY ONCE
# ============================================================================

def _tail_gap_contribution(gap, alpha):
    """Analytic α-Tsallis integral over a flat-CDF tail slab of length ``gap``.

    Beyond the support the CDF is flat (F ≡ 1 on the right, F ≡ 0 on the left)
    while the indicator disagrees, so the divergence-form integrand collapses to
    the constant ``s_α = 1/(α−1)``.  Integrated over a slab of length ``gap`` the
    tail therefore adds exactly ``gap / (α−1)`` — and *only* this, once.
    """
    return gap / (alpha - 1.0)


@pytest.mark.parametrize("alpha", [a for a in CRTS_ALPHAS if abs(a - 2.0) < 1e-9])
@pytest.mark.parametrize("gap", [0.5, 1.0, 2.0, 3.5])
def test_crts_no_tail_double_count_right(alpha, gap):
    """A right-tail target must add gap/(α−1) ONCE, not twice.

    Regression for a double-counting bug: the grid is padded with a zero-mass
    catch-all bin whose flat-CDF slab integral already yields gap/(α−1).  A
    stray *analytic* tail term on top of that would inflate CRTS by a second
    gap/(α−1).  We isolate the tail by differencing the far-target score
    against the same forecast scored at the grid edge (gap = 0), and require
    the difference to equal a SINGLE gap/(α−1).
    """
    edges = np.array([0.0, 1.0, 2.0, 3.0])
    probas = np.array([0.25, 0.5, 0.25])
    dist = _make_dist(edges, probas)

    hi = float(np.asarray(dist.bin_edges).ravel()[-1])
    y_edge = np.array([hi], dtype=np.float32)      # target at the right edge
    y_far = np.array([hi + gap], dtype=np.float32)  # target gap beyond it

    m_edge = compute_scoring_rules(dist, y_edge)["crts_alpha_2.0"]
    m_far = compute_scoring_rules(dist, y_far)["crts_alpha_2.0"]

    observed_tail = m_far - m_edge
    expected_tail = _tail_gap_contribution(gap, alpha)

    # A double count would make observed_tail ≈ 2 * expected_tail.
    assert abs(observed_tail - expected_tail) < 1e-6, (
        f"right tail gap={gap}: observed Δ={observed_tail:.6f} "
        f"expected {expected_tail:.6f} (2x would be {2*expected_tail:.6f}); "
        f"|err|={abs(observed_tail - expected_tail):.3e}"
    )
    # And the whole thing must still coincide with the β=1 energy score.
    es_far = compute_scoring_rules(dist, y_far)["energy_score_beta_1.0"]
    assert abs(m_far - es_far) < 1e-7


@pytest.mark.parametrize("alpha", [a for a in CRTS_ALPHAS if abs(a - 2.0) < 1e-9])
@pytest.mark.parametrize("gap", [0.5, 1.0, 2.0, 3.5])
def test_crts_no_tail_double_count_left(alpha, gap):
    """A left-tail target must add gap/(α−1) ONCE, not twice (mirror image)."""
    edges = np.array([0.0, 1.0, 2.0, 3.0])
    probas = np.array([0.25, 0.5, 0.25])
    dist = _make_dist(edges, probas)

    lo = float(np.asarray(dist.bin_edges).ravel()[0])
    y_edge = np.array([lo], dtype=np.float32)
    y_far = np.array([lo - gap], dtype=np.float32)

    m_edge = compute_scoring_rules(dist, y_edge)["crts_alpha_2.0"]
    m_far = compute_scoring_rules(dist, y_far)["crts_alpha_2.0"]

    observed_tail = m_far - m_edge
    expected_tail = _tail_gap_contribution(gap, alpha)

    assert abs(observed_tail - expected_tail) < 1e-6, (
        f"left tail gap={gap}: observed Δ={observed_tail:.6f} "
        f"expected {expected_tail:.6f} (2x would be {2*expected_tail:.6f}); "
        f"|err|={abs(observed_tail - expected_tail):.3e}"
    )
    es_far = compute_scoring_rules(dist, y_far)["energy_score_beta_1.0"]
    assert abs(m_far - es_far) < 1e-7


def test_crts_slab_flat_tail_counts_gap_once():
    """Slab-level check: a flat-CDF tail integrates to exactly gap/(α−1).

    Exercises ``_crts_slab_integral`` directly on the degenerate tail geometry
    (F_lo = F_hi, indicator disagreeing) that ``pad_to_common_grid`` produces.
    The result must be the SINGLE gap/(α−1) contribution, confirming the padded
    catch-all bin — not a bolted-on analytic term — is what carries the tail.
    """
    for alpha in (1.01, 1.2, 1.5, 2.0):
        for gap in (0.5, 1.0, 2.0):
            # Right tail: F ≡ 1, indicator q = 0 over the slab.
            right = _crts_slab_integral(
                torch.tensor(1.0, dtype=torch.float64),
                torch.tensor(1.0, dtype=torch.float64),
                torch.tensor(gap, dtype=torch.float64),
                0.0, alpha,
            ).item()
            # Left tail: F ≡ 0, indicator q = 1 over the slab.
            left = _crts_slab_integral(
                torch.tensor(0.0, dtype=torch.float64),
                torch.tensor(0.0, dtype=torch.float64),
                torch.tensor(gap, dtype=torch.float64),
                1.0, alpha,
            ).item()
            expected = gap / (alpha - 1.0)
            # With the natural [0, 1] clamp the certain-and-wrong point term is
            # (1−F)^{α−1} = 0^{α−1} = 0 *exactly* at every order, so the tail
            # equals gap/(α−1) to machine precision for all α — including
            # α=1.01, where the old [eps, 1−eps] clamp left a ~73% residual.
            # A double count would be a full extra gap/(α−1).
            tol = 1e-9 * expected
            assert abs(right - expected) < tol, (
                f"right flat tail α={alpha} gap={gap}: {right} vs {expected}"
            )
            assert abs(left - expected) < tol, (
                f"left flat tail α={alpha} gap={gap}: {left} vs {expected}"
            )


# ============================================================================
# Independent SciPy check of the slab-integral quadrature helper
# ============================================================================

def _scipy_slab_integral(F_lo, F_hi, width, q, alpha):
    """Independent SciPy ``quad`` evaluation of ∫ s_α(F(u), q) du over a slab.

    Re-derives the divergence-form α-Tsallis integrand from scratch (no shared
    code with the production helper) and integrates it adaptively with
    ``scipy.integrate.quad`` over the slab coordinate ``t ∈ [0, 1]``, scaled by
    ``width``.  Used to confirm the fixed 8-point Gauss–Legendre rule in
    ``_crts_slab_integral`` returns the same value.  Mirrors the production
    helper's *natural* ``[0, 1]`` clamp (not the biased ``[eps, 1-eps]``).
    """
    from scipy.integrate import quad

    offset = 1.0 / alpha - 1.0 / (alpha - 1.0)

    def integrand(t):
        F = F_lo + t * (F_hi - F_lo)
        F = min(max(F, 0.0), 1.0)
        integral_term = (F ** alpha + (1.0 - F) ** alpha) / alpha
        if q == 1.0:
            point_term = F ** (alpha - 1.0) / (alpha - 1.0)
        else:
            point_term = (1.0 - F) ** (alpha - 1.0) / (alpha - 1.0)
        return integral_term - point_term - offset

    val, _ = quad(integrand, 0.0, 1.0, epsabs=1e-12, epsrel=1e-12)
    return val * width


_SLAB_CASES = [
    # (F_lo, F_hi, width, q)
    (0.0, 0.25, 1.0, 0.0),
    (0.25, 0.75, 1.0, 0.0),
    (0.25, 0.75, 1.0, 1.0),
    (0.75, 1.0, 0.5, 1.0),
    (0.1, 0.9, 2.3, 0.0),
    (0.4, 0.6, 0.7, 1.0),
    (0.02, 0.98, 1.0, 0.0),
    (0.5, 0.5, 1.0, 1.0),   # flat CDF (degenerate slope)
]


@pytest.mark.parametrize("F_lo,F_hi,width,q", _SLAB_CASES)
@pytest.mark.parametrize("alpha", CRTS_ALPHAS)
def test_slab_integral_matches_scipy_quad(F_lo, F_hi, width, q, alpha):
    """_crts_slab_integral must match an independent scipy.integrate.quad.

    The production helper uses a fixed 16-point Gauss–Legendre rule; SciPy's
    adaptive ``quad`` is a completely independent integrator.  At α=2 the
    integrand is quadratic so agreement is at machine precision; for α∈(1,2)
    the fixed rule carries a small truncation error (worst ~1e-6 on wide slabs
    with α→1), hence the looser bound.
    """
    got = _crts_slab_integral(
        torch.tensor(F_lo, dtype=torch.float64),
        torch.tensor(F_hi, dtype=torch.float64),
        torch.tensor(width, dtype=torch.float64),
        float(q), float(alpha),
    ).item()
    ref = _scipy_slab_integral(F_lo, F_hi, width, float(q), float(alpha))

    tol = 1e-11 if abs(alpha - 2.0) < 1e-9 else 2e-5
    assert abs(got - ref) < tol, (
        f"α={alpha} F_lo={F_lo} F_hi={F_hi} w={width} q={q}: "
        f"helper={got!r} scipy={ref!r} |Δ|={abs(got - ref):.3e}"
    )


def test_slab_integral_broadcasts_over_batch():
    """The helper vectorises over (n_samples, n_bins) matching per-element quad."""
    torch.manual_seed(0)
    F_lo = torch.rand(4, 5, dtype=torch.float64) * 0.5
    F_hi = F_lo + torch.rand(4, 5, dtype=torch.float64) * 0.5
    width = torch.rand(4, 5, dtype=torch.float64) + 0.1
    alpha = 1.5

    got = _crts_slab_integral(F_lo, F_hi, width, 0.0, alpha)
    for i in range(4):
        for j in range(5):
            ref = _scipy_slab_integral(
                F_lo[i, j].item(), F_hi[i, j].item(), width[i, j].item(),
                0.0, alpha,
            )
            assert abs(got[i, j].item() - ref) < 1e-6


def test_crts_batch_consistency():
    """Batched evaluation matches per-sample evaluation at α=2."""
    edges, p = _gaussian_hist(64)
    ys = np.array([-1.0, 0.0, 0.4, 1.3], dtype=np.float32)

    probas = np.repeat(p[None, :].astype(np.float32), len(ys), axis=0)
    mids = 0.5 * (edges[:-1] + edges[1:])
    mean = (probas * mids[None, :]).sum(axis=1).astype(np.float64)
    dist = DistributionPrediction(
        probas=probas,
        bin_edges=edges.astype(np.float32),
        bin_midpoints=mids.astype(np.float32),
        mean=mean,
    )
    m_batch = compute_scoring_rules(dist, ys)

    # Mean over samples of per-sample crts must equal the batch crts.
    per = []
    for yi in ys:
        di = _make_dist(edges, p)
        per.append(compute_scoring_rules(di, np.array([yi], np.float32))["crts_alpha_2.0"])
    assert abs(m_batch["crts_alpha_2.0"] - float(np.mean(per))) < 1e-9
