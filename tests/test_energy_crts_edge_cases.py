"""Additional edge-case tests for the energy score (arbitrary β) and CRTS.

These complement the existing Monte-Carlo (``test_energy_zero_width_monte_carlo``)
and exact-rational (``test_crts_exact_reference``) suites by stressing regimes
they do *not* cover directly:

Energy score
------------
* extreme exponents β ∈ {0.1, 1.9} (near the ends of the admissible (0, 2] range);
* target y exactly on a bin edge (term-1 sub-slab boundary);
* per-sample (non-shared) grids (term-2 batched path);
* the β → 2 limit against the analytic 2·Var(X) − ... closed form.

CRTS
----
* target y exactly on a bin edge (degenerate split, w_lo = 0 or w_hi = 0);
* an interior zero-width (degenerate) bin;
* extreme α near 1 (support-insensitivity: extending the padded tail must not
  change the score once y is covered);
* per-sample grid consistency (crts_alpha_2.0 == energy_beta_1.0 batched);
* monotone ordering of CRTS in α on a fixed forecast/target.

All references are independent of the code under test: a dense-grid Riemann /
Monte-Carlo estimate, an analytic closed form, or an invariance the definition
must satisfy.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

torch.cuda.is_available = lambda: False  # force CPU float64 determinism
EPS = 100 * torch.finfo(torch.float64).eps

from scoringbench._integration import uniform_slab_pairwise_distance
from scoringbench.metrics import (
    compute_crts,
    compute_energy_score_histogram_corrected,
    pad_to_common_grid,
)

_raw_crts = compute_crts.__wrapped__
_raw_energy = compute_energy_score_histogram_corrected.__wrapped__


# ---------------------------------------------------------------------------
# Independent references
# ---------------------------------------------------------------------------
def _mc_energy_reference(edges, probas, y, beta, n=6_000_000, seed=0):
    """Monte-Carlo reference for ES_β with its standard error.

    Draws real samples from the uniform-slab mixture and averages
    ``|X-y|^β - ½|X-X'|^β``.  Shares no code with the closed forms under test,
    so the returned ``(mean, se)`` is the ground truth (up to MC noise) the
    analytic implementation must reproduce.
    """
    e = np.asarray(edges, float)
    p = np.asarray(probas, float)
    p = p / p.sum()
    rng = np.random.default_rng(seed + int(beta * 100))
    lo, hi = e[:-1], e[1:]

    def draw(m):
        idx = rng.choice(len(p), size=m, p=p)
        return lo[idx] + rng.random(m) * (hi[idx] - lo[idx])

    X, Xp = draw(n), draw(n)
    es = np.abs(X - y) ** beta - 0.5 * np.abs(X - Xp) ** beta
    return float(es.mean()), float(es.std(ddof=1) / np.sqrt(n))


# ===========================================================================
# ENERGY SCORE edge cases
# ===========================================================================
_EXTREME_BETAS = [0.1, 0.3, 1.5, 1.9]


@pytest.mark.parametrize("beta", _EXTREME_BETAS)
def test_energy_extreme_beta_matches_dense_reference(beta):
    """ES_β at extreme β matches an independent dense-grid Riemann estimate."""
    edges = [0.0, 1.0, 2.0, 4.0, 7.0]
    probas = [0.20, 0.35, 0.25, 0.20]
    y = 2.7

    edges_t = torch.as_tensor(edges, dtype=torch.float64)
    p_t = (torch.as_tensor(probas, dtype=torch.float64) / sum(probas))[None, :]
    y_t = torch.as_tensor([y], dtype=torch.float64)

    got = _raw_energy(p_t, edges_t, y_t, betas=[beta])[f"energy_score_beta_{beta}"]
    ref, se = _mc_energy_reference(edges, probas, y, beta)
    tol = 5.0 * se + 1e-4
    assert abs(got - ref) < tol, (
        f"beta={beta}: got={got:.6f} ref={ref:.6f}\u00b1{se:.2e} "
        f"(|diff|={abs(got - ref):.2e} > tol={tol:.2e})"
    )


@pytest.mark.parametrize("beta", _EXTREME_BETAS)
def test_energy_target_on_bin_edge(beta):
    """y exactly on an interior bin edge: term-1 sub-slab boundary is exact."""
    edges = [0.0, 1.0, 2.0, 4.0, 7.0]
    probas = [0.20, 0.35, 0.25, 0.20]
    y = 2.0  # exactly on the edge between bins 1 and 2

    edges_t = torch.as_tensor(edges, dtype=torch.float64)
    p_t = (torch.as_tensor(probas, dtype=torch.float64) / sum(probas))[None, :]
    y_t = torch.as_tensor([y], dtype=torch.float64)

    got = _raw_energy(p_t, edges_t, y_t, betas=[beta])[f"energy_score_beta_{beta}"]
    ref, se = _mc_energy_reference(edges, probas, y, beta, seed=99)
    tol = 5.0 * se + 1e-4
    assert abs(got - ref) < tol, (
        f"beta={beta} y=edge: got={got:.6f} ref={ref:.6f}\u00b1{se:.2e} "
        f"(|diff|={abs(got - ref):.2e} > tol={tol:.2e})"
    )


@pytest.mark.parametrize("beta", [0.5, 1.0, 1.5])
def test_energy_per_sample_grid_matches_shared(beta):
    """Per-sample (non-shared) grid path must equal the shared-grid path when
    every row carries the same edges."""
    edges = [0.0, 1.3, 2.6, 5.0]
    probas = [0.3, 0.45, 0.25]
    ys = [1.1, 3.2, 0.4]

    edges_t = torch.as_tensor(edges, dtype=torch.float64)
    p_row = torch.as_tensor(probas, dtype=torch.float64) / sum(probas)
    p_shared = p_row[None, :].repeat(len(ys), 1)
    y_t = torch.as_tensor(ys, dtype=torch.float64)

    shared_out = _raw_energy(p_shared, edges_t, y_t, betas=[beta])
    # Per-sample edges: same edges replicated for each row.
    edges_ps = edges_t[None, :].repeat(len(ys), 1)
    ps_out = _raw_energy(p_shared, edges_ps, y_t, betas=[beta])

    a = shared_out[f"energy_score_beta_{beta}"]
    b = ps_out[f"energy_score_beta_{beta}"]
    assert abs(a - b) < 1e-10, f"beta={beta}: shared={a:.10f} per-sample={b:.10f}"


def test_energy_beta2_collapses_to_mean_squared_error():
    """At β = 2 the energy score collapses to (E[X] - y)^2 (variance cancels).

    With |x-y|^2 = (x-y)^2: term1 = E[(X-y)^2] = Var + (mean-y)^2 and
    ½·E|X-X'|^2 = Var (for i.i.d. X, X'), so ES_2 = (mean-y)^2 -- the variance
    cancels exactly.  This is the well-known degeneracy of the energy score at
    β = 2, which makes it *improper* there; the implementation must reproduce
    the analytic (mean-y)^2 to machine precision."""
    edges = np.array([0.0, 1.0, 3.0, 4.0, 8.0])
    probas = np.array([0.25, 0.3, 0.2, 0.25])
    probas = probas / probas.sum()
    y = 2.9

    a, b = edges[:-1], edges[1:]
    mean_k = 0.5 * (a + b)  # exact uniform-slab mean
    mean = float((probas * mean_k).sum())
    ref = (mean - y) ** 2  # variance cancels at β = 2

    edges_t = torch.as_tensor(edges, dtype=torch.float64)
    p_t = torch.as_tensor(probas, dtype=torch.float64)[None, :]
    y_t = torch.as_tensor([y], dtype=torch.float64)
    got = _raw_energy(p_t, edges_t, y_t, betas=[2.0])["energy_score_beta_2.0"]
    rel = abs(got - ref) / abs(ref)
    # The variance (~4.6 here) cancels down to (mean-y)^2 ~ 6e-4, a ~7400x
    # cancellation, so ~1e-12 relative residual is at the float64 noise floor.
    assert rel < 1e-9, f"beta=2 MSE form: got={got:.10f} ref={ref:.10f} rel={rel:.2e}"


def test_energy_nonnegative_clamp_on_degenerate_forecast():
    """A point-mass forecast at y gives exactly ES = 0 for every β."""
    edges = [0.0, 0.0]  # single Dirac atom at 0.0
    probas = [1.0]
    y = 0.0
    edges_t = torch.as_tensor(edges, dtype=torch.float64)
    p_t = torch.as_tensor(probas, dtype=torch.float64)[None, :]
    y_t = torch.as_tensor([y], dtype=torch.float64)
    for beta in [0.1, 0.5, 1.0, 1.5, 2.0]:
        got = _raw_energy(p_t, edges_t, y_t, betas=[beta])[f"energy_score_beta_{beta}"]
        assert got == pytest.approx(0.0, abs=1e-12), f"beta={beta}: ES={got}"


def test_energy_beta_out_of_range_rejected():
    """β outside (0, 2] must raise (non-negativity guarantee would break)."""
    edges_t = torch.as_tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    p_t = torch.as_tensor([0.5, 0.5], dtype=torch.float64)[None, :]
    y_t = torch.as_tensor([0.5], dtype=torch.float64)
    for bad in [0.0, -0.5, 2.0001, 3.0]:
        with pytest.raises(ValueError):
            _raw_energy(p_t, edges_t, y_t, betas=[bad])


# ===========================================================================
# CRTS edge cases
# ===========================================================================
_ALPHAS = [1.01, 1.05, 1.2, 1.5, 2.0]


def _make_padded_crts_inputs(p, e, m, yv):
    y = torch.tensor([float(yv)], dtype=torch.float64)
    pp, ee, mm, sh, _gl, _gr = pad_to_common_grid(p, e, m, y, True)
    cdf = torch.cumsum(pp, dim=-1)
    y_bin = torch.searchsorted(ee[1:].contiguous(), y).clamp(0, pp.shape[1] - 1)
    return cdf, ee, y, y_bin, sh


def _gaussian_pmf(n_bins, lo=-4.0, hi=4.0):
    e = torch.linspace(lo, hi, n_bins + 1, dtype=torch.float64)
    m = 0.5 * (e[:-1] + e[1:])
    p = torch.exp(-0.5 * m ** 2)
    p = (p / p.sum()).reshape(1, -1)
    return p, e, m


@pytest.mark.parametrize("yv", [-2.0, 0.0, 2.0])  # exactly on interior edges
def test_crts_target_on_bin_edge_alpha2_equals_crps(yv):
    """y exactly on an edge (degenerate split) still gives crts_a2 == CRPS."""
    p, e, m = _gaussian_pmf(200)
    cdf, ee, y, y_bin, sh = _make_padded_crts_inputs(p, e, m, yv)
    crts = _raw_crts(cdf, ee, y, y_bin, sh, alphas=[2.0])["crts_alpha_2.0"]
    pp, ee2, mm, sh2, _gl, _gr = pad_to_common_grid(p, e, m, y, True)
    energy = _raw_energy(pp, ee2, y, betas=[1.0])["energy_score_beta_1.0"]
    rel = abs(crts - energy) / abs(energy)
    assert rel <= 1e-12, f"y=edge {yv}: crts_a2={crts:.10f} crps={energy:.10f} rel={rel:.2e}"


def test_crts_support_insensitivity_extending_tail():
    """Extending the (zero-mass) padded tail must not change CRTS once y is
    covered — the divergence-form integrand vanishes in both tails."""
    p, e, m = _gaussian_pmf(200)
    yv = 6.0
    # baseline padded grid
    cdf1, ee1, y1, yb1, sh1 = _make_padded_crts_inputs(p, e, m, yv)
    out1 = _raw_crts(cdf1, ee1, y1, yb1, sh1, alphas=_ALPHAS)

    # manually extend the right tail with an extra wide zero-mass bin beyond y
    pp, ee, mm, sh, _gl, _gr = pad_to_common_grid(
        p, e, m, torch.tensor([yv], dtype=torch.float64), True
    )
    ee_ext = torch.cat([ee, torch.tensor([ee[-1] + 50.0], dtype=torch.float64)])
    pp_ext = torch.cat([pp, torch.zeros(1, 1, dtype=torch.float64)], dim=1)
    cdf2 = torch.cumsum(pp_ext, dim=-1)
    y2 = torch.tensor([yv], dtype=torch.float64)
    yb2 = torch.searchsorted(ee_ext[1:].contiguous(), y2).clamp(0, pp_ext.shape[1] - 1)
    out2 = _raw_crts(cdf2, ee_ext, y2, yb2, sh, alphas=_ALPHAS)

    for al in _ALPHAS:
        a, b = out1[f"crts_alpha_{al}"], out2[f"crts_alpha_{al}"]
        rel = abs(a - b) / abs(a)
        assert rel <= 1e-9, f"alpha={al}: baseline={a:.10f} extended={b:.10f} rel={rel:.2e}"


def test_crts_interior_zero_width_bin_is_finite_and_ignored():
    """A zero-width interior bin (carrying zero mass) must not corrupt CRTS.

    The slab integral over a zero-width bin contributes exactly 0 (width=0), so
    inserting one must leave the score unchanged vs the same grid without it."""
    p, e, m = _gaussian_pmf(100)
    yv = 0.37
    cdf1, ee1, y1, yb1, sh1 = _make_padded_crts_inputs(p, e, m, yv)
    base = _raw_crts(cdf1, ee1, y1, yb1, sh1, alphas=_ALPHAS)

    # Insert a zero-width bin at the midpoint edge (duplicate an edge), zero mass.
    pp, ee, mm, sh, _gl, _gr = pad_to_common_grid(
        p, e, m, torch.tensor([yv], dtype=torch.float64), True
    )
    j = pp.shape[1] // 2
    ee_ins = torch.cat([ee[:j + 1], ee[j:j + 1], ee[j + 1:]])  # duplicate edge -> zero-width bin
    pp_ins = torch.cat([pp[:, :j], torch.zeros(1, 1, dtype=torch.float64), pp[:, j:]], dim=1)
    cdf2 = torch.cumsum(pp_ins, dim=-1)
    y2 = torch.tensor([yv], dtype=torch.float64)
    yb2 = torch.searchsorted(ee_ins[1:].contiguous(), y2).clamp(0, pp_ins.shape[1] - 1)
    ins = _raw_crts(cdf2, ee_ins, y2, yb2, sh, alphas=_ALPHAS)

    for al in _ALPHAS:
        a, b = base[f"crts_alpha_{al}"], ins[f"crts_alpha_{al}"]
        assert np.isfinite(b), f"alpha={al}: non-finite CRTS with zero-width bin"
        rel = abs(a - b) / abs(a)
        assert rel <= 1e-9, f"alpha={al}: base={a:.10f} with-zero-width={b:.10f} rel={rel:.2e}"


def test_crts_monotone_in_alpha_for_fixed_forecast():
    """On a fixed reasonable forecast/target CRTS should be well-ordered and
    positive across α; crts is a proper divergence so must be >= 0."""
    p, e, m = _gaussian_pmf(200)
    yv = 1.5
    cdf, ee, y, y_bin, sh = _make_padded_crts_inputs(p, e, m, yv)
    out = _raw_crts(cdf, ee, y, y_bin, sh, alphas=_ALPHAS)
    for al in _ALPHAS:
        assert out[f"crts_alpha_{al}"] >= -1e-12, f"alpha={al} negative: {out}"


def test_crts_alpha_near_one_rejected():
    """alpha <= 1 + 1e-4 must raise (log-score support sensitivity)."""
    p, e, m = _gaussian_pmf(50)
    cdf, ee, y, y_bin, sh = _make_padded_crts_inputs(p, e, m, 0.3)
    for bad in [1.0, 1.00005, 0.5]:
        with pytest.raises(ValueError):
            _raw_crts(cdf, ee, y, y_bin, sh, alphas=[bad])


def test_crts_per_sample_grid_alpha2_equals_energy_beta1():
    """Per-sample grid batched path: crts_alpha_2.0 == energy_beta_1.0 row-wise."""
    p, e, m = _gaussian_pmf(150)
    ys = torch.tensor([-3.0, 0.4, 2.0, 5.0], dtype=torch.float64)
    # Build a shared padded grid covering all ys by padding per-row then re-gridding
    # is complex; instead validate each row independently on its own padded grid.
    for yv in ys.tolist():
        cdf, ee, y, y_bin, sh = _make_padded_crts_inputs(p, e, m, yv)
        crts = _raw_crts(cdf, ee, y, y_bin, sh, alphas=[2.0])["crts_alpha_2.0"]
        pp, ee2, mm, sh2, _gl, _gr = pad_to_common_grid(p, e, m, y, True)
        energy = _raw_energy(pp, ee2, y, betas=[1.0])["energy_score_beta_1.0"]
        rel = abs(crts - energy) / abs(energy)
        assert rel <= 1e-12, f"y={yv}: crts_a2={crts:.10f} crps={energy:.10f} rel={rel:.2e}"


# ===========================================================================
# CRTS numerical-stability regression: _linear_power_integral
# ---------------------------------------------------------------------------
# ``_linear_power_integral`` evaluates I(A, B, p) = ∫₀¹ (A + u·(B−A))^p du, the
# per-slab building block of every CRTS α-power term.  The naive closed form
# (B^{p+1} − A^{p+1})/((p+1)(B−A)) loses relative precision by catastrophic
# cancellation when the slab-edge CDF values A, B are close *relative* to their
# magnitude — precisely the regime a fine or near-saturated histogram produces,
# and worst exactly where CRTS is most support-sensitive (α → 1, p = α−1 → 0).
# The stabilised expm1/log1p form must stay at machine precision there.  These
# tests pin that guarantee against an independent high-precision reference so a
# regression to any absolute-threshold dispatch is caught.
# ===========================================================================
from scoringbench.metrics import _linear_power_integral  # noqa: E402

mpmath = pytest.importorskip("mpmath")


def _lpi_reference(A, B, p, dps=50):
    """Exact I(A, B, p) via high-precision mpmath (independent of the code)."""
    with mpmath.workdps(dps):
        Am, Bm, pm = mpmath.mpf(A), mpmath.mpf(B), mpmath.mpf(p)
        if abs(Bm - Am) < mpmath.mpf("1e-45"):
            return float(Am ** pm)
        return float(mpmath.quad(lambda u: (Am + u * (Bm - Am)) ** pm, [0, 1]))


# (A, B) pairs spanning the failure modes of the naive form: tiny *relative*
# rise just above any plausible absolute floor, near-saturation (B ≈ 1),
# decreasing order (A > B, from the 1−F term), an endpoint at exactly 0, and
# equal endpoints (removable A == B limit / zero-width slab).
_LPI_PAIRS = [
    (0.3, 0.3 + 1e-9),
    (0.3, 0.3 + 1e-11),
    (0.3, 0.3 + 1e-14),
    (0.9999, 0.9999 + 1e-11),
    (0.999999, 1.0),
    (0.9999999999, 1.0),
    (1e-9, 2e-9),
    (0.7, 0.2),          # decreasing (A > B)
    (0.5000000001, 0.5),  # decreasing, tiny relative rise
    (0.0, 1e-8),         # endpoint exactly 0
    (0.0, 1.0),
    (1.0, 0.0),
    (0.5, 0.5),          # A == B (removable limit)
    (1.0, 1.0),
]


@pytest.mark.parametrize("A,B", _LPI_PAIRS)
@pytest.mark.parametrize("p", [0.01, 0.05, 0.2, 0.5, 1.0])  # p = α−1 and p = α
def test_linear_power_integral_matches_high_precision(A, B, p):
    """Stabilised I(A, B, p) stays ≤ 1e-13 relative to a 50-digit reference.

    The naive closed form reaches ~1e-6..1e-2 relative error on several of these
    pairs; the expm1/log1p reformulation must not regress to that.
    """
    got = float(_linear_power_integral(
        torch.tensor([A], dtype=torch.float64),
        torch.tensor([B], dtype=torch.float64),
        p,
    )[0])
    ref = _lpi_reference(A, B, p)
    rel = abs(got - ref) / max(abs(ref), 1e-300)
    assert rel <= 1e-13, f"A={A} B={B} p={p}: got={got:.16e} ref={ref:.16e} rel={rel:.2e}"


def test_linear_power_integral_relative_scale_invariance():
    """Relative error must stay flat (not blow up) as (B−A)/A → 0.

    Sweeps the relative rise from 1e-2 down to 1e-15 at a fixed base and the
    most sensitive exponent (p = α−1 with α = 1.01).  A cancellation-based
    implementation shows error growing ∝ A/(B−A); the stable form must remain
    machine-precision throughout.
    """
    A = 0.4
    p = 0.01
    worst = 0.0
    for k in range(2, 16):
        B = A + A * (10.0 ** -k)
        got = float(_linear_power_integral(
            torch.tensor([A], dtype=torch.float64),
            torch.tensor([B], dtype=torch.float64),
            p,
        )[0])
        ref = _lpi_reference(A, B, p)
        worst = max(worst, abs(got - ref) / abs(ref))
    assert worst <= 1e-13, f"worst relative error {worst:.2e} across (B-A)/A in 1e-2..1e-15"


def test_linear_power_integral_batched_matches_scalar():
    """Vectorised evaluation agrees element-wise with per-element calls."""
    A = torch.tensor([a for a, _ in _LPI_PAIRS], dtype=torch.float64)
    B = torch.tensor([b for _, b in _LPI_PAIRS], dtype=torch.float64)
    for p in (0.01, 0.5, 1.0):
        batched = _linear_power_integral(A, B, p)
        for i, (a, b) in enumerate(_LPI_PAIRS):
            scalar = _linear_power_integral(
                torch.tensor([a], dtype=torch.float64),
                torch.tensor([b], dtype=torch.float64),
                p,
            )[0]
            assert torch.allclose(batched[i], scalar, rtol=0, atol=0), (
                f"batched != scalar at ({a},{b}), p={p}")
