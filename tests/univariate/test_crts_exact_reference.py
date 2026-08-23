"""High-precision CRTS regression tests against an exact rational reference.

The reworked ``compute_crts`` (commit 597b2cb) integrates the divergence-form
binary α-Tsallis score over slab-linear CDF segments.  Two numerical hazards
live in the small-α / saturated-tail regime:

1. **CDF residual.** ``cdf = cumsum(probas)`` accumulates O(machine-eps)
   round-off, so its final value is ``1 - r`` with ``r ~ ±1e-16``.  On a
   saturated tail slab (F ≡ 1, indicator q = 0) the point term charges
   ``(1-F)^{α-1} = r^{α-1}``, and ``r^{α-1} ≈ 0.71`` at α = 1.01 — mischarging
   the out-of-support gap term by up to ~65 % of the whole score.
2. **Quadrature error.** A fixed Gauss–Legendre rule samples F strictly inside
   each slab and cannot resolve the diverging endpoint derivative as α → 1, so
   it mischarges near-saturated slabs by O(1e-3) even at 16 nodes.

``compute_crts`` guards (1) by renormalising the CDF so ``F(last edge) = 1``
exactly, and (2) by integrating each slab in *closed form* (exact for every
admissible α).  These tests pin both fixes to a mpmath rational ground truth at
a tolerance far tighter than the historical GL-16 error, with a special focus
on α = 1.01 targets that fall *outside* the predictive support.

The ground truth holds the pmf as exact ``fractions.Fraction`` values so the
cumulative sums — and hence F on a saturated slab — are exactly right, and
integrates each slab-linear segment with 60-digit mpmath quadrature.
"""

from fractions import Fraction as Fr

import pytest
import torch

from scoringbench.univariate.metrics import compute_crts, pad_to_common_grid

mp = pytest.importorskip("mpmath")

torch.cuda.is_available = lambda: False  # force CPU

# Bypass the @force_precision(float64) decorator so tests control the dtype.
_raw_crts = compute_crts.__wrapped__

ALPHAS = [1.01, 1.05, 1.2, 1.5, 2.0]


# ---------------------------------------------------------------------------
# Exact rational + mpmath ground truth
# ---------------------------------------------------------------------------
def _crts_exact_rational(p_fr, edges_fr, y_fr, alpha, dps=60):
    """CRTS with F built from EXACT rational cumulative sums.

    ``p_fr``      : list[Fraction]  pmf on the (already padded) grid
    ``edges_fr``  : list[Fraction]  n_bins + 1 edges
    ``y_fr``      : Fraction        target
    """
    mp.mp.dps = dps
    a = mp.mpf(alpha)
    offset = 1 / a - 1 / (a - 1)

    def s(F, q):
        it = (F ** a + (1 - F) ** a) / a
        pt = (F ** (a - 1) if q else (1 - F) ** (a - 1)) / (a - 1)
        return it - pt - offset

    cum = [Fr(0)]
    for v in p_fr:
        cum.append(cum[-1] + v)

    def _mpf(fr):
        return mp.mpf(fr.numerator) / fr.denominator

    total = mp.mpf(0)
    Y = _mpf(y_fr)
    for k in range(len(p_fr)):
        A, B = _mpf(edges_fr[k]), _mpf(edges_fr[k + 1])
        if B <= A:
            continue
        Fa, Fb = _mpf(cum[k]), _mpf(cum[k + 1])

        def Fof(t):
            f = Fa + (Fb - Fa) * (t - A) / (B - A)
            return min(max(f, mp.mpf(0)), mp.mpf(1))

        segs = [(A, Y), (Y, B)] if A < Y < B else [(A, B)]
        for L, R in segs:
            if R <= L:
                continue
            q = 1 if L >= Y else 0
            total += mp.quad(lambda t: s(Fof(t), q), [L, R])
    return float(total)


def _gaussian_pmf(n_bins, lo=-4.0, hi=4.0):
    """Float64 Gaussian-ish pmf row (as production sees it) plus its exact
    rational normalisation (as the ground truth sees it)."""
    e = torch.linspace(lo, hi, n_bins + 1, dtype=torch.float64)
    m = 0.5 * (e[:-1] + e[1:])
    p = torch.exp(-0.5 * m ** 2)
    p = (p / p.sum()).reshape(1, -1)
    p_fr = [Fr(float(v)) for v in p[0]]
    S = sum(p_fr)
    p_fr = [v / S for v in p_fr]
    assert sum(p_fr) == 1
    return p, e, m, p_fr


def _run_production(p, e, m, yv):
    """Pad exactly as the production entry point does, then call compute_crts."""
    y = torch.tensor([float(yv)], dtype=torch.float64)
    pp, ee, mm, sh, _gl, _gr = pad_to_common_grid(p, e, m, y, True)
    cdf = torch.cumsum(pp, dim=-1)
    y_bin = torch.searchsorted(ee[1:].contiguous(), y).clamp(0, pp.shape[1] - 1)
    out = _raw_crts(cdf, ee, y, y_bin, sh, alphas=ALPHAS)
    return out, pp, ee, sh, _gl


def _exact_on_padded(p_fr, e, m, pp, ee, gl, n_bins, yv):
    """Exact rational reference on the SAME padded grid production used."""
    npad = pp.shape[1] - n_bins
    pad_left = 1 if gl.item() > 0 else 0
    pp_fr = [Fr(0)] * pad_left + list(p_fr) + [Fr(0)] * (npad - pad_left)
    ee_fr = [Fr(float(v)) for v in ee]
    return {al: _crts_exact_rational(pp_fr, ee_fr, Fr(float(yv)), al) for al in ALPHAS}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
# Tolerances: the closed-form slab integral + CDF renormalisation should match
# the exact reference to O(1e-9) or better even at α = 1.01.  We keep a small
# margin for the float64 CDF the production code actually receives.
_RTOL = {1.01: 5e-9, 1.05: 5e-10, 1.2: 5e-11, 1.5: 5e-12, 2.0: 5e-13}


@pytest.mark.parametrize("n_bins,yv", [
    (50, 6.0), (200, 6.0), (500, 6.0), (1000, 6.0),   # right tail (residual-prone)
    (200, -6.0), (500, -6.0),                          # left tail (structural zero)
    (100, 0.37), (200, 3.5),                           # inside support
])
def test_crts_matches_exact_reference(n_bins, yv):
    """compute_crts must match the exact rational reference to high precision,
    including y far outside the predictive support at α = 1.01."""
    p, e, m, p_fr = _gaussian_pmf(n_bins)
    out, pp, ee, sh, gl = _run_production(p, e, m, yv)
    exact = _exact_on_padded(p_fr, e, m, pp, ee, gl, n_bins, yv)

    for al in ALPHAS:
        got = out[f"crts_alpha_{al}"]
        ref = exact[al]
        rel = abs(got - ref) / abs(ref)
        assert rel <= _RTOL[al], (
            f"n_bins={n_bins} y={yv} alpha={al}: "
            f"got={got:.10f} ref={ref:.10f} rel={rel:.2e} > {_RTOL[al]:.0e}"
        )


def test_crts_alpha_1_01_saturated_tail_is_exact():
    """Focused regression for the residual-driven α=1.01 defect.

    With y = +6 outside a 500-bin N(0,1) grid the pre-fix score dropped to
    ~75.8 against an exact value of ~217.8 (a 65 % error) purely because the
    cumsum residual r ~ 1e-15 gave (1-F)^{0.01} ~ 0.71 on the saturated tail.
    """
    n_bins, yv = 500, 6.0
    p, e, m, p_fr = _gaussian_pmf(n_bins)
    out, pp, ee, sh, gl = _run_production(p, e, m, yv)
    exact = _exact_on_padded(p_fr, e, m, pp, ee, gl, n_bins, yv)

    got = out["crts_alpha_1.01"]
    ref = exact[1.01]
    rel = abs(got - ref) / abs(ref)
    assert rel <= 5e-9, f"got={got:.10f} ref={ref:.10f} rel={rel:.2e}"


def test_crts_reflection_invariance_alpha_1_01():
    """CRTS(F, y) must equal CRTS(reflected F, -y) at α = 1.01.

    The residual bug broke this by ~65 % because only the right tail carries a
    cumsum residual (F_left[:,0] is a structural zero, so the left tail is
    unaffected).  After the fix both sides must agree to numerical precision.
    """
    n_bins = 500
    for yv in (6.0, 3.0):
        # forecast for +y
        p, e, m, _ = _gaussian_pmf(n_bins)
        out_pos, *_ = _run_production(p, e, m, yv)
        # reflected forecast for -y: reverse the pmf and negate/reverse edges
        p_ref = torch.flip(p, dims=[1]).contiguous()
        e_ref = torch.flip(-e, dims=[0]).contiguous()
        m_ref = 0.5 * (e_ref[:-1] + e_ref[1:])
        out_neg, *_ = _run_production(p_ref, e_ref, m_ref, -yv)
        for al in ALPHAS:
            a = out_pos[f"crts_alpha_{al}"]
            b = out_neg[f"crts_alpha_{al}"]
            rel = abs(a - b) / abs(a)
            assert rel <= 1e-10, f"y={yv} alpha={al}: {a:.10f} vs {b:.10f} rel={rel:.2e}"


def test_crts_alpha_2_equals_crps():
    """crts_alpha_2.0 must equal the β=1 energy score (CRPS) to precision.

    This is the exactness anchor: at α=2 the integrand is quadratic in F and the
    closed-form slab integral coincides with the energy-score discretisation.
    """
    from scoringbench.univariate.metrics import compute_energy_score_histogram_corrected
    _raw_energy = compute_energy_score_histogram_corrected.__wrapped__

    for n_bins, yv in [(50, 0.37), (200, 0.37), (200, 6.0), (500, -6.0)]:
        p, e, m, _ = _gaussian_pmf(n_bins)
        y = torch.tensor([float(yv)], dtype=torch.float64)
        pp, ee, mm, sh, _gl, _gr = pad_to_common_grid(p, e, m, y, True)
        cdf = torch.cumsum(pp, dim=-1)
        y_bin = torch.searchsorted(ee[1:].contiguous(), y).clamp(0, pp.shape[1] - 1)
        crts = _raw_crts(cdf, ee, y, y_bin, sh, alphas=[2.0])["crts_alpha_2.0"]
        energy = _raw_energy(pp, ee, y, betas=[1.0])["energy_score_beta_1.0"]
        rel = abs(crts - energy) / abs(energy)
        assert rel <= 1e-12, (
            f"n_bins={n_bins} y={yv}: crts_a2={crts:.10f} "
            f"energy_b1={energy:.10f} rel={rel:.2e}"
        )
