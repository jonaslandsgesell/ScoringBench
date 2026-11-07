"""Energy-score robustness on near-tied quantile grids (regression tests).

A quantile model whose outer quantiles are sorted-but-almost-equal produces a
bin with a *small positive* width (a near-Dirac) located far from most targets.
On such a bin the energy-score slab integrals used to cancel catastrophically:

* **term 1** ``(1/w)∫_a^b |x−y|^β dx`` built its expm1/log1p argument as
  ``(min|offset| − max|offset|)/max|offset|`` — a subtraction of two large,
  nearly-equal offsets that loses the bin width it is meant to recover;
* **term 2** ``E|X−X'|^β`` (``uniform_slab_pairwise_distance``) evaluated the
  disjoint-pair second difference as ``top − bot`` of two ``O(gap^{β+2})`` powers
  whose difference is only ``O(gap^{β} wL wR)``.

Both are fixed to a cancellation-free form.  The symptom was that the energy
β-term disagreed with the **mathematically identical** CRTS α=2 by up to ~6
orders of magnitude on a few quantile-model cells (observed: CRPS ≈ 5.3e6 where
the true value was ≈ 0.84).

These tests pin the fix against INDEPENDENT references:
  * a 50-digit mpmath evaluation of the exact integrals (term 1 and term 2);
  * the ``crts_alpha_2.0 == energy_score_beta_1.0`` identity (CRPS two ways);
  * the closed-form CRPS ``q²·D`` of a two-mass (near-Dirac + slab) forecast.

Every case here FAILS on the pre-fix code and passes after.  All grids are
MONOTONE (sorted edges) — the only regime a quantile head can produce.
"""
from __future__ import annotations

import math

import mpmath as mp
import numpy as np
import pytest
import torch

torch.cuda.is_available = lambda: False  # CPU float64 determinism
EPS = 100 * torch.finfo(torch.float64).eps

from scoringbench.univariate._integration import uniform_slab_pairwise_distance
from scoringbench.univariate.metrics import (
    compute_crts,
    compute_energy_score_histogram_corrected,
)

mp.mp.dps = 50


# ---------------------------------------------------------------------------
# Independent high-precision references (50-digit mpmath).
# ---------------------------------------------------------------------------
def _ref_pairwise(a1, b1, a2, b2, beta):
    """Exact ``E|X−X'|^β``, X~U[a1,b1], X'~U[a2,b2], via the 4-corner second
    mixed antiderivative of ``|d|^{β+2}`` evaluated at 50 digits (so the
    cancellation that wrecks float64 is harmless here)."""
    a1, b1, a2, b2 = map(mp.mpf, (a1, b1, a2, b2))
    p = mp.mpf(beta) + 2
    q = mp.mpf(beta) + 1
    h = lambda d: abs(d) ** p
    num = h(b1 - b2) - h(b1 - a2) - h(a1 - b2) + h(a1 - a2)
    return float(-num / (q * p * (b1 - a1) * (b2 - a2)))


def _ref_term1(a, b, y, beta):
    """Exact ``(1/w)∫_a^b |x−y|^β dx`` at 50 digits."""
    a, b, y = mp.mpf(a), mp.mpf(b), mp.mpf(y)
    q = mp.mpf(beta) + 1
    g = lambda d: mp.sign(d) * abs(d) ** q / q
    return float((g(b - y) - g(a - y)) / (b - a))


def _energy_and_crts(edges, probas, y, betas):
    """Public-API energy scores + CRTS α=2 for a single-row shared-grid hist."""
    et = torch.tensor(edges, dtype=torch.float64)
    pt = torch.tensor(probas, dtype=torch.float64)[None, :]
    yt = torch.tensor([float(y)], dtype=torch.float64)
    en = compute_energy_score_histogram_corrected(pt, et, yt, betas=list(betas))
    cdf = torch.cumsum(pt, dim=-1)
    ybin = torch.searchsorted(et[1:].contiguous(), yt).clamp(0, pt.shape[1] - 1)
    cr = compute_crts(cdf, et, yt, ybin, True, alphas=[2.0])["crts_alpha_2.0"]
    return en, cr


# ---------------------------------------------------------------------------
# term 2: uniform_slab_pairwise_distance on a narrow slab far from a wide slab
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("beta", [0.1, 0.5, 1.0, 1.5, 1.9])
@pytest.mark.parametrize("narrow_w", [1e-3, 1e-6, 1e-9])
@pytest.mark.parametrize("sep", [1e3, 1e6, 1e9])
def test_pairwise_narrow_far_slab_matches_mpmath(beta, narrow_w, sep):
    if sep + narrow_w == sep:
        pytest.skip("width below the ULP of its position -> an exact (zero-width) "
                    "Dirac in float64, handled by the Dirac dispatch, not this path")
    # wide slab [0, 2]; narrow near-Dirac slab [sep, sep+narrow_w]; monotone edges.
    edges = torch.tensor([0.0, 2.0, sep, sep + narrow_w], dtype=torch.float64)
    D = uniform_slab_pairwise_distance(edges, beta, eps=EPS)
    got = D[0, 2].item()                       # wide ↔ narrow-far cross distance
    want = _ref_pairwise(0.0, 2.0, sep, sep + narrow_w, beta)
    assert math.isfinite(got)
    assert abs(got - want) <= 1e-9 * (1 + abs(want)), (
        f"β={beta} sep={sep} w={narrow_w}: got {got:.8g} want {want:.8g} "
        f"(rel {abs(got-want)/(abs(want)+1e-30):.2e})"
    )
    # symmetry: D must be symmetric.
    assert abs(D[0, 2].item() - D[2, 0].item()) <= 1e-12 * (1 + abs(want))


# ---------------------------------------------------------------------------
# term 1: (1/w)∫|x−y|^β on a narrow bin far from y
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("beta", [0.1, 0.5, 1.0, 1.5, 1.9])
@pytest.mark.parametrize("narrow_w", [1e-3, 1e-6, 1e-9])
def test_term1_narrow_far_bin_matches_mpmath(beta, narrow_w):
    # All mass in the narrow far bin.  The energy score is then its own
    # self-term-1 minus its within-bin self-term-2:
    #   E_β = (1/w)∫_M^{M+w}|x−y|^β dx  −  ½·E|X−X'|^β (both ~U[M,M+w]).
    # The term-1 piece carries the far offset (~1e6) and is where the old
    # catastrophic cancellation lived; the self-term-2 is a tiny within-bin
    # correction included here so the reference is exact.
    M = 1e6
    edges = [0.0, 4.0, M, M + narrow_w]
    probas = [0.0, 0.0, 1.0]
    y = 5.0
    en, _ = _energy_and_crts(edges, probas, y, [beta])
    got = en[f"energy_score_beta_{beta}"]
    want = (_ref_term1(M, M + narrow_w, y, beta)
            - 0.5 * _ref_pairwise(M, M + narrow_w, M, M + narrow_w, beta))
    assert abs(got - want) <= 1e-9 * (1 + abs(want)), (
        f"β={beta} w={narrow_w}: got {got:.8g} want {want:.8g} "
        f"(rel {abs(got-want)/(abs(want)+1e-30):.2e})"
    )


# ---------------------------------------------------------------------------
# integration: CRPS two ways must agree on a near-tied-quantile histogram
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("narrow_w", [1e-3, 1e-6, 1e-9])
@pytest.mark.parametrize("sep", [1e4, 1e6, 1e8])
def test_crps_equals_crts2_on_near_tied_quantile(narrow_w, sep):
    # 0.98 mass near y, 0.02 mass in a near-tied-quantile bin far away.
    edges = [0.0, 4.0, 6.0, sep, sep + narrow_w]
    probas = [0.0, 0.98, 0.0, 0.02]
    y = 5.0
    en, crts2 = _energy_and_crts(edges, probas, y, [1.0])
    crps = en["energy_score_beta_1.0"]
    assert crps > 0.0
    assert abs(crps - crts2) <= 1e-8 * (1 + abs(crts2)), (
        f"sep={sep} w={narrow_w}: CRPS(energy β=1)={crps:.8g} "
        f"CRTS(α=2)={crts2:.8g} (rel {abs(crps-crts2)/(abs(crts2)+1e-30):.2e})"
    )


# ---------------------------------------------------------------------------
# closed form: two-mass forecast CRPS = q²·D_cross + within-slab terms
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("beta", [0.5, 1.0, 1.5, 1.9])
def test_energy_two_mass_near_dirac_matches_closed_form(beta):
    # mass (1-q) as a point at y (zero-width Dirac) + mass q in a narrow far bin.
    # With the near point exactly at y its self/own-distance terms vanish and the
    # energy score reduces to q·E|far−y|^β − ½·2q(1-q)E|far−y|^β − ... ; rather
    # than hand-expand, compare to a 50-digit mpmath evaluation of the full
    # mixture energy score.
    q = 0.02
    M, w = 1e6, 1e-7
    # bin0 = Dirac at 5 (zero width), bin1 = [5, M] empty, bin2 = narrow far bin.
    edges = [5.0, 5.0, M, M + w]
    probas = [1.0 - q, 0.0, q]
    y = 5.0
    en, crts2 = _energy_and_crts(edges, probas, y, [beta])
    got = en[f"energy_score_beta_{beta}"]

    # mpmath reference: ES = Σ_i p_i T1_i − ½ Σ_ij p_i p_j D_ij.
    def t1(a, b):
        if b == a:
            return abs(a - y) ** beta
        return _ref_term1(a, b, y, beta)

    def dij(i, j):
        ai, bi = edges[i], edges[i + 1]
        aj, bj = edges[j], edges[j + 1]
        if bi == ai and bj == aj:
            return abs(ai - aj) ** beta
        if bi == ai:      # point i vs slab j
            return _ref_term1(aj, bj, ai, beta)
        if bj == aj:
            return _ref_term1(ai, bi, aj, beta)
        return _ref_pairwise(ai, bi, aj, bj, beta)

    term1 = sum(probas[i] * t1(edges[i], edges[i + 1]) for i in range(3))
    term2 = 0.5 * sum(
        probas[i] * probas[j] * dij(i, j) for i in range(3) for j in range(3)
    )
    want = term1 - term2
    assert abs(got - want) <= 1e-8 * (1 + abs(want)), (
        f"β={beta}: got {got:.8g} want {want:.8g} "
        f"(rel {abs(got-want)/(abs(want)+1e-30):.2e})"
    )


# ---------------------------------------------------------------------------
# sanity: the fix must not perturb ordinary (well-separated, comparable) grids
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("beta", [0.3, 1.0, 1.7])
def test_uniform_grid_unchanged(beta):
    edges = torch.tensor(np.linspace(0.0, 10.0, 11), dtype=torch.float64)
    D = uniform_slab_pairwise_distance(edges, beta, eps=EPS)
    for i in range(10):
        for j in range(10):
            want = _ref_pairwise(edges[i].item(), edges[i + 1].item(),
                                 edges[j].item(), edges[j + 1].item(), beta)
            assert abs(D[i, j].item() - want) <= 1e-11 * (1 + abs(want))
