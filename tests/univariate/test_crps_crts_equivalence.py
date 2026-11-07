"""CRPS consistency: a fast, implementation-agnostic correctness guard.

Two facts that any correct implementation of the univariate rules MUST satisfy,
checked here against references that share no code with the production path:

  1. ``crts_alpha_2.0 == energy_score_beta_1.0`` — the α=2 Tsallis (Brier)
     integrand ``s_2(F, q) = (F − q)²`` integrates to exactly the CRPS, which the
     energy score at β=1 also equals (Gneiting & Raftery 2007).  The two code
     paths compute it in completely different ways (CDF slab integral vs. the
     ``E|X−y| − ½E|X−X'|`` pairwise form), so their agreement is a strong,
     general correctness signal.

  2. both equal an INDEPENDENT CRPS: a dense-grid quadrature of
     ``∫(F(t) − 1{t≥y})² dt`` (and, for one case, a 50-digit mpmath evaluation).

The forecasts include the regime that used to break the energy path — a
near-tied-quantile bin (tiny positive width) far from the target — at several
scales, driven through the public ``compute_scoring_rules`` entry point as well
as the individual rule functions.  These are deliberately NOT tied to the
internal ``uniform_slab_pairwise_distance`` geometry, so they keep guarding
correctness if that implementation is rewritten.

Fast: the rule-level cases run in milliseconds; one end-to-end and one mpmath
case add ~a few seconds total.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

torch.cuda.is_available = lambda: False  # CPU float64 determinism

from scoringbench.univariate.metrics import (
    compute_crts,
    compute_energy_score_histogram_corrected,
    compute_scoring_rules,
)
from scoringbench.univariate.wrappers import DistributionPrediction


# ---------------------------------------------------------------------------
# Independent references (share no code with the rules under test)
# ---------------------------------------------------------------------------
def _dense_crps(edges, probas, y, n=2_000_001, pad=10.0):
    """CRPS = ∫(F(t) − 1{t≥y})² dt of the piecewise-linear-CDF histogram, by
    dense trapezoid quadrature on a grid independent of the production code."""
    edges = np.asarray(edges, float)
    cum = np.concatenate([[0.0], np.cumsum(np.asarray(probas, float).ravel())])
    cum = cum / cum[-1]
    lo = min(edges[0], float(y)) - pad
    hi = max(edges[-1], float(y)) + pad
    t = np.linspace(lo, hi, n)
    F = np.interp(t, edges, cum, left=0.0, right=1.0)
    _trap = getattr(np, 'trapezoid', None) or np.trapz
    return float(_trap(((F - (t >= float(y))) ** 2), t))


def _energy_b1(edges, probas, y):
    et = torch.tensor(edges, dtype=torch.float64)
    pt = torch.tensor(probas, dtype=torch.float64)[None, :]
    yt = torch.tensor([float(y)], dtype=torch.float64)
    return compute_energy_score_histogram_corrected(pt, et, yt, betas=[1.0])["energy_score_beta_1.0"]


def _crts_a2(edges, probas, y):
    et = torch.tensor(edges, dtype=torch.float64)
    pt = torch.tensor(probas, dtype=torch.float64)[None, :]
    yt = torch.tensor([float(y)], dtype=torch.float64)
    cdf = torch.cumsum(pt, dim=-1)
    ybin = torch.searchsorted(et[1:].contiguous(), yt).clamp(0, pt.shape[1] - 1)
    return compute_crts(cdf, et, yt, ybin, True, alphas=[2.0])["crts_alpha_2.0"]


def _dist(edges, probas):
    """DistributionPrediction with the native grid kept in float64."""
    edges = np.asarray(edges, dtype=np.float64)
    mids = 0.5 * (edges[:-1] + edges[1:])
    probas = np.asarray(probas, dtype=np.float64).reshape(1, -1)
    mean = np.array([float((mids * probas.ravel()).sum())], dtype=np.float64)
    return DistributionPrediction(
        probas=probas, bin_edges=edges, bin_midpoints=mids, mean=mean,
        train_range=(float(edges.min()), float(edges.max())),
    )


# (name, edges, probas, y) — ordinary forecasts + near-tied-quantile far bins.
_CASES = {
    "uniform":              ([0.0, 10.0], [1.0], 5.0),
    "uniform_fine":         (list(np.linspace(0, 10, 11)), [0.1] * 10, 5.0),
    "skewed":               ([0.0, 1.0, 2.0, 5.0, 20.0], [0.5, 0.2, 0.2, 0.1], 3.0),
    # near-tied outer quantile (tiny positive width) far from y, several scales:
    "near_tie_1e2":         ([0.0, 4.0, 6.0, 1e2, 1e2 + 1e-6], [0.0, 0.98, 0.0, 0.02], 5.0),
    "near_tie_1e4":         ([0.0, 4.0, 6.0, 1e4, 1e4 + 1e-4], [0.0, 0.98, 0.0, 0.02], 5.0),
    "near_tie_1e6":         ([0.0, 4.0, 6.0, 1e6, 1e6 + 1e-3], [0.0, 0.98, 0.0, 0.02], 5.0),
    "two_near_ties":        ([0.0, 1.0, 1.0 + 1e-7, 500.0, 500.0 + 1e-5],
                             [0.3, 0.0, 0.5, 0.2], 0.5),
}


@pytest.mark.parametrize("name", list(_CASES))
def test_crts_alpha2_equals_energy_beta1(name):
    """The core invariant, via the two rule functions directly (fast)."""
    edges, probas, y = _CASES[name]
    e = _energy_b1(edges, probas, y)
    c = _crts_a2(edges, probas, y)
    assert abs(e - c) <= 1e-9 * (1 + abs(c)), (
        f"{name}: energy β=1 = {e:.10g} vs CRTS α=2 = {c:.10g} "
        f"(rel {abs(e-c)/(abs(c)+1e-30):.2e})"
    )


@pytest.mark.parametrize("name", list(_CASES))
def test_crps_matches_independent_dense_quadrature(name):
    """Both paths equal an independent dense-grid CRPS."""
    edges, probas, y = _CASES[name]
    ref = _dense_crps(edges, probas, y)
    assert abs(_energy_b1(edges, probas, y) - ref) <= 1e-3 * (1 + ref)
    assert abs(_crts_a2(edges, probas, y) - ref) <= 1e-3 * (1 + ref)


def test_crps_equals_crts_through_public_api():
    """End-to-end through ``compute_scoring_rules`` (the production entry point)
    on a near-tied-quantile forecast: the returned crps and crts_alpha_2.0 agree."""
    d = _dist([0.0, 4.0, 6.0, 1e6, 1e6 + 1e-3], [0.0, 0.98, 0.0, 0.02])
    r = compute_scoring_rules(d, np.array([5.0]))
    assert abs(r["crps"] - r["crts_alpha_2.0"]) <= 1e-8 * (1 + abs(r["crts_alpha_2.0"]))
    assert r["crps"] > 0.0


def test_equivalence_against_mpmath():
    """50-digit mpmath cross-check that CRPS (energy form) == CRPS (Brier
    integral form) == both code paths, on a near-tied-quantile forecast."""
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 50
    edges = [0.0, 4.0, 6.0, 1e6, 1e6 + 1e-6]
    probas = [0.0, 0.98, 0.0, 0.02]
    y = 5.0

    E = [mp.mpf(x) for x in edges]
    P = [mp.mpf(x) for x in probas]
    F = [mp.mpf(0)]
    for pk in P:
        F.append(F[-1] + pk)
    F = [f / F[-1] for f in F]
    yy = mp.mpf(y)

    # Brier integral form, exact per slab (split at y), 50 digits.
    brier = mp.mpf(0)
    for k in range(len(P)):
        a, b, Fl, Fr = E[k], E[k + 1], F[k], F[k + 1]
        if b <= a:
            continue
        pts = [a, b] if not (a < yy < b) else [a, yy, b]
        for i in range(len(pts) - 1):
            lo, hi = pts[i], pts[i + 1]
            ind = mp.mpf(1) if lo >= yy else mp.mpf(0)
            Fv = lambda t: Fl + (Fr - Fl) * (t - a) / (b - a)
            brier += mp.quad(lambda t: (Fv(t) - ind) ** 2, [lo, hi])
    brier = float(brier)

    assert abs(_energy_b1(edges, probas, y) - brier) <= 1e-6 * (1 + abs(brier))
    assert abs(_crts_a2(edges, probas, y) - brier) <= 1e-6 * (1 + abs(brier))
