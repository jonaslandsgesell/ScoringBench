"""Regression + safety tests for the *off-diagonal* slab↔slab energy maths.

Historically the pairwise term ``E|X − X'|^β`` of the energy score used a
**midpoint approximation** for the off-diagonal entries of the pairwise
distance matrix::

    D_old[i, j] = |m_i − m_j|^β        (m = bin midpoint)

That approximation is *exact only for β = 1* (CRPS): for a linear kernel the
expected distance between two independent uniforms equals the distance between
their means.  For every other β it is wrong by O(w²) — it ignores the spread of
each slab — and the error grows with the bin width.  Because the whole project
was validated with CRPS, the bug hid in plain sight.

The current implementation replaces that with the **exact 4-corner closed
form** (see :func:`uniform_slab_pairwise_distance`)::

    E|X_i − X_j|^β = − [ h(b_i−b_j) − h(b_i−a_j) − h(a_i−b_j) + h(a_i−a_j) ]
                       / ((β+1)(β+2) · w_i · w_j),      h(d) = |d|^{β+2}

which integrates the kernel over both slabs analytically and is exact for
disjoint, overlapping and diagonal slab pairs alike.

This module contains two families of tests:

* ``test_offdiagonal_beats_old_midpoint`` — a genuine **regression** test.  On
  *wide* slabs the old midpoint value is off by 0.05 … 2.7 for β ≠ 1 while the
  exact form matches a scipy double-integral to ≈ 1e-9.  The test asserts both
  facts, so it would have *failed with the old implementation and passes now*.

* ``test_four_corner_no_breakdown`` (+ helpers) — **safety** tests that pin the
  4-corner form in near-degenerate "danger zones" (touching edges, extreme
  width ratios, sub-micron bins, large coordinate offsets that stress
  subtractive cancellation).  They assert the result stays finite,
  non-negative and — where a reference integral is trustworthy — accurate, so
  the closed form is safe to rely on in production.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy import integrate

from scoringbench._integration import uniform_slab_pairwise_distance

# Deterministic CPU float64 throughout.
torch.cuda.is_available = lambda: False
EPS = 100 * torch.finfo(torch.float64).eps

_BETAS = [0.5, 1.0, 1.5, 2.0]


# --------------------------------------------------------------------------- #
# References
# --------------------------------------------------------------------------- #
def _quad_offdiag(ai: float, bi: float, aj: float, bj: float, beta: float) -> float:
    """Exact off-diagonal reference E|X_i − X_j|^β via 2-D quadrature.

    ``X_i ~ U[ai, bi]``, ``X_j ~ U[aj, bj]`` independent.  Shares *no* code with
    the closed form under test.
    """
    kernel = lambda x, xp: abs(x - xp) ** beta  # noqa: E731
    val, _ = integrate.dblquad(kernel, aj, bj, lambda _: ai, lambda _: bi)
    return val / ((bi - ai) * (bj - aj))


def _old_midpoint_offdiag(mi: float, mj: float, beta: float) -> float:
    """The *old* (buggy) off-diagonal approximation ``|m_i − m_j|^β``.

    Reproduced here purely so the regression test can demonstrate that it
    disagrees with the truth for β ≠ 1.
    """
    return abs(mi - mj) ** beta


# --------------------------------------------------------------------------- #
# Regression test: exact 4-corner beats the old midpoint approximation
# --------------------------------------------------------------------------- #
# Deliberately WIDE bins (width 4) so the midpoint error O(w²) is large and the
# separation between "old" and "new" is unambiguous.
_WIDE_GRID = torch.tensor([0.0, 4.0, 8.0, 12.0], dtype=torch.float64)


@pytest.mark.parametrize("beta", _BETAS)
def test_offdiagonal_beats_old_midpoint(beta: float) -> None:
    """The exact form matches quadrature; the old midpoint form does not (β≠1).

    This is the regression that would have failed with the previous
    implementation: for every β ≠ 1 the midpoint approximation is wrong by a
    macroscopic amount on these wide bins, whereas the current 4-corner closed
    form reproduces the true double integral to ~1e-9.
    """
    grid = _WIDE_GRID
    a = grid[:-1].numpy()
    b = grid[1:].numpy()
    m = 0.5 * (a + b)
    n = len(a)

    D = uniform_slab_pairwise_distance(grid, beta, eps=EPS).numpy()

    max_exact_err = 0.0
    max_old_err = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue  # off-diagonal only
            ref = _quad_offdiag(a[i], b[i], a[j], b[j], beta)
            max_exact_err = max(max_exact_err, abs(D[i, j] - ref))
            old = _old_midpoint_offdiag(m[i], m[j], beta)
            max_old_err = max(max_old_err, abs(old - ref))

    # (1) The current closed form is essentially exact.
    assert max_exact_err < 1e-6, (
        f"exact 4-corner off-diagonal deviates from quadrature by "
        f"{max_exact_err:.2e} at beta={beta}"
    )

    if abs(beta - 1.0) < 1e-12:
        # β = 1 is the one case where the midpoint approximation is *also*
        # exact — this is precisely why the bug went unnoticed for CRPS.
        assert max_old_err < 1e-6
    else:
        # For every other β the old approximation is macroscopically wrong,
        # far outside the tolerance the exact form satisfies.  This is the
        # crux of the regression.
        assert max_old_err > 1e-2, (
            f"old midpoint approximation unexpectedly close (err={max_old_err:.2e}) "
            f"at beta={beta}; regression test would not distinguish old vs new"
        )
        # And the new form must be orders of magnitude better than the old one.
        assert max_exact_err < max_old_err / 1e3


# --------------------------------------------------------------------------- #
# Safety tests: the 4-corner form must not break down in degenerate regimes
# --------------------------------------------------------------------------- #
# Each grid probes a different numerical hazard for the subtractive 4-corner
# formula.  ``trust_quad`` marks whether a scipy reference is reliable there
# (quadrature itself struggles with extreme width ratios / near-zero widths).
_SAFETY_GRIDS = {
    # b_i == a_j: two slabs sharing an edge (the |d|^{β+2} corner terms include
    # a zero argument).
    "adjacent_touching": (torch.tensor([0.0, 1.0, 2.0]), True),
    # 12 orders of magnitude width ratio between neighbouring slabs.
    "extreme_width_ratio": (torch.tensor([0.0, 1e-6, 1e6]), False),
    # Five sub-micron slabs: tiny widths in every denominator.
    "sub_micron_bins": (torch.linspace(0.0, 1e-4, 6), True),
    # Width just above the Dirac threshold — must stay on the slab path.
    "near_dirac_slab": (torch.tensor([0.0, 5e-11, 1.0]), False),
    # Large coordinate offset with small widths: the corner differences are
    # tiny numbers formed from huge |d|^{β+2} values → worst case for
    # catastrophic cancellation (guarded by the internal float64 upcast).
    "large_offset_small_slab": (
        torch.tensor([1e5, 1e5 + 1e-3, 1e5 + 2e-3]),
        True,
    ),
}


@pytest.mark.parametrize("name", list(_SAFETY_GRIDS))
@pytest.mark.parametrize("beta", _BETAS)
def test_four_corner_no_breakdown(name: str, beta: float) -> None:
    """The 4-corner form is finite, non-negative and (where checkable) accurate.

    Guarantees the closed form is safe to use in production even in the
    near-degenerate regimes that maximise subtractive cancellation.
    """
    grid, trust_quad = _SAFETY_GRIDS[name]
    grid = grid.to(torch.float64)
    a = grid[:-1].numpy()
    b = grid[1:].numpy()
    n = len(a)

    D = uniform_slab_pairwise_distance(grid, beta, eps=EPS).numpy()

    # No NaNs / infs anywhere, including the diagonal.
    assert np.isfinite(D).all(), f"non-finite entry in D for {name}, beta={beta}"

    # A distance^β expectation is non-negative (allow a tiny negative slack for
    # float round-off).
    assert (D >= -1e-9).all(), f"negative pairwise distance for {name}, beta={beta}"

    # Symmetry: E|X_i − X_j|^β == E|X_j − X_i|^β.
    assert np.allclose(D, D.T, atol=1e-9, rtol=1e-9), (
        f"pairwise distance not symmetric for {name}, beta={beta}"
    )

    if not trust_quad:
        # scipy quadrature is unreliable at these extremes; finiteness /
        # non-negativity / symmetry are the meaningful guarantees.
        return

    # Where quadrature is trustworthy, the off-diagonals must match it.
    max_rel = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            wi = b[i] - a[i]
            wj = b[j] - a[j]
            if wi <= EPS or wj <= EPS:
                continue  # Dirac path, not the 4-corner slab formula
            ref = _quad_offdiag(a[i], b[i], a[j], b[j], beta)
            denom = max(abs(ref), 1.0)
            max_rel = max(max_rel, abs(D[i, j] - ref) / denom)

    assert max_rel < 1e-6, (
        f"4-corner off-diagonal deviates from quadrature by rel {max_rel:.2e} "
        f"for {name}, beta={beta}"
    )


def test_four_corner_matches_dirac_limit() -> None:
    """As a slab width → 0 the 4-corner form must approach the point↔slab limit.

    This checks the closed form does not *silently* break down near the Dirac
    threshold: a very thin (but still > eps) slab must give essentially the same
    answer as an exact point mass at its midpoint.
    """
    beta = 1.5
    mid = 3.0
    # Thin slab around ``mid`` (still on the slab path) vs an exact Dirac atom.
    half = 1e-4
    slab_grid = torch.tensor([0.0, mid - half, mid + half, 6.0], dtype=torch.float64)
    dirac_grid = torch.tensor([0.0, mid, mid, 6.0], dtype=torch.float64)

    D_slab = uniform_slab_pairwise_distance(slab_grid, beta, eps=EPS).numpy()
    D_dirac = uniform_slab_pairwise_distance(dirac_grid, beta, eps=EPS).numpy()

    assert np.isfinite(D_slab).all() and np.isfinite(D_dirac).all()
    # The off-diagonal coupling of the thin slab (index 1) to the neighbouring
    # slabs must match the Dirac-atom coupling to O(half).
    assert np.allclose(D_slab[1, [0, 2]], D_dirac[1, [0, 2]], atol=1e-3), (
        f"thin slab off-diagonals {D_slab[1, [0, 2]]} disagree with Dirac limit "
        f"{D_dirac[1, [0, 2]]}"
    )
