"""Tests for the shared numerical-integration primitives.

``scoringbench._integration`` provides exact closed-form integration primitives
for the scoring rules:

1. ``uniform_axis_integral`` — composite trapezoid / midpoint quadrature over a
   uniform 1-D grid (used by the quantile-weighted CRPS).
2. ``uniform_slab_pairwise_distance`` — exact E|X - X'|^β for uniform slabs
   (used by the energy score / CRPS term-2).

These tests pin both against closed-form integrals and ``scipy.integrate``
so that any future change to the shared helpers is caught here rather than
surfacing indirectly through a scoring-rule regression.
"""

import numpy as np
import pytest
import torch

from scoringbench._integration import (  # noqa: E402
    uniform_axis_integral,
    uniform_slab_pairwise_distance,
)

_EPS64 = 100 * torch.finfo(torch.float64).eps


# ============================================================================
# uniform_axis_integral
# ============================================================================

def test_uniform_trapezoid_linear_is_exact():
    """Composite trapezoid integrates a linear function exactly."""
    a, b, m = 0.0, 4.0, 5
    xs = torch.linspace(a, b, m, dtype=torch.float64)
    vals = 2.0 * xs + 1.0                                   # ∫ = [x^2 + x]_0^4 = 20
    got = uniform_axis_integral(vals, a, b, rule="trapezoid")
    assert abs(got.item() - 20.0) < 1e-12


def test_uniform_trapezoid_matches_numpy_trapezoid():
    a, b, m = -1.0, 2.0, 51
    xs = torch.linspace(a, b, m, dtype=torch.float64)
    vals = torch.exp(xs)
    got = uniform_axis_integral(vals, a, b, rule="trapezoid").item()
    ref = float(np.trapezoid(vals.numpy(), xs.numpy()))
    assert abs(got - ref) < 1e-12


def test_uniform_midpoint_matches_historical_wcrps_spacing():
    """midpoint rule reproduces the legacy d_alpha = 1/(m+1) weighting.

    The quantile-weighted CRPS historically sampled ``linspace(0.01,0.99,99)``
    and multiplied the sum by ``1/(99+1)``.  With a=0, b=1 and 99 interior
    samples the midpoint rule must reproduce exactly that: h = 1/100.
    """
    m = 99
    vals = torch.linspace(0.01, 0.99, m, dtype=torch.float64) ** 2  # arbitrary integrand
    got = uniform_axis_integral(vals, 0.0, 1.0, rule="midpoint").item()
    d_alpha = 1.0 / (m + 1)
    legacy = float(vals.sum().item()) * d_alpha
    assert abs(got - legacy) < 1e-14


def test_uniform_integral_dim_argument():
    """Integration reduces the requested axis only."""
    a, b, m = 0.0, 1.0, 5
    # values shape (3, m); each row = c_r * ones -> ∫ = c_r * 1
    c = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)[:, None]
    vals = c.expand(3, m).contiguous()
    got = uniform_axis_integral(vals, a, b, rule="trapezoid", dim=-1)
    assert torch.allclose(got, c.squeeze(-1), atol=1e-13)


def test_uniform_trapezoid_needs_two_points():
    with pytest.raises(ValueError):
        uniform_axis_integral(torch.tensor([1.0], dtype=torch.float64), 0.0, 1.0,
                              rule="trapezoid")


def test_uniform_unknown_rule_raises():
    with pytest.raises(ValueError):
        uniform_axis_integral(torch.zeros(4, dtype=torch.float64), 0.0, 1.0,
                              rule="simpson")


# ============================================================================
# uniform_slab_pairwise_distance — exact E|X - X'|^beta for uniform slabs
# ============================================================================

def _dblquad_pair(a1, b1, a2, b2, beta):
    """Reference E|X-X'|^beta by direct 2-D quadrature over the two slabs."""
    from scipy.integrate import dblquad

    val, _ = dblquad(
        lambda xp, x: abs(x - xp) ** beta,
        a1, b1, lambda _x: a2, lambda _x: b2,
    )
    return val / ((b1 - a1) * (b2 - a2))


@pytest.mark.parametrize("beta", [0.3, 0.5, 1.0, 1.5, 2.0])
@pytest.mark.parametrize(
    "a1,b1,a2,b2",
    [
        (0.0, 1.0, 1.0, 2.0),    # adjacent (share an edge)
        (0.0, 1.0, 5.0, 6.0),    # far disjoint
        (0.0, 2.0, 3.0, 4.0),    # disjoint, unequal widths
        (0.0, 3.0, 1.0, 2.0),    # nested / overlapping
        (0.0, 1.0, 0.0, 1.0),    # identical (self / diagonal)
    ],
)
def test_slab_pairwise_matches_dblquad(beta, a1, b1, a2, b2):
    """The closed form equals a brute-force 2-D quadrature for every pair type.

    One formula must be exact for disjoint, overlapping and diagonal pairs — the
    correctness guarantee that lets it replace both the midpoint off-diagonal and
    the separately special-cased diagonal in the energy-score term-2.

    The primitive works on a *contiguous* edge vector, so the two slabs
    ``[a1,b1]`` and ``[a2,b2]`` are embedded as the first and last bins of a
    3-bin grid; the throwaway middle bin only affects entries we never read.
    (For adjacent, tiling slabs the two bins are adjacent and read as ``(0,1)``.)
    """
    ref = _dblquad_pair(a1, b1, a2, b2, beta)
    if abs(b1 - a2) < 1e-12:  # slabs tile -> genuine adjacent 2-bin grid
        edges = torch.tensor([a1, b1, b2], dtype=torch.float64)
        D = uniform_slab_pairwise_distance(edges, beta, eps=_EPS64)
        got = D[0, 1].item()
    else:                     # embed as bins 0 and 2 of a 3-bin grid
        edges = torch.tensor([a1, b1, a2, b2], dtype=torch.float64)
        D = uniform_slab_pairwise_distance(edges, beta, eps=_EPS64)
        got = D[0, 2].item()
    # Tolerance is set by dblquad's own accuracy (the closed form is exact); the
    # identical-slab / small-β integrand |x-x'|^β is mildly singular on the
    # diagonal, so the reference itself carries ~1e-8 quadrature error there.
    assert abs(got - ref) < 1e-7


@pytest.mark.parametrize("beta", [0.3, 0.5, 1.0, 1.5, 2.0])
@pytest.mark.parametrize("width", [0.5, 1.0, 2.5])
def test_slab_pairwise_diagonal_is_self_distance(beta, width):
    """The diagonal reproduces the exact uniform self-distance 2w^β/((β+1)(β+2))."""
    edges = torch.tensor([0.0, width, 2 * width], dtype=torch.float64)
    D = uniform_slab_pairwise_distance(edges, beta, eps=_EPS64)
    expect = (2.0 * width ** beta) / ((beta + 1.0) * (beta + 2.0))
    assert abs(D[0, 0].item() - expect) < 1e-12


@pytest.mark.parametrize("beta", [0.3, 0.5, 1.0, 1.5, 2.0])
def test_slab_pairwise_symmetric(beta):
    """The matrix is symmetric: E|X_i - X'_j|^β = E|X_j - X'_i|^β."""
    edges = torch.tensor([-2.0, -0.5, 0.7, 1.3, 3.0], dtype=torch.float64)
    D = uniform_slab_pairwise_distance(edges, beta, eps=_EPS64)
    assert torch.allclose(D, D.T, atol=1e-13)


def test_slab_pairwise_beta1_collapses_to_midpoint():
    """At β = 1, |x - x'| is linear so the off-diagonal collapses to |m_i - m_j|.

    This is exactly why the CRPS (β = 1 energy score) is unaffected by the fix.
    """
    beta = 1.0
    edges = torch.tensor([0.0, 1.0, 2.5, 4.0], dtype=torch.float64)
    mids = 0.5 * (edges[:-1] + edges[1:])
    D = uniform_slab_pairwise_distance(edges, beta, eps=_EPS64)
    off = ~torch.eye(3, dtype=torch.bool)
    midpoint = (mids[:, None] - mids[None, :]).abs()
    assert torch.allclose(D[off], midpoint[off], atol=1e-13)


def test_slab_pairwise_batched_matches_shared():
    """The batched (per-sample edges) path matches the shared single-grid path."""
    edges = torch.tensor([-1.0, 0.2, 1.1, 2.0, 3.5], dtype=torch.float64)
    beta = 1.5
    shared = uniform_slab_pairwise_distance(edges, beta, eps=_EPS64)
    batched = uniform_slab_pairwise_distance(
        edges[None, :].repeat(4, 1), beta, eps=_EPS64
    )
    assert batched.shape == (4, 4, 4)
    for k in range(4):
        assert torch.allclose(batched[k], shared, atol=1e-13)
