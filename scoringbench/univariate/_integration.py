"""Shared numerical-integration primitives for the scoring rules.

Almost every scoring rule in :mod:`scoringbench.metrics` reduces to an *exact*
closed-form per-bin sum (CRTS via :func:`_linear_power_integral`, the energy
score / CRPS via :func:`uniform_slab_pairwise_distance` and its term-1 sibling,
all evaluated cancellation-free with ``expm1``/``log1p``) — no quadrature error
anywhere.  The one genuinely *numerical* quadrature that remains is:

* **Uniform-grid quadrature over a 1-D axis** — used by the quantile-weighted
  CRPS, which integrates a pinball loss over evenly spaced probability levels
  ``α ∈ (0, 1)``.

Centralising it here (rather than each caller rolling its own ``d_alpha``
spacing) makes the spacing explicit and lets a single test suite pin the
primitive against SciPy.

All helpers are torch-native and dtype/device-preserving so they compose with
the ``force_precision(torch.float64)`` decorator used throughout ``metrics``.
"""

from __future__ import annotations

import torch

__all__ = [
    "uniform_axis_integral",
    "uniform_slab_pairwise_distance",
]


def uniform_axis_integral(
    values: torch.Tensor,
    a: float,
    b: float,
    *,
    rule: str = "trapezoid",
    dim: int = -1,
) -> torch.Tensor:
    """Integrate samples taken on a *uniform* grid spanning ``[a, b]``.

    ``values`` holds the integrand evaluated at ``len`` equally spaced points
    along ``dim``.  The grid convention matches how the sample points were
    generated:

    * ``rule="trapezoid"`` — points are the closed grid
      ``linspace(a, b, m)`` and the composite trapezoidal rule is applied
      (spacing ``h = (b − a) / (m − 1)``).  This is the accurate default.
    * ``rule="midpoint"`` — points are ``m`` interior samples and each carries
      equal weight ``h = (b − a) / (m + 1)`` (a left/mid Riemann sum with the
      two open end intervals accounted for).  Kept only for exact
      back-compatibility with the historical quantile-weighted CRPS, which
      sampled ``linspace(0.01, 0.99, 99)`` and multiplied the sum by
      ``1 / (99 + 1)``.

    Parameters
    ----------
    values : torch.Tensor
        Integrand samples; integrated along ``dim``.
    a, b : float
        Integration limits.
    rule : {"trapezoid", "midpoint"}
        Quadrature rule (see above).
    dim : int
        Axis along which to integrate.

    Returns
    -------
    torch.Tensor
        The integral, with ``dim`` reduced.
    """
    m = values.shape[dim]
    if rule == "trapezoid":
        if m < 2:
            raise ValueError("trapezoid rule needs >= 2 samples.")
        h = (b - a) / (m - 1)
        w = torch.full((m,), h, dtype=values.dtype, device=values.device)
        w[0] *= 0.5
        w[-1] *= 0.5
    elif rule == "midpoint":
        if m < 1:
            raise ValueError("midpoint rule needs >= 1 sample.")
        h = (b - a) / (m + 1)
        w = torch.full((m,), h, dtype=values.dtype, device=values.device)
    else:
        raise ValueError(f"unknown rule {rule!r}; use 'trapezoid' or 'midpoint'.")

    shape = [1] * values.ndim
    shape[dim] = m
    return (values * w.reshape(shape)).sum(dim=dim)


def uniform_slab_pairwise_distance(
    edges: torch.Tensor,
    beta: float,
    *,
    eps: float,
) -> torch.Tensor:
    """Exact ``E|X − X'|^β`` for two independent draws from *uniform slabs*.

    Each histogram bin ``k`` is modelled as a uniform density on
    ``[edges_k, edges_{k+1}]`` (the same piecewise-constant / linear-CDF law the
    energy-score term-1 uses).  For bins ``i`` and ``j`` with ``X ~ U[a_i, b_i]``
    and ``X' ~ U[a_j, b_j]`` the pairwise expectation

        E|X − X'|^β = ∫∫ |x − x'|^β dx dx' / (w_i w_j)

    has the closed form (second mixed antiderivative of ``|d|^{β+2}``)::

        E|X − X'|^β = − [ h(b_i−b_j) − h(b_i−a_j) − h(a_i−b_j) + h(a_i−a_j) ]
                      / [ (β+1)(β+2) w_i w_j ],     h(d) = |d|^{β+2}.

    This one formula is exact for **every** pair of *positive-width* slabs —
    disjoint, overlapping, and the diagonal ``i = j`` (where it reduces to
    ``2 w^β / ((β+1)(β+2))``), so it replaces both the midpoint off-diagonal
    approximation ``|m_i − m_j|^β`` (exact only at ``β = 1``) and the separately
    special-cased diagonal correction.  At ``β = 1`` the off-diagonal collapses
    to ``|m_i − m_j|`` exactly (``|x − x'|`` is linear), which is why CRPS is
    unaffected.

    **Zero-width (Dirac) bins.**  A bin whose width is ``≤ eps`` is a point mass
    at its location ``m = (a + b) / 2 = a = b``, not a uniform slab.  The general
    4-corner form has ``w_i w_j`` in the denominator and becomes ``0 / 0`` (and
    numerically unstable) in that limit, so those quadrants are dispatched to the
    exact degenerate closed forms:

    * **point ↔ slab** (``w_i → 0``, slab ``[a_j, b_j]``)::

          E|m_i − X'_j|^β = [ g(b_j − m_i) − g(a_j − m_i) ] / w_j,
          g(d) = sign(d) |d|^{β+1} / (β + 1),

      i.e. the exact ``(1/w_j) ∫_{a_j}^{b_j} |m_i − x'|^β dx'`` — **not** the
      midpoint approximation ``|m_i − m_j|^β`` (which is off by ``O(w_j^2)`` for
      ``β ≠ 1``).

    * **point ↔ point** (both widths ``→ 0``): ``|m_i − m_j|^β`` exactly.

    Parameters
    ----------
    edges : torch.Tensor
        Bin edges, shape ``(n_bins + 1,)`` (shared grid) or
        ``(n_samples, n_bins + 1)`` (per-sample grid).  For a Dirac bin both of
        its edges coincide (``a = b = m``); pass the *true* grid edges so the
        point ↔ slab dispatch fires on the correct, collapsed interval.
    beta : float
        Energy-score exponent ``β ∈ (0, 2]``.
    eps : float
        Width threshold below which a bin is treated as a Dirac point mass.

    Returns
    -------
    torch.Tensor
        Pairwise distance matrix ``D`` with shape ``(n_bins, n_bins)`` (shared)
        or ``(n_samples, n_bins, n_bins)`` (per-sample), where
        ``D[..., i, j] = E|X_i − X'_j|^β``.
    """
    p = beta + 2.0
    q = beta + 1.0
    a = edges[..., :-1]                      # (..., n_bins) left edges
    b = edges[..., 1:]                       # (..., n_bins) right edges
    w = b - a                                # (..., n_bins) true widths (≥ 0)
    m = 0.5 * (a + b)                         # (..., n_bins) bin locations

    # Broadcast bin i (rows) against bin j (cols): add a trailing/leading axis.
    ai, aj = a[..., :, None], a[..., None, :]
    bi, bj = b[..., :, None], b[..., None, :]
    mi, mj = m[..., :, None], m[..., None, :]
    # Floored widths keep the slab-slab division finite; the Dirac quadrants
    # (where the true width is ~0) are overwritten below, so the floor never
    # leaks into the returned value.
    wi = w[..., :, None].clamp(min=eps)
    wj = w[..., None, :].clamp(min=eps)

    def h(d: torch.Tensor) -> torch.Tensor:
        return d.abs().pow(p)

    # --- slab ↔ slab, KINK regime (touching / overlapping / self, i.e. the two
    # slabs share a point or interior).  Here the two widths are comparable, the
    # 4-corner second-mixed-antiderivative form is well-conditioned, and it is
    # the ONLY form that resolves the |x − x'|^β kink sitting on/inside the box.
    num = h(bi - bj) - h(bi - aj) - h(ai - bj) + h(ai - aj)
    closed = -num / (q * p * wi * wj)

    # --- slab ↔ slab, DISJOINT regime.  When the slabs do not overlap, the naive
    # 4-corner form catastrophically cancels for disparate widths (h(b_i−b_j) etc.
    # are large and nearly equal, then divided by the tiny product w_i w_j).  The
    # integral is still EXACT and elementary — no quadrature is required — the
    # only issue is *how* the alternating sum of powers is evaluated.  On a
    # disjoint pair all four corner offsets share one sign (the whole of one slab
    # lies to one side of the other), so ``num`` is a second finite difference of
    # ``h(d) = |d|^p`` over positive arguments.  Group it into two SAME-SIGN power
    # differences and evaluate each cancellation-free via expm1/log1p::
    #
    #     num = h(b_i−b_j) − h(b_i−a_j) − h(a_i−b_j) + h(a_i−a_j)
    #         = [h(a_i−a_j) − h(a_i−b_j)] − [h(b_i−a_j) − h(b_i−b_j)]
    #         =        bot                −        top,
    #
    # where each bracket is ``hi^p − lo^p`` with ``hi ≥ lo ≥ 0`` (the outer edge
    # minus the near/far inner edge).  Writing the two "far" offsets as ``hi`` and
    # the matching "near" offsets as ``lo`` and using
    # ``hi^p − lo^p = hi^p · (1 − (lo/hi)^p) = hi^p · (−expm1(p·log1p((lo−hi)/hi)))``
    # keeps every subtraction between quantities of the same sign and comparable
    # magnitude, so the tiny ``w_i w_j`` denominator no longer amplifies round-off.
    # This is the SAME expm1/log1p trick the kink branch avoids needing (it has an
    # interior kink so the corners already differ in sign); here it replaces the
    # former Gauss–Legendre fallback entirely — exact, ~10 (..., i, j) temporaries
    # instead of ~a dozen (..., i, j, GL_NODES) ones, and no quadrature error.
    def _pow_diff(hi: torch.Tensor, lo: torch.Tensor) -> torch.Tensor:
        """``hi^p − lo^p`` for ``hi ≥ lo ≥ 0``, cancellation-free (0 if hi = 0)."""
        hi_pos = hi > 0
        safe_hi = torch.where(hi_pos, hi, torch.ones_like(hi))
        ratio = (lo - hi) / safe_hi          # ∈ [−1, 0]
        val = safe_hi.pow(p) * (-torch.expm1(p * torch.log1p(ratio)))
        return torch.where(hi_pos, val, torch.zeros_like(val))

    # Orient each pair so slab i is on the LEFT and slab j on the RIGHT (all four
    # offsets aj/bj − ai/bi ≥ 0).  |x − x'|^β is symmetric under the swap, so this
    # only relabels the corners; it does not change the value.
    left_lo = torch.minimum(ai, aj)
    left_hi = torch.minimum(bi, bj)
    right_lo = torch.maximum(ai, aj)
    right_hi = torch.maximum(bi, bj)
    # Corner offsets (right edge − left edge), all ≥ 0 on a disjoint pair:
    #   d_ll = right_lo − left_lo,  d_lh = right_lo − left_hi,
    #   d_hl = right_hi − left_lo,  d_hh = right_hi − left_hi.
    # num = h(d_hh) − h(d_hl) − h(d_lh) + h(d_ll)
    #     = [h(d_hh) − h(d_hl)] − [h(d_lh) − h(d_ll)]   (each bracket same-sign):
    #   h(d_hh) − h(d_hl):  hi = right_hi − left_lo,  lo = right_hi − left_hi  (≥0)
    #   h(d_lh) − h(d_ll):  hi = right_lo − left_lo,  lo = right_lo − left_hi  (≥0)
    top = _pow_diff(right_hi - left_lo, right_hi - left_hi)
    bot = _pow_diff(right_lo - left_lo, right_lo - left_hi)
    num_disjoint = top - bot
    disjoint_val = num_disjoint / (q * p * wi * wj)

    # Exact-float disjoint predicate (cancellation-free): the slabs are disjoint
    # iff one lies entirely to the left of the other.  Equality (touching) is
    # routed to the closed form, which is exact at a shared corner.
    disjoint = (bi <= aj) | (bj <= ai)
    D = torch.where(disjoint, disjoint_val, closed)

    # --- Dirac dispatch (only if any bin is (near-)zero width). ---
    zero = w <= eps                          # (..., n_bins)
    if bool(zero.any()):
        zi = zero[..., :, None]              # bin i is a point mass
        zj = zero[..., None, :]              # bin j is a point mass

        # Stable (1/w) ∫ |c − x|^β dx over a slab, given the two signed offsets
        # of the point c from the slab edges.  When the point is far outside a
        # narrow slab both offsets are large and nearly equal, so the naive
        # g(hi) − g(lo) = (|hi|^q − |lo|^q)/q cancels; evaluate it via expm1/log1p
        # exactly as in the disjoint slab-slab branch.  When the point lies inside
        # the slab the two offsets have opposite signs (no cancellation) and we
        # fall back to the direct closed form.
        def point_slab(c: torch.Tensor, lo_e: torch.Tensor,
                       hi_e: torch.Tensor, width: torch.Tensor) -> torch.Tensor:
            d_lo = c - lo_e
            d_hi = c - hi_e
            same_side = (d_lo * d_hi) > 0    # point outside [lo_e, hi_e]
            a_lo = d_lo.abs()
            a_hi = d_hi.abs()
            big = torch.maximum(a_lo, a_hi)
            small = torch.minimum(a_lo, a_hi)
            big_pos = big > 0
            safe_big = torch.where(big_pos, big, torch.ones_like(big))
            r = (small - big) / safe_big
            outside = safe_big.pow(q) * (-torch.expm1(q * torch.log1p(r)))
            outside = torch.where(big_pos, outside, torch.zeros_like(outside)) / q
            # Point inside: ∫ = (|d_lo|^q + |d_hi|^q)/q (kink at c splits the slab).
            inside = (a_lo.pow(q) + a_hi.pow(q)) / q
            integral = torch.where(same_side, outside, inside)
            return integral / width

        # point i (at m_i) ↔ slab j: (1/w_j) ∫_{a_j}^{b_j} |m_i − x'|^β dx'.
        ps_i_j = point_slab(mi, aj, bj, wj)
        # slab i ↔ point j (at m_j): (1/w_i) ∫_{a_i}^{b_i} |x − m_j|^β dx.
        ps_j_i = point_slab(mj, ai, bi, wi)
        # point ↔ point.
        pp = (mi - mj).abs().pow(beta)

        # Precedence: both points → pp; exactly one point → the matching
        # point↔slab form; neither → the slab-slab value already in D.
        both = zi & zj
        D = torch.where(zi & ~zj, ps_i_j, D)
        D = torch.where(zj & ~zi, ps_j_i, D)
        D = torch.where(both, pp, D)

    return D
