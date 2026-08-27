"""CDF grid-building utilities for ScoringBench wrappers.

Every prediction reduces to ONE lossless intermediate: monotone CDF nodes
``(cdf_support_point, cdf_levels)`` -- abscissae where the model places its cumulative mass.
From those nodes two grids are derived directly, with no round-trip through an
intermediate histogram:

* the NATIVE grid (:func:`cdf_nodes_to_native_PMF_grid` / :func:`interpolate_cdf_to_grid_with_equally_sized_bins`) --
  scored by the grid-robust rules; atoms preserved (quantiles) or binned to a
  bounded uniform grid (samples, whose eCDF can have as many nodes as draws and
  would make the O(n_bins^2) native energy score explode), and
* the RESAMPLED grid (:func:`resample_cdf_nodes_to_support_outer_hull_y_train_set_y_instance_prediction_grid`) -- the same
  nodes PCHIP-resampled onto ``num_equally_sized_bins`` uniform bins over the
  per-instance grow-only support; scored by the density rules.

The three source builders (:func:`quantiles_to_cdf_nodes`,
:func:`samples_to_cdf_nodes`) turn a wrapper's raw output into those nodes; a
grid-native model (TabPFN bar distribution) skips nodes entirely and its native
PMF grid is kept verbatim for both views.

The raw nodes are used AS-IS: no ``C = 0/1`` anchoring and no invented tail.  A
quantile grid reaches only ``[alpha_0, alpha_{K-1}]`` and an eCDF only
``[1/2n, 1 - 1/2n]``; that residual tail mass is folded into the outermost bins
by the per-bin renormalization every grid builder already applies
(:func:`_normalize` for the native quantile grid, the in-place normalize in
:func:`_interpolate_cdf_to_grid_with_equally_sized_bins` for the binned/resampled
grids).  With ``K >= 100`` levels
or ``n >= 300`` draws the residual is <=1% on every rule, so the wrappers simply
supply enough levels/draws rather than fabricate a tail.

Public surface
--------------
``DistributionPredictionView``
    The view on the probabilistic prediction (either native PMF grid for scoring rules for M1, M2 or resampled smoothed grid for L1,L2 scoring rules).
``quantiles_to_cdf_nodes``, ``samples_to_cdf_nodes``
    Source -> monotone CDF nodes ``(cdf_support_point, cdf_levels)``.
``cdf_nodes_to_native_PMF_grid``, ``interpolate_cdf_to_grid_with_equally_sized_bins``
    CDF nodes -> native PMF grid (nodes-as-edges, or bounded uniform binning).
``resample_cdf_nodes_to_support_outer_hull_y_train_set_y_instance_prediction_grid``
    CDF nodes -> the density-rule (resampled) grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator


# ===========================================================================
# Shared low-level helpers (monotone CDF evaluation)
# ===========================================================================

def _evaluate_monotone_cdf(x_row, c_row, edges_row):
    """Evaluate a monotone CDF at ``edges_row`` with a shape-preserving cubic.

    ``x_row`` is non-decreasing abscissae and ``c_row`` the matching
    non-decreasing CDF values (both 1-D); ``edges_row`` are the query points.

    Why PCHIP rather than linear interpolation
    ------------------------------------------
    A linear CDF makes the implied density piecewise *constant*, so a sharp mode
    is flattened onto the bin holding it -- the density at the target is then
    wrong even though the per-bin mass is right.  A monotone cubic (PCHIP) fits a
    C1 curve through the same nodes without overshooting, so ``diff(C)`` follows
    the true density's shape between the predicted levels while staying
    non-negative.  Masses are the exact CDF increment at the bin edges,
    ``C(e_{k+1}) - C(e_k)``, not a sampled derivative.
    ``tests/univariate/test_interpolation_scheme_selection.py`` compares this
    against linear and derivative-sampling variants and selects it.

    Ties (atoms) are collapsed, then deliberately smoothed
    ------------------------------------------------------
    PCHIP needs *strictly* increasing abscissae, but a CDF read off quantiles /
    histogram edges may repeat an ``x`` where an atom lives.  Each run of tied
    abscissae is collapsed to its single coordinate carrying the run's *last* CDF
    value -- ``np.interp``'s "last matching node wins" rule -- so the cumulative
    mass AT the atom is exact, and a target on the atom is scored against the bin
    ``searchsorted(edges[1:], y)`` credits it to.

    The jump does NOT survive as a jump.  PCHIP then interpolates smoothly through
    the collapsed nodes, spreading the atom's mass over the interval below it: at
    200 bins only ~0.5% of it reaches the single bin holding the atom.  That is
    the point -- this resampler feeds the resampled view, whose job is to
    neutralise atoms so the width-dividing density rules see a finite ``p/w``
    (see the container comment in ``base.py``).  Predictions that must keep their
    jumps are scored on the native view, which is never resampled.

    With two or fewer distinct abscissae there is no cubic to fit and the
    evaluation falls back to ``np.interp``.

    Extreme coordinates
    -------------------
    PCHIP evaluates its cubic in the local variable ``t = x - x_i``, so ``t**3``
    overflows to ``inf`` for spacings past ``~1e103`` and ``inf * 0.0 = nan``,
    which the clip below silently turns into a flat PMF.  Rescaling ``x`` by a
    power of two fixes it exactly -- the interpolant is equivariant in ``x`` and
    ``np.ldexp`` only edits the exponent, so the result is bit-identical.

    That normalises the overall magnitude but not the *ratio* of adjacent
    spacings, and the node slopes ``dC / h_k`` still overflow when one gap
    underflows against another (``[-1e-9, 0, 1e300]``).  No rescaling helps there,
    so we fall back to a linear CDF: still monotone and mass-exact, forfeiting
    only the density shape that
    ``tests/univariate/test_interpolation_scheme_selection.py`` selects PCHIP for.
    """
    # Collapse tied abscissae (atoms) to their single coordinate.  np.unique
    # returns the sorted unique abscissae; the scatter below maps each input node
    # onto its slot, later duplicates overwriting earlier ones so each collapsed
    # node carries the LAST input CDF value at that abscissa -- an atom's whole
    # jump is kept, and a query on the atom reads its post-jump cumulative mass.
    xs = np.unique(x_row)
    order = np.searchsorted(xs, x_row)
    cs = np.zeros(xs.shape, dtype=np.float64)
    cs[order] = c_row                    # later duplicates overwrite -> last wins
    # EXCEPT the leftmost node: it must carry the CDF's PRE-jump value there (the
    # first input node), else an atom sitting ON the left abscissa makes last-wins
    # raise the floor and the clip below silently discards the mass below it.
    cs[0] = c_row[0]
    cs = np.maximum.accumulate(cs)       # guard monotonicity after the scatter

    if xs.size == 1:
        # A pure point mass (Dirac): every abscissa collapsed to one coordinate,
        # so the whole jump lives in c_row's span rather than in cs (one value).
        # np.interp would return a flat CDF, differencing to zero mass and a
        # uniform-over-the-whole-support fallback; instead evaluate the true step
        # (c_row's pre-jump value strictly below the atom, its post-jump value at
        # or above) so the mass lands entirely in the single bin whose right edge
        # first reaches the atom.
        return np.where(edges_row < xs[0], c_row[0], c_row[-1])
    if xs.size < 3:
        return np.interp(edges_row, xs, cs)

    # Exponent-only rescaling, so the significands (and the cubic) are unchanged.
    scale = np.max(np.abs(xs))
    shift = np.frexp(scale)[1] if (np.isfinite(scale) and scale > 0.0) else 0
    xs_s = np.ldexp(xs, -shift)
    edges_s = np.ldexp(edges_row, -shift)

    # The shift can round two adjacent abscissae together; PCHIP needs them strict.
    if np.all(np.diff(xs_s) > 0.0):
        try:
            # Raise rather than let an inf/nan reach the clip as a plausible PMF.
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                c = PchipInterpolator(xs_s, cs, extrapolate=True)(edges_s)
            if not np.all(np.isfinite(c)):
                raise FloatingPointError("non-finite PCHIP evaluation")
        except (FloatingPointError, ValueError):
            c = np.interp(edges_s, xs_s, cs)
    else:
        c = np.interp(edges_s, xs_s, cs)

    # PCHIP is monotone on its nodes, but extrapolation past the ends can leave
    # [0, 1]; clip so the differenced masses stay a valid PMF.
    return np.clip(c, cs[0], cs[-1])




def _uniform_grid(lo, hi, n_bins):
    """``n_bins + 1`` equal-width edges spanning ``[lo, hi]`` per row.

    ``lo`` / ``hi`` are ``(rows, 1)`` support ends; the returned edges are
    ``(rows, n_bins + 1)``.  The single place a uniform grid is laid out: the
    endpoints are pinned exactly (the affine map need not reproduce them bit for
    bit) and a running max removes any negative width left by round-off.
    """
    edges = lo + (hi - lo) * np.linspace(0.0, 1.0, n_bins + 1)[None, :]
    edges[:, 0] = lo[:, 0]
    edges[:, -1] = hi[:, 0]
    np.maximum.accumulate(edges, axis=-1, out=edges)
    return edges


def _interpolate_cdf_to_grid_with_equally_sized_bins(x, c, lo, hi, n_bins):
    """Bin monotone CDF nodes onto ``n_bins`` equal-width bins over ``[lo, hi]``.

    The ONE binning primitive.  Both native paths that need a uniform grid (the
    sample histogram) and the resampled (grow-only) grid call it: each row's CDF
    is evaluated at the uniform edges with the shape-preserving interpolant
    (:func:`_evaluate_monotone_cdf`) and forward-differenced into a normalized
    PMF.  Because ``C`` is monotone the masses are the exact non-negative
    increments ``C(e_{k+1}) - C(e_k)``.

    ``x`` / ``c`` are ``(rows, m)`` non-decreasing nodes (``x`` a shared 1-D row
    is broadcast); ``lo`` / ``hi`` are ``(rows,)`` per-row support ends.  Returns
    ``(edges, probas)`` with edges ``(rows, n_bins + 1)`` and probas
    ``(rows, n_bins)``.
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    c = np.atleast_2d(np.asarray(c, dtype=np.float64))
    rows = max(x.shape[0], c.shape[0])
    x_shared = x.shape[0] == 1
    edges = _uniform_grid(np.asarray(lo, float)[:, None], np.asarray(hi, float)[:, None], n_bins)
    probas = np.empty((rows, n_bins), dtype=np.float64)
    for i in range(rows):
        cc = _evaluate_monotone_cdf(x[0] if x_shared else x[i], c[i], edges[i])
        m = np.maximum(np.diff(cc), 0.0)
        s = m.sum()
        probas[i] = m / s if s > 0.0 else 1.0 / n_bins
    return edges, probas


# ===========================================================================
# Source -> monotone CDF nodes (the one lossless intermediate)
# ===========================================================================

def quantiles_to_cdf_nodes(q, alphas):
    """Read a per-sample quantile matrix as CDF nodes -- used verbatim, no anchor.

    The quantile function ``alpha -> q(alpha)`` IS the CDF: ``q`` (``(rows, K)``)
    are the abscissae, ``alphas`` (``(K,)``, one shared row) the cumulative mass.
    The nodes are returned AS-IS -- ``C`` stops at ``[alpha_0, alpha_{K-1}]``, and
    the small ``~1/K`` tail beyond the outermost levels is folded into the
    outermost bins by the downstream per-bin renormalization rather than an
    invented tail (safe for ``K >= 100``; see the module docstring).  Tied
    quantiles stay coincident, so a model's atoms survive as repeated abscissae.

    Returns ``(cdf_support_point, cdf_levels)`` -- each ``(rows, K)``.
    """
    xx = np.atleast_2d(np.asarray(q, dtype=np.float64))
    cc = np.broadcast_to(np.asarray(alphas, dtype=np.float64), xx.shape)
    return xx, cc


def samples_to_cdf_nodes(row):
    """Empirical CDF nodes of one row of draws; strictly increasing abscissae.

    The unique sorted draws become nodes with mid-rank (Hazen) CDF values
    ``(cum - count/2) / n`` -- strictly in ``(0, 1)``, so the CDF is monotone and
    (crucially) has NO tied abscissae, so binning it can never produce a
    zero-width bin.  A single distinct value is a point mass; it is duplicated so
    there are two abscissae to bin between.  The Hazen values are returned AS-IS
    (``C`` never reaches ``0``/``1``); the ``~1/2n`` tail beyond the extreme draws
    is folded into the outermost bins by the downstream renormalization (safe for
    ``n >= 300``; see the module docstring).

    Returns ``(cdf_support_point, cdf_levels)`` -- each ``(1, u)`` for ``u`` distinct draws.
    """
    u, counts = np.unique(np.asarray(row, dtype=np.float64), return_counts=True)
    if u.shape[0] == 1:
        u = np.repeat(u, 2)
        counts = np.array([1, 1])
    c = (np.cumsum(counts) - 0.5 * counts) / counts.sum()
    return u[None, :], c[None, :]


# ===========================================================================
# CDF nodes -> native PMF grid (grid-robust rules)
# ===========================================================================

def _normalize(probas):
    """Non-negative masses summing to 1 per row (uniform fallback for a dead row)."""
    p = np.maximum(probas, 0.0)
    s = p.sum(axis=-1, keepdims=True)
    return np.where(s > 0.0, p / np.where(s > 0.0, s, 1.0), 1.0 / p.shape[-1])


def cdf_nodes_to_native_PMF_grid(cdf_support_point, cdf_levels):
    """Use the CDF nodes DIRECTLY as bin edges -- no resample, atoms kept.

    The native PMF grid for quantile-based wrappers: the nodes ARE the edges and the
    masses are ``_normalize(diff(c))``.  The nodes are unanchored, so ``diff(c)``
    covers only the interior span ``[alpha_0, alpha_{K-1}]``; ``_normalize`` then
    spreads that unit of mass over the bins (interior renormalization -- the
    residual tail is folded in, not fabricated).  No regularity is imposed, so
    tied nodes stay coincident and a model's atoms survive as zero-width Dirac
    bins that the grid-robust rules score exactly.

    Returns ``(bin_edges, probas)``: edges ``(rows, m)``, probas ``(rows, m - 1)``.
    """
    edges = np.atleast_2d(np.asarray(cdf_support_point, dtype=np.float64))
    c = np.atleast_2d(np.asarray(cdf_levels, dtype=np.float64))
    return edges, _normalize(np.diff(c, axis=-1))


def interpolate_cdf_to_grid_with_equally_sized_bins(cdf_support_point, cdf_levels, n_bins):
    """Bin CDF nodes onto ``n_bins`` uniform bins over their own hull.

    The native PMF grid for sample-based wrappers: an eCDF can have as many nodes as
    draws, and the native energy score is O(n_bins^2), so the nodes are binned to
    a bounded uniform grid rather than used verbatim.  The nodes are strictly
    increasing (:func:`samples_to_cdf_nodes`), so every bin has positive width.

    Returns ``(bin_edges, probas)`` -- edges ``(rows, n_bins + 1)``.
    """
    x = np.atleast_2d(np.asarray(cdf_support_point, dtype=np.float64))
    return _interpolate_cdf_to_grid_with_equally_sized_bins(x, cdf_levels, x[:, 0], x[:, -1], n_bins)


# ===========================================================================
# CDF nodes / native PMF grid -> resampled grid (density rules)
# ===========================================================================

def resample_cdf_nodes_to_support_outer_hull_y_train_set_y_instance_prediction_grid(cdf_support_point, cdf_levels, y_lo, y_hi, n_bins):
    """Resample CDF nodes onto a GROW-ONLY grid: union(train range, this row's hull).

    The train range ``[y_lo, y_hi]`` is a FLOOR, never a ceiling: each row ``i``
    is resampled onto ``n_bins`` equal bins spanning
    ``[min(y_lo, x_i[0]), max(y_hi, x_i[-1])]`` -- the train range widened to the
    row's own node hull.  The grid always covers the row's mass (nothing
    truncated) and always covers the train range (any in-range target is
    interior).  The support is a pure function of the train range and THIS
    prediction -- it never reads another model, so grids are per-model and are
    built in the single existing predict->score pass.

    Returns ``(bin_edges, probas)``.  Edges are 1-D ``(n_bins + 1,)`` when every
    row shares one support (a shared 1-D abscissa in), else 2-D
    ``(rows, n_bins + 1)``.
    """
    x = np.atleast_2d(np.asarray(cdf_support_point, dtype=np.float64))
    c = np.atleast_2d(np.asarray(cdf_levels, dtype=np.float64))
    rows = max(x.shape[0], c.shape[0])
    x_shared = x.shape[0] == 1
    if x_shared:
        lo = np.full(rows, min(float(y_lo), float(x[0, 0])))
        hi = np.full(rows, max(float(y_hi), float(x[0, -1])))
    else:
        lo = np.minimum(float(y_lo), x[:, 0])
        hi = np.maximum(float(y_hi), x[:, -1])

    edges, probas = _interpolate_cdf_to_grid_with_equally_sized_bins(x, c, lo, hi, n_bins)
    # When every row lands on the same support drop edges to 1-D so metrics.py
    # stays on its cheaper shared-grid branch.
    if bool(np.all(lo == lo[0]) and np.all(hi == hi[0])):
        return edges[0], probas
    return edges, probas


# ===========================================================================
# Scored-view container
# ===========================================================================

@dataclass
class DistributionPredictionView:
    """One scored view of a prediction: either the native PMF grid or the resampled grid.

    Returned by :attr:`~.base.DistributionPrediction.native` /
    :attr:`~.base.DistributionPrediction.resampled`; the two views share this
    structure and are distinguished only by which property hands them out.

    ``bin_edges`` / ``bin_midpoints`` are 1-D (shared grid) or 2-D (per-sample).
    On the **native** view widths may be zero (atoms); on the **resampled** view
    widths are strictly positive by construction.
    """
    probas: np.ndarray
    bin_edges: np.ndarray
    bin_midpoints: np.ndarray
    mean: np.ndarray
    is_sample_based: bool = False
    is_grid_native: bool = False
