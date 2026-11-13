"""CDF grid-building utilities for ScoringBench wrappers.

Every prediction reduces to ONE lossless intermediate: monotone CDF nodes
``(cdf_support_point, cdf_levels)`` -- abscissae where the model places its cumulative mass.
From those nodes two grids are derived directly, with no round-trip through an
intermediate histogram:

* the NATIVE grid (:func:`cdf_nodes_to_native_PMF_grid`) -- the nodes used as
  bin edges, atoms preserved; scored by the grid-robust rules, and
* the RESAMPLED grid (:func:`resample_cdf_nodes_to_support_outer_hull_y_train_set_y_instance_prediction_grid`) -- the same
  nodes linearly resampled onto ``num_equally_sized_bins`` uniform bins over the
  per-instance grow-only support; scored by the density rules.

The source builders (:func:`quantiles_to_cdf_nodes`,
:func:`samples_to_quantile_nodes`) turn a wrapper's raw output into those nodes; a
grid-native model (TabPFN bar distribution) skips nodes entirely and its native
PMF grid is kept verbatim for both views.  Samples are read as Hazen quantiles
and then follow the quantile route.

The raw nodes are used AS-IS: no ``C = 0/1`` anchoring and no invented tail.  A
quantile grid reaches only ``[alpha_0, alpha_{K-1}]`` and an eCDF only
``[1/2n, 1 - 1/2n]``.  On the NATIVE grid the residual tail mass is kept as
zero-width atoms at the outermost nodes (:func:`with_tail_atoms`), so every
reported level stays exact -- renormalizing instead would shift all interior
levels and bias coverage/interval scores by ~``alpha_0 + 1 - alpha_{K-1}``.
The RESAMPLED (density) grid needs positive widths, so it spreads the same tail
masses as uniform slabs between the outermost node and the support edge
(:func:`_uniform_slab`); only a tail whose hull already reaches the support edge
is folded into the interior by the normalize in
:func:`_interpolate_cdf_to_grid_with_equally_sized_bins`.

Public surface
--------------
``DistributionPredictionView``
    The view on the probabilistic prediction (either native PMF grid for scoring rules for M1, M2 or resampled smoothed grid for L1,L2 scoring rules).
``quantiles_to_cdf_nodes``, ``samples_to_quantile_nodes``
    Source -> monotone CDF nodes ``(cdf_support_point, cdf_levels)``.
``cdf_nodes_to_native_PMF_grid``
    CDF nodes -> native PMF grid (nodes-as-edges).
``interpolate_cdf_to_grid_with_equally_sized_bins``
    CDF nodes -> equal-width bins over their own hull.
``resample_cdf_nodes_to_support_outer_hull_y_train_set_y_instance_prediction_grid``
    CDF nodes -> the density-rule (resampled) grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ===========================================================================
# Shared low-level helpers (monotone CDF evaluation)
# ===========================================================================

def _evaluate_monotone_cdf(x_row, c_row, edges_row):
    """Evaluate a monotone CDF at ``edges_row`` by linear interpolation.

    ``x_row`` is non-decreasing abscissae and ``c_row`` the matching
    non-decreasing CDF values (both 1-D); ``edges_row`` are the query points.

    Linear interpolation of the CDF nodes (Izbicki et al. 2026,
    https://arxiv.org/pdf/2603.26611) gives a
    piecewise-constant density ``dalpha/dq`` between nodes. Masses are the exact CDF
    increments at the bin edges, ``C(e_{k+1}) - C(e_k)``.

    Ties (atoms) are collapsed, then deliberately smoothed
    ------------------------------------------------------
    A CDF read off quantiles / histogram edges may repeat an ``x`` where an atom
    lives.  Each run of tied abscissae is collapsed to its single coordinate
    carrying the run's *last* CDF value -- ``np.interp``'s "last matching node
    wins" rule -- so the cumulative mass AT the atom is exact.  The jump does NOT
    survive as a jump: the atom's mass is spread linearly over the interval below
    it, and the equal-width grid then caps the density at ``p/w``.  Predictions
    that must keep their jumps are scored on the native view, never resampled.
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
    # np.interp is flat (C_first / C_last) beyond the hull.
    return np.interp(edges_row, xs, cs)




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
    is evaluated at the uniform edges with the linear interpolant
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
        node_row = x[0] if x_shared else x[i]
        cc = _evaluate_monotone_cdf(node_row, c[i], edges[i])
        if node_row[0] == node_row[-1] == edges[i, 0]:
            cc[0] = c[i, 0]
        m = np.maximum(np.diff(cc), 0.0)
        m += _uniform_slab(edges[i], edges[i, 0], node_row[0], c[i, 0])
        m += _uniform_slab(edges[i], node_row[-1], edges[i, -1], 1.0 - c[i, -1])
        s = m.sum()
        probas[i] = m / s if s > 0.0 else 1.0 / n_bins
    return edges, probas


def _uniform_slab(edges, a, b, mass):
    """Bin masses of ``mass`` spread uniformly over ``[a, b]``; nothing if ``b <= a``."""
    if not (b > a and mass > 0.0):
        return 0.0
    overlap = np.clip(np.minimum(edges[1:], b) - np.maximum(edges[:-1], a), 0.0, None)
    return mass * overlap / (b - a)


# ===========================================================================
# Source -> monotone CDF nodes (the one lossless intermediate)
# ===========================================================================

def quantiles_to_cdf_nodes(q, alphas):
    """Read a per-sample quantile matrix as CDF nodes -- used verbatim, no anchor.

    The quantile function ``alpha -> q(alpha)`` IS the CDF: ``q`` (``(rows, K)``)
    are the abscissae, ``alphas`` (``(K,)``, one shared row) the cumulative mass.
    The nodes are returned AS-IS -- ``C`` stops at ``[alpha_0, alpha_{K-1}]``; the
    native grid keeps the tail beyond the outermost levels as atoms at those
    nodes (:func:`cdf_nodes_to_native_PMF_grid`) rather than inventing a tail.
    Tied quantiles stay coincident, so a model's atoms survive as repeated abscissae.

    Returns ``(cdf_support_point, cdf_levels)`` -- each ``(rows, K)``.
    """
    xx = np.atleast_2d(np.asarray(q, dtype=np.float64))
    cc = np.broadcast_to(np.asarray(alphas, dtype=np.float64), xx.shape)
    return xx, cc


def samples_to_quantile_nodes(samples, n_levels):
    """Hazen quantiles of each row of draws at ``n_levels`` equal-mass levels.

    Levels are ``(k - 0.5) / K`` with ``K = min(n_levels, n_draws)``; at
    ``K = n_draws`` the nodes are exactly the sorted draws (the Hazen eCDF), ties
    included.  Returns ``(support (rows, K), levels (K,))`` for the quantile route.
    """
    samples = np.atleast_2d(np.asarray(samples, dtype=np.float64))
    k = max(1, min(int(n_levels), samples.shape[1]))
    levels = (np.arange(k) + 0.5) / k
    if k == samples.shape[1]:
        # Exact draws: np.quantile would interpolate them with round-off.
        return np.sort(samples, axis=1), levels
    return np.quantile(samples, levels, axis=1, method="hazen").T, levels


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

    The native PMF grid for quantile-based wrappers: the nodes ARE the edges and
    ``diff(c)`` the interior masses, so every level the model reported is kept
    exactly (``F(q_k) = alpha_k``).  The unreported tail masses ``alpha_0`` and
    ``1 - alpha_{K-1}`` become zero-width atoms AT the outermost nodes: the model
    says only that this mass lies beyond ``q_0`` / ``q_{K-1}``, so no support is
    invented (flat quantile-function clamping, as ``np.interp`` extrapolates).
    Renormalizing ``diff(c)`` instead would rescale every interior level to
    ``(alpha - alpha_0) / (alpha_{K-1} - alpha_0)`` and bias coverage/interval
    scores (95% -> ~93% for levels 0.01..0.99).  A tail atom is added only when
    some row has tail mass there.  Tied nodes stay coincident, so a model's
    atoms survive as zero-width Dirac bins that the grid-robust rules score
    exactly.

    Returns ``(bin_edges, probas)``: edges ``(rows, m)``, probas ``(rows, m - 1)``.
    """
    edges = np.atleast_2d(np.asarray(cdf_support_point, dtype=np.float64))
    c = np.clip(np.atleast_2d(np.asarray(cdf_levels, dtype=np.float64)), 0.0, 1.0)
    return with_tail_atoms(edges, np.diff(c, axis=-1), c)


def with_tail_atoms(edges, interior_probas, cdf_levels):
    """Append the unreported tail masses as zero-width atoms at the outer edges.

    ``interior_probas`` must carry the raw (un-renormalized) mass between the
    outermost nodes, i.e. sum to ``c_{-1} - c_0``.  The lower tail ``c_0`` becomes
    an atom at ``edges[:, 0]`` and the upper tail ``1 - c_{-1}`` one at
    ``edges[:, -1]``, so the native CDF hits every reported level exactly and no
    support is invented.  An atom is added only when some row has tail mass there.
    """
    edges = np.atleast_2d(np.asarray(edges, dtype=np.float64))
    probas = np.atleast_2d(np.asarray(interior_probas, dtype=np.float64))
    c = np.clip(np.atleast_2d(np.asarray(cdf_levels, dtype=np.float64)), 0.0, 1.0)
    lower_tail, upper_tail = c[:, :1], 1.0 - c[:, -1:]
    if np.any(lower_tail > 0.0):
        edges = np.concatenate([edges[:, :1], edges], axis=-1)
        probas = np.concatenate([np.broadcast_to(lower_tail, (probas.shape[0], 1)), probas], axis=-1)
    if np.any(upper_tail > 0.0):
        edges = np.concatenate([edges, edges[:, -1:]], axis=-1)
        probas = np.concatenate([probas, np.broadcast_to(upper_tail, (probas.shape[0], 1))], axis=-1)
    return edges, _normalize(probas)


def interpolate_cdf_to_grid_with_equally_sized_bins(cdf_support_point, cdf_levels, n_bins):
    """Bin CDF nodes onto ``n_bins`` uniform bins over their own hull.

    Masses are the linear-CDF increments, normalized to 1 over the node hull.

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
