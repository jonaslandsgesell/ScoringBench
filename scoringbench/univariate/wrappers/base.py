"""Base classes for ScoringBench probabilistic model wrappers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator

EPS = 100 * np.finfo(np.float64).eps

# A point mass has no width, so a histogram has to invent one.  MIN_PAD is the
# absolute floor near the origin; TIE_PAD_REL scales the invented width with the
# row's own magnitude so it does not underflow at large coordinates.
MIN_PAD = 1e-7
TIE_PAD_REL = 1e-6


# ---------------------------------------------------------------------------
# Regridding
# ---------------------------------------------------------------------------

def regular_support(x, n_bins):
    """Support ``(z_min, z_max)`` for a regular grid over the abscissae ``x``.

    Both outputs have shape ``(rows, 1)``; ``x`` is ``(rows, m)`` sorted.

    Tail extension
    --------------
    A quantile prediction stops at its outermost levels, so the mass below
    ``alpha_0`` and above ``alpha_{K-1}`` has nowhere to live and the CDF never
    reaches 0 or 1 inside the predicted range.  The convention here (TabICL's) is
    to extend the support by one *local* spacing on each side -- the gap between
    the two outermost abscissae -- so that ``C = 0`` / ``C = 1`` can be pinned at
    the extended ends.  Reading the extension off the prediction's own outer
    spacing keeps it scale free: a sharp row gets a narrow tail, a diffuse one a
    wide tail, and no external length is imposed on the problem.

    Degeneracy guard
    ----------------
    Two things can leave the span unusable, and both are handled by widening it
    symmetrically around its own centre:

    * **No support at all** -- every abscissa ties, so there is no outer spacing
      to read and both ends coincide.
    * **A span too thin to resolve** -- ``n_bins`` edges cut from the span must be
      *representable*, so the span has to exceed ``n_bins`` ULP at its own
      coordinate; far from the origin one ULP can dwarf an absolute pad and round
      every interior edge onto its neighbour, reintroducing the zero-width bins
      this module exists to remove.

    The floor is therefore ``max(n_bins * ulp(x), |x| * TIE_PAD_REL, MIN_PAD)``:
    representable at any magnitude, proportional to the row's own scale, and with
    an absolute fallback near zero where the ULP argument says nothing.
    """
    left = np.maximum(x[:, 1:2] - x[:, 0:1], 0.0)
    right = np.maximum(x[:, -1:] - x[:, -2:-1], 0.0)
    z_min = x[:, 0:1] - left
    z_max = x[:, -1:] + right

    scale = np.maximum(np.abs(z_min), np.abs(z_max))
    min_span = np.maximum(
        np.maximum(n_bins * np.spacing(scale), scale * TIE_PAD_REL),
        MIN_PAD,
    )
    short = (z_max - z_min) < min_span
    if np.any(short):
        mid = 0.5 * (z_min + z_max)
        half = 0.5 * min_span
        z_min = np.where(short, mid - half, z_min)
        z_max = np.where(short, mid + half, z_max)
    return z_min, z_max


def _collapse_ties(x_row):
    """Collapse tied abscissae; return ``(xs, order)`` for :func:`_monotone_cdf_at`.

    ``xs`` are the sorted unique abscissae of ``x_row`` and ``order`` maps each
    input node onto its slot in ``xs``.  Split out of :func:`_monotone_cdf_at` so
    a *shared* abscissa row -- every sample resampled against one grid, as when a
    natively-gridded model's borders are repaired -- pays for the sort once
    instead of once per sample.
    """
    xs = np.unique(x_row)
    return xs, np.searchsorted(xs, x_row)


def _monotone_cdf_at(x_row, c_row, edges_row, collapsed=None):
    """Evaluate a monotone CDF at ``edges_row`` with a shape-preserving cubic.

    ``x_row`` is non-decreasing abscissae and ``c_row`` the matching
    non-decreasing CDF values (both 1-D); ``edges_row`` are the query points.
    ``collapsed`` optionally supplies a precomputed :func:`_collapse_ties`
    result for ``x_row`` (identical output, just not recomputed per row).

    Why PCHIP rather than linear interpolation
    ------------------------------------------
    A linear CDF makes the implied density piecewise *constant*, so a sharp mode
    is flattened onto the bin holding it -- the density at the target is then
    wrong even though the per-bin mass is right.  A monotone cubic (PCHIP) fits a
    C1 curve through the same nodes without overshooting, so ``diff(C)`` follows
    the true density's shape between the predicted levels while staying
    non-negative.  Masses are the exact CDF increment at the bin edges,
    ``C(e_{k+1}) - C(e_k)``, not a sampled derivative.
    `tests/test_interpolation_scheme_selection.py` compares this against linear
    and derivative-sampling variants and selects it.

    Ties (atoms) are preserved
    --------------------------
    PCHIP needs *strictly* increasing abscissae, but a CDF read off quantiles /
    histogram edges may repeat an ``x`` where an atom lives.  Each run of tied
    abscissae is collapsed to its single coordinate carrying the run's *last* CDF
    value -- exactly ``np.interp``'s "last matching node wins" rule -- so the
    atom's whole jump is realised at that one coordinate and lands, undivided, in
    the output bin ``searchsorted(edges[1:], y)`` would send a target on the atom
    to.  With two or fewer distinct abscissae there is no cubic to fit and the
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
    `tests/test_interpolation_scheme_selection.py` selects PCHIP for.
    """
    # np.unique returns the *sorted unique* abscissae; map each to the CDF value
    # of the LAST input node at that abscissa so an atom's full jump is kept.
    xs, order = _collapse_ties(x_row) if collapsed is None else collapsed
    cs = np.zeros(xs.shape, dtype=np.float64)
    cs[order] = c_row                    # later duplicates overwrite -> last wins
    cs = np.maximum.accumulate(cs)       # guard monotonicity after the scatter

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


def resample_cdf_to_regular_grid(x, y, n_bins, shared_x=False):
    """Interpolate a CDF onto a regular grid; return ``(bin_edges, probas)``.

    ``x`` is ``(rows, m)`` non-decreasing abscissae and ``y`` the matching
    non-decreasing CDF values -- ``(rows, m)``, or ``(m,)`` when every row shares
    the same levels.  ``x[:, 0]`` and ``x[:, -1]`` are taken as the support ends,
    so the caller is responsible for having extended them (see
    :func:`regular_support`) and for anchoring ``y`` at 0 and 1 there if it wants
    the full unit of mass inside the grid.  Set ``shared_x=True`` only when every
    row of ``x`` is identical; the returned edges are then 1-D.

    ``n_bins + 1`` equally spaced edges span ``[x[:, 0], x[:, -1]]``; ``C`` is
    interpolated there by :func:`_monotone_cdf_at` and forward-differenced into
    bin masses.  Because ``C`` is monotone the differences are non-negative and
    total mass is exactly ``C(x_last) - C(x_first)``; because they are taken at
    the bin edges each mass is the exact increment ``C(edge_{k+1}) - C(edge_k)``.

    ``shared_x=True`` promises every row of ``x`` is identical.  The edges depend
    on ``x`` alone, so the output grid is genuinely shared too and is returned 1-D
    to keep ``metrics.py`` on its shared-grid branch.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    rows = x.shape[0]
    y2 = np.broadcast_to(y, x.shape) if y.ndim == 1 else y

    # With shared abscissae every row's edges coincide, so build the single row.
    x_rows = x[0:1] if shared_x else x
    z_min = x_rows[:, 0:1]
    z_max = x_rows[:, -1:]

    frac = np.linspace(0.0, 1.0, n_bins + 1)[None, :]
    edges = z_min + (z_max - z_min) * frac
    # linspace hits 0 and 1 exactly, but the affine map need not reproduce the
    # endpoints bit for bit; pin them so the support is exactly as advertised.
    edges[:, 0] = z_min[:, 0]
    edges[:, -1] = z_max[:, 0]

    # Round-off in the affine map can leave an interior edge a hair below its
    # predecessor; a running max removes the resulting negative width.  Done in
    # place so we do not allocate a second (rows, n_bins+1) copy.  Must precede
    # the CDF evaluation so the masses are differenced at the *final* edges.
    np.maximum.accumulate(edges, axis=-1, out=edges)

    # A shared abscissa row has one tie structure; collapse it once, not per row.
    collapsed = _collapse_ties(x[0]) if shared_x else None

    probas = np.empty((rows, n_bins), dtype=np.float64)
    for i in range(rows):
        e_row = edges[0] if shared_x else edges[i]
        c = _monotone_cdf_at(x[i], y2[i], e_row, collapsed=collapsed)
        m = np.maximum(np.diff(c), 0.0)
        s = m.sum()
        probas[i] = m / s if s > 0.0 else 1.0 / n_bins

    return (edges[0] if shared_x else edges), probas


def cdf_nodes_to_regular_grid(x_nodes, c_nodes, n_bins, shared_x=False):
    """Extend support, pin ``C = 0/1`` at the ends, resample onto a regular grid.

    The one entry point every wrapper shares: given monotone CDF nodes
    ``(x_nodes, c_nodes)`` -- ``(rows, m)``, or ``(m,)`` for a shared row -- it
    extends the support by one local spacing (:func:`regular_support`), anchors
    ``C = 0`` / ``C = 1`` on the extension so the grid holds the whole unit of
    mass, and hands off to :func:`resample_cdf_to_regular_grid`.  Owning the
    extend-and-anchor step here keeps every converter on the identical scheme and
    off re-implementing it.

    ``shared_x=True`` declares every row of ``x_nodes`` identical and is forwarded
    to :func:`resample_cdf_to_regular_grid`, which then returns 1-D edges.  The
    extension is read off ``x`` alone, so a shared input stays shared.
    """
    x = np.atleast_2d(np.asarray(x_nodes, dtype=np.float64))
    rows = x.shape[0]
    c = np.broadcast_to(np.asarray(c_nodes, dtype=np.float64), x.shape) if np.ndim(c_nodes) == 1 else np.asarray(c_nodes, dtype=np.float64)

    # Shared abscissae extend identically, so extend the single row and broadcast
    # the result back as a view instead of concatenating `rows` copies of it.
    x_ext = x[0:1] if shared_x else x
    z_min, z_max = regular_support(x_ext, n_bins)
    xx = np.concatenate([z_min, x_ext, z_max], axis=-1)
    if shared_x:
        xx = np.broadcast_to(xx, (rows, xx.shape[-1]))
    yy = np.concatenate([np.zeros((rows, 1)), c, np.ones((rows, 1))], axis=-1)
    return resample_cdf_to_regular_grid(xx, yy, n_bins, shared_x=shared_x)


def _is_regular(e, w):
    """True when every row of ``e`` is a positive, uniform grid with widths ``w``.

    Regularity is the *output* form of :func:`regrid_to_uniform`, so recognising
    it is what makes the map idempotent: the wrappers that already interpolate
    onto a regular grid themselves (``quantiles_to_distribution``,
    ``samples_to_distribution``) pass straight through the container instead of
    having their tails extended and their CDF resampled a second time.

    The tolerance is set by the *edge coordinates*, not by the span.  A grid built
    as ``z_min + (z_max - z_min) * k / n`` rounds each edge to the nearest double,
    so a width can be off by a couple of ULP of the edges it is a difference of.
    When a narrow grid sits far from the origin those ULP dwarf the span, so
    comparing against the span alone would call the grid irregular and regrid it
    forever.
    """
    if w.size == 0 or not np.all(w > 0.0):
        return False
    span = w.sum(axis=-1, keepdims=True)
    target = span / w.shape[-1]
    # Two edges per width, plus slack for the multiply and the divide above.
    tol = 4.0 * np.spacing(np.abs(e).max(axis=-1, keepdims=True))
    return bool(np.all(np.abs(w - target) <= tol))


def regrid_to_uniform(bin_edges, probas):
    """Re-express a PMF on a *regular* per-row grid; return ``(bin_edges, probas)``.

    A quantile-edged prediction uses the predicted quantiles *as* bin edges, so a
    target with repeated values makes edges coincide and bins collapse to
    ``w_k = 0`` (>90% of them on some benchmark datasets), leaving the histogram
    density ``p_k / w_k`` at 0/0.  Rather than shuffle width between neighbours,
    the input is read as a CDF (``C_0 = 0``, ``C_j = sum(p_{<j})``, ``C_K = 1``)
    and resampled onto a positive-width regular grid via
    :func:`cdf_nodes_to_regular_grid`.  Mass is conserved and the bin count is
    preserved, so no wrapper's resolution is silently changed.

    Parameters
    ----------
    bin_edges : (n_bins+1,) or (n_samples, n_bins+1) array
        Per-row edges.  Shape is preserved: a shared 1-D grid stays 1-D.
    probas : (n_samples, n_bins) array
        Per-row PMF.

    Returns
    -------
    (bin_edges, probas)
        ``bin_edges`` keeps the input's rank: the repaired edges depend on the
        input edges alone, so one shared grid maps to one shared grid (1-D in,
        1-D out -- see ``_sanitize_native_grid`` for why that matters).
        An already-regular grid is returned untouched, making the map idempotent.
    """
    e_in = np.asarray(bin_edges, dtype=np.float64)
    p_in = np.asarray(probas, dtype=np.float64)
    if p_in.ndim == 1:
        p_in = p_in[None, :]

    shared = e_in.ndim == 1
    e = e_in[None, :] if shared else e_in

    n_bins = p_in.shape[1]
    if n_bins == 0 or e.shape[-1] != n_bins + 1:
        return bin_edges, probas

    # An already-regular grid is a fixed point: returning it byte for byte makes
    # the map idempotent, so wrappers that resample themselves are not resampled
    # again and a shared grid stays on metrics.py's cheaper shared-grid branch.
    w = np.diff(e, axis=-1)
    if _is_regular(e, w):
        return bin_edges, probas

    rows = max(e.shape[0], p_in.shape[0])
    # Keep a shared grid as one row instead of `rows` identical copies: cheaper in
    # memory and much faster to score downstream.
    e = np.array(e if shared else np.broadcast_to(e, (rows, n_bins + 1)), dtype=np.float64)
    p = np.array(np.broadcast_to(p_in, (rows, n_bins)), dtype=np.float64)

    # Monotone edges (a caller may hand over an unsorted row) and a normalized,
    # non-negative PMF, so the interpolated C is monotone within [0, 1].
    e = np.sort(e, axis=-1)
    p = np.clip(np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    tot = p.sum(axis=-1, keepdims=True)
    p = np.where(tot > 0.0, p / np.where(tot > 0.0, tot, 1.0), 1.0 / n_bins)

    # CDF at the input edges (C_0 = 0, ..., C_K = 1), then the shared
    # extend-anchor-resample path.
    c = np.concatenate([np.zeros((rows, 1)), np.cumsum(p, axis=-1)], axis=-1)
    c[:, -1] = 1.0                                        # exact, not just to rounding
    if shared:
        # `e` is the one shared row; broadcast it back against the per-sample CDF
        # rows as a *view* and tell the resampler the abscissae are shared, so it
        # returns 1-D edges without ever allocating the (rows, n_bins+1) block.
        e = np.broadcast_to(e, (rows, n_bins + 1))
    return cdf_nodes_to_regular_grid(e, c, n_bins, shared_x=shared)


def _has_positive_widths(bin_edges, probas):
    """True when ``bin_edges`` forms a well-formed grid with every width > 0.

    ``False`` for a malformed grid (edge count != bins + 1, or no bins), so the
    caller hands those to :func:`regrid_to_uniform`, which returns them unchanged
    -- the single "trust as-is" decision stays in one place.
    """
    e = np.asarray(bin_edges, dtype=np.float64)
    n_bins = np.asarray(probas).shape[-1]
    if n_bins == 0 or e.shape[-1] != n_bins + 1:
        return False
    # Require widths above EPS, not merely > 0: a genuinely positive but
    # sub-EPS width would survive as-is here and reach the density metrics,
    # where the width clamp then makes f_k = p_k / w_k inconsistent.  Widths in
    # (0, EPS] are treated as degenerate and repaired by regrid_to_uniform,
    # whose MIN_PAD span floor guarantees widths well above EPS.
    return bool(np.all(np.diff(e, axis=-1) > EPS))


def _sanitize_native_grid(bin_edges, probas):
    """Zero-width guard for a model that emits its own histogram grid.

    A natively-gridded model (``is_natively_gridded_model=True``, e.g. TabPFN's
    bar-distribution borders) is trusted to emit positive-width bins -- even
    *irregular* ones -- so a clean grid is returned as the *same objects* (byte
    for byte) and never resampled.  Only when a border repeats (any number of
    ties, giving one or more ``w_k = 0`` bins whose density ``p_k / w_k`` is
    ``0/0``) does the grid fall back to :func:`regrid_to_uniform`, which reads
    the PMF as a CDF and resamples it onto positive widths.

    TabPFN's borders contain ties, so that repair runs on every real prediction;
    it keeps a shared 1-D grid 1-D, which is much faster to score downstream.
    """
    if _has_positive_widths(bin_edges, probas):
        return bin_edges, probas
    return regrid_to_uniform(bin_edges, probas)


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------

@dataclass
class DistributionPrediction:
    """Unified probabilistic prediction container.

    bin_edges / bin_midpoints may be 1-D (shared grid, same for every sample)
    or 2-D (per-sample grid, e.g. when derived from per-sample quantiles).
    metrics.py handles both cases.

    On construction the PMF is re-expressed on a *regular* per-sample grid by
    ``regrid_to_uniform``, which interpolates the prediction's CDF onto that
    grid.  Every consumer therefore sees strictly positive bin widths and no one
    has to special-case atoms.  ``bin_midpoints`` is recomputed from the returned
    edges so it always matches them in value and dimensionality.

    is_natively_gridded_model marks models that already emit a fixed regular
    histogram grid (TabPFN's Riemann-distribution borders).  Such a grid needs no
    repair and resampling it could only blur it, so the regridding is skipped and
    the model's own edges and PMF reach the metrics untouched.

    is_sample_based flags predictions whose PMF was derived from conditional
    draws.  It is informational only; metrics.py scores every prediction the same
    way (the regular-grid PMF already gives a well-defined density).
    """
    probas: np.ndarray         # (n_samples, n_bins)  — PMF: mass per bin, sums to 1
    bin_edges: np.ndarray      # (n_bins+1,) or (n_samples, n_bins+1)
    bin_midpoints: np.ndarray  # (n_bins,)   or (n_samples, n_bins)
    mean: np.ndarray           # (n_samples,)
    is_sample_based: bool = False            # True when PMF came from conditional draws
    is_natively_gridded_model: bool = False  # True when the model emits its own regular grid

    def __post_init__(self):
        if self.is_natively_gridded_model:
            # A model with its own grid is trusted, but a tied border would leave
            # a zero-width (0/0-density) bin; the cheap guard leaves a clean grid
            # untouched and only repairs when a tie actually appears.
            self.bin_edges, self.probas = _sanitize_native_grid(self.bin_edges, self.probas)
        else:
            self.bin_edges, self.probas = regrid_to_uniform(self.bin_edges, self.probas)
        # Regridding may promote a shared 1-D grid to a per-sample 2-D one;
        # recompute the midpoints from the (possibly rewritten) edges so they
        # stay consistent in both value and dimensionality.
        self.bin_midpoints = 0.5 * (self.bin_edges[..., :-1] + self.bin_edges[..., 1:])


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class ProbabilisticWrapper:
    """Base class for ScoringBench model wrappers.

    Subclass and implement fit(), predict(), predict_distribution().
    If predict_distribution() is not yet supported, leave it raising
    NotImplementedError — cv.py will skip distributional metrics gracefully.
    """

    def fit(self, X, y) -> "ProbabilisticWrapper":
        raise NotImplementedError

    def predict(self, X) -> np.ndarray:
        raise NotImplementedError

    def predict_distribution(self, X) -> DistributionPrediction:
        raise NotImplementedError
