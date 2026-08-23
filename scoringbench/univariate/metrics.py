"""Scoring rules and point metrics for tabular regression.

All functions work on numpy arrays. PyTorch is used internally for GPU acceleration
when available; falls back to CPU otherwise.

Public API
----------
compute_metrics(dist, y_true) -> dict
    All metrics: MAE, RMSE, R², CRPS, sharpness, dispersion,
    90%/95% coverage and interval scores, energy scores β∈{0.5,1,1.5,2},
    CRTS α∈{1.01,1.2,1.5,2}, DPD β∈{0.01,0.2,0.5,1.0}, pseudospherical α∈{1.5,2,3}.

compute_point_metrics(y_true, y_pred) -> dict
    MAE, RMSE, R².

compute_scoring_rules(dist, y_true) -> dict
    CRPS, sharpness, dispersion, coverage and interval scores,
    energy scores, CRTS, DPD, wCRPS_left, wCRPS_right, wCRPS_center.
    dist is a DistributionPrediction from scoringbench.wrappers.
    bin_edges / bin_midpoints may be 1-D (shared grid) or 2-D (per-sample).
    The grid is padded with zero-mass bins at evaluation time so that every
    target y is interior to the integration domain.
    Uses PyTorch on GPU when available; falls back to CPU otherwise.

pad_to_common_grid(probas, bin_edges, bin_mids, y, shared) -> (probas, bin_edges, bin_mids, shared)
    Extend the bin grid with zero-mass bins until every target y is covered.
    Called automatically by compute_scoring_rules; exposed for testing.
"""

import functools
import logging
import time

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ._integration import (
    uniform_axis_integral,
    uniform_slab_pairwise_distance,
)
from .wrappers import DistributionPrediction

logger = logging.getLogger(__name__)

# Energy score β values reported as additional metrics
ENERGY_BETAS = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 1.7, 1.8, 1.9]
DPD_BETAS = [0.01, 0.2, 0.5, 1.0]  # β values for Density Power Divergence scoring rule; 0.01 chosen to avoid log-score support sensitivity
# Tsallis orders α for the Continuous Ranked Tsallis Score (CRTS).  The binary
# α-Tsallis loss recovers the log score as α → 1 (hence CRLS) and the
# Brier/quadratic score at α = 2 (hence CRPS); values in (1, 2) interpolate.
# All must be > 1 (α → 1 reintroduces log-score support sensitivity).
CRTS_ALPHAS = [1.01, 1.2, 1.5, 2.0]
# Orders α for Good's (1971) pseudospherical score.  α = 2 is the classical
# spherical score.  All must be > 1 (α → 1 is the support-sensitive log score).
PSEUDOS_ALPHAS = [1.5, 2.0, 3.0]
DPD_BASED_KEYS = ("cde_loss", *[f"dpd_beta_{b}" for b in DPD_BETAS], *[f"crts_alpha_{a}" for a in CRTS_ALPHAS])
# Central coverage levels (%) reported via coverage_{level} / interval_score_{level};
# the corresponding significance level is alpha = 1 - level/100.
COVERAGE_LEVELS = [20, 40, 60, 80, 90, 95]

# Number of (chunk, n_bins, n_bins) intermediates held concurrently by the
# disjoint-slab branch of uniform_slab_pairwise_distance (the closed form keeps
# ~a dozen edge/offset/power-difference temporaries — left/right corners, two
# _pow_diff calls, num, closed, D, etc. — all of the returned matrix's shape,
# with NO quadrature-node axis).  Used to size the per-sample energy-score
# term-2 chunk against the true peak rather than the returned matrix alone.
# Conservative (rounded up) so the peak lands within the element budget.
_T2_LIVE_INTERMEDIATES = 2


# ---------------------------------------------------------------------------
# Numerical precision
# ---------------------------------------------------------------------------

def force_precision(dtype: torch.dtype = torch.float64):
    """Decorator: upcast every floating-point tensor argument to ``dtype``.

    Histogram scoring rules repeatedly form differences of large, nearly-equal
    quantities — CRPS/energy ``term1 - term2``, variance ``E[X²] - E[X]²``,
    CDE ``∫g² - 2g(y)``, DPD ``∫f^{1+β} - point``.  Evaluated in float32 these
    suffer catastrophic cancellation (observed: CRPS down to ~ -33 on sharp,
    large-scale histograms), violating mathematical guarantees such as
    "energy score ≥ 0" or "variance ≥ 0".  Computing in float64 restores them.

    Integer/index tensors (bin indices, sample indices) and non-tensor
    arguments are passed through unchanged.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def cast(x):
                if isinstance(x, torch.Tensor) and x.is_floating_point():
                    return x.to(dtype)
                return x
            new_args = tuple(cast(a) for a in args)
            new_kwargs = {k: cast(v) for k, v in kwargs.items()}
            return func(*new_args, **new_kwargs)
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_metrics(dist: DistributionPrediction, y_true: np.ndarray) -> dict:
    """All metrics from a DistributionPrediction."""
    return {
        **compute_point_metrics(y_true, dist.mean),
        **compute_scoring_rules(dist, y_true),
    }


def compute_point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE, RMSE, R²."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2":   float(r2_score(y_true, y_pred)),
    }


@force_precision(torch.float64)
def compute_dpd_scores(probas: torch.Tensor, bin_widths: torch.Tensor,
                       g_y: torch.Tensor, betas: list, shared: bool,
                       density_integral=None) -> dict:
    """
    Density Power Divergence (DPD) scoring rule for histogram-based predictive densities.

    For a predictive density f and parameter β>0:
        S_β(f, y) = ∫ f(t)^{1+β} dt - (1 + 1/β) f(y)^β

    Propriety of this rule is a statement about a *single* density f appearing in
    both terms, so the integral and the point term must be two functionals of the
    same f.  ``g_y`` supplies ``f(y)`` and ``density_integral`` supplies
    ``∫ f^{power}``; the caller is responsible for taking both from one density
    (see ``_density_terms``).

    Parameters
    ----------
    density_integral : callable(power) -> (n_samples,) tensor, optional
        Returns ``∫ f(t)^{power} dt`` per sample.  Defaults to the
        piecewise-constant histogram density ``f_hist = p_k / w_k``, for which
        the integral is available in closed form:
        ``∫ f_hist^{1+β} = ∑_k p_k^{1+β} / w_k^{β}``.  The production path passes
        the integral of ``unified_bin_density``, which is that same closed form.

    Returns mean DPD across samples for each β as keys `dpd_beta_{β}`.
    """
    results = {}
    # Broadcast-ready bin widths: (1, n_bins) or (n_samples, n_bins)
    bw = bin_widths[None, :] if shared else bin_widths

    # No eps clamp on g_y: for β > 0, f(y)^β is well-defined at f(y)=0 (it
    # equals 0, contributing 0 to the point term).  The caller guarantees β > 0
    # for the production path (β=0 / log-score is excluded).  Padding the grid
    # to cover y ensures f(y) is the genuine density value, not a floor.
    if density_integral is None:
        eps = 100 * torch.finfo(probas.dtype).eps
        _require_positive_widths(bw, eps, "compute_dpd_scores")
        def density_integral(power):
            # ∫ f_hist^{power} dt = ∑_k p_k^{power} / w_k^{power-1}
            return (probas.pow(power) / bw.pow(power - 1.0)).sum(dim=-1)

    for beta in betas:
        if beta < 0:
            raise ValueError("DPD beta must be >= 0")

        # β -> 0 limit recovers the (negative) log score up to an additive constant.
        if abs(beta) < 1e-12:
            loss = -torch.log(g_y)
        else:
            # Integral term: ∫ f^{1+β}
            integral = density_integral(1.0 + beta)

            # Point evaluation term: (1 + 1/β) * f(y)^β
            point_term = (1.0 + 1.0 / beta) * g_y.pow(beta)

            loss = integral - point_term

        results[f"dpd_beta_{beta}"] = loss.mean().item()

    return results


@force_precision(torch.float64)
def compute_pseudospherical_scores(g_y: torch.Tensor, alphas: list,
                                   density_integral, eps: float) -> dict:
    """Good's (1971) pseudospherical score for histogram predictive densities.

    Affinely-normalised form (Gneiting & Raftery 2007, eq. 12), for order α > 1:

        PseudoS_α(f, y) = 1/(α-1) · [ f(y)^{α-1} / (∫ f(t)^α dt)^{(α-1)/α} − 1 ].

    α = 2 is the classical spherical score; α → 1 degenerates to the
    (support-sensitive) log score, so α is restricted to > 1.  The score is
    strictly proper (maximised at f = g by Hölder), scale-invariant, and
    support-insensitive (the point term f(y)^{α-1} → 0 as f(y) → 0).  Propriety
    requires the SAME density f in both f(y) and ‖f‖_α, so ``g_y`` and
    ``density_integral`` must come from one density (see ``_density_terms``).
    Positively oriented, so we return ``-PseudoS_α`` to keep the module's
    "lower = better" loss convention.

    Parameters
    ----------
    g_y : (n_samples,) tensor
        Pointwise predictive density at the target, f(y).
    alphas : list[float]
        Orders α, each > 1.
    density_integral : callable(power) -> (n_samples,) tensor
        Returns ``∫ f(t)^{power} dt`` per sample, from the SAME density as ``g_y``.
    eps : float
        Numerical floor for the norm to avoid 0-division on empty forecasts.

    Returns
    -------
    dict[str, float]
        ``{"pseudospherical_alpha_{α}": value, ...}`` for each α, negated to a
        loss (lower = better).  ``α = 2`` is the spherical score.
    """
    for alpha in alphas:
        if alpha <= 1.0:
            raise ValueError(
                f"pseudospherical alpha must be > 1 (α → 1 is the support-sensitive "
                f"log score); got {alpha}."
            )

    results: dict = {}
    for alpha in alphas:
        norm_alpha = density_integral(alpha).clamp(min=eps)      # ∫ f^α dt
        # ‖f‖_α^{α-1} = (∫ f^α dt)^{(α-1)/α}
        denom = norm_alpha.pow((alpha - 1.0) / alpha)
        # f(y)^{α-1}; f(y) ≥ 0 so the fractional power is well defined.
        ratio = g_y.clamp(min=0.0).pow(alpha - 1.0) / denom      # (f(y)/‖f‖_α)^{α-1}
        # Gneiting & Raftery (2007, eq. 12) affine normalisation:
        #   PseudoS_α = 1/(α-1) · (ratio − 1),  positively oriented.
        score = (ratio - 1.0) / (alpha - 1.0)
        # Negate so the reported metric is a loss (lower = better).
        results[f"pseudospherical_alpha_{alpha}"] = (-score).mean().item()

    return results


@force_precision(torch.float64)
def compute_energy_score_histogram_corrected(
        probas: torch.Tensor, 
        bin_edges: torch.Tensor, 
        y: torch.Tensor, 
        betas: list = [0.2, 0.5, 1.0, 1.5, 2.0],
    ) -> dict:
        """
        Computes the Energy Score with exact uniform interval-correction.
        At beta=1.0, this mathematically equals the exact continuous CRPS.

        Runs in float64 (see ``force_precision``): term1 - term2 is a difference
        of large, nearly-equal values whose float32 cancellation can drive the
        (non-negative) energy score / CRPS below zero. The per-sample clamp below
        is a final guard restoring the mathematical guarantee score >= 0.

        Parameters
        ----------
        probas : torch.Tensor
            Per-bin probability mass, shape ``(n_samples, n_bins)``.
        bin_edges : torch.Tensor
            The *true* grid edges, shape ``(n_bins + 1,)`` (shared grid) or
            ``(n_samples, n_bins + 1)`` (per-sample grid).  This is the single
            source of truth for the bin geometry: midpoints
            ``m_k = (e_k + e_{k+1}) / 2`` and widths ``w_k = e_{k+1} − e_k`` are
            derived from it.  Because the real edges are used, a Dirac
            (zero-width) bin has both its edges coincident at its location, so
            the term-2 pairwise distance uses the exact point↔slab closed form
            for it — not a midpoint approximation (which is off by ``O(w^2)`` for
            ``β ≠ 1``).
        y : torch.Tensor
            Observations, shape ``(n_samples,)``.
        betas : list of float
            Energy-score exponents ``β ∈ (0, 2]``.
        """
        device = probas.device
        n_samples, n_bins = probas.shape
        shared = (bin_edges.ndim == 1)

        # bin_edges is the single source of truth for the geometry; derive the
        # midpoints and widths from it (rebuilding them if the caller passed a
        # separate representation elsewhere).
        bin_mids = 0.5 * (bin_edges[..., :-1] + bin_edges[..., 1:])   # (...,) n_bins
        bin_widths = bin_edges[..., 1:] - bin_edges[..., :-1]         # (...,) n_bins

        mids_ext = bin_mids[None, :] if shared else bin_mids
        widths_ext = bin_widths[None, :] if shared else bin_widths
        
        results = {}

        # Non-negativity (and the clamp(min=0) guard below) requires ||·||^beta
        # to be conditionally negative definite, which holds only for beta in
        # (0, 2].  Outside this range the energy score can be legitimately
        # negative and clamping would corrupt it — reject all betas up-front.
        for beta in betas:
            if not (0.0 < beta <= 2.0):
                raise ValueError(f"Energy score beta must lie in (0, 2]; got beta={beta}.")

        eps = 100 * torch.finfo(probas.dtype).eps
        # A bin with (near-)zero width is a Dirac point mass at its midpoint,
        # not a uniform slab.  Detect these once (β-independent) so we can use
        # the correct point-mass distance instead of the degenerate integral.
        zero_width = widths_ext <= eps

        # Element budget for chunked loops: keeps the *peak* intermediate within
        # ~0.5 GiB in float64.  The per-sample term-2 primitive
        # (uniform_slab_pairwise_distance) now peaks at ~a dozen intermediates of
        # its returned (chunk, n_bins, n_bins) matrix's shape — the disjoint-slab
        # branch is an exact closed form with NO quadrature-node axis, so there is
        # no longer a hidden quadrature-node blow-up to size against.
        elem_budget = 64_000_000
        if shared:
            # term1 intermediates are (chunk, n_bins); term2 uses an (n_bins, n_bins) matrix.
            chunk_size_t1 = max(1, min(n_samples, elem_budget // max(1, n_bins)))
        else:
            # term1 intermediates are (chunk, n_bins).
            chunk_size_t1 = max(1, min(n_samples, elem_budget // max(1, n_bins)))
            # term2 peak: the primitive holds ~_T2_LIVE_INTERMEDIATES
            # (chunk, n_bins, n_bins) temporaries concurrently, so the true peak
            # is ``chunk · n_bins² · _T2_LIVE_INTERMEDIATES``.  Size the chunk
            # against that peak (not the single returned matrix) so a fine
            # per-sample grid stays within budget.
            t2_peak_per_sample = max(1, n_bins * n_bins * _T2_LIVE_INTERMEDIATES)
            chunk_size_t2 = max(1, min(256, elem_budget // t2_peak_per_sample))

        # The shared-grid term-2 matrix D is built per-β inside the loop below
        # (not cached for all β): each β uses it once, so caching all of them
        # would hold len(betas) dense (n_bins, n_bins) float64 matrices at once
        # — the main OOM driver. Per-β keeps the peak at a single n_bins² matrix.

        # Pre-compute beta-independent term1 quantities to avoid recomputing them
        # for every beta value (12 betas → 12x speedup for this part).
        # These are the edge offsets, magnitudes, and branch predicates that don't
        # depend on the exponent β, only on the bin geometry and targets.
        t1_cache = []
        for i in range(0, n_samples, chunk_size_t1):
            end = min(i + chunk_size_t1, n_samples)
            p_c = probas[i:end]
            y_c = y[i:end]
            ul_c = (mids_ext if shared else bin_mids[i:end]) - widths_ext / 2.0 - y_c[:, None]
            ur_c = (mids_ext if shared else bin_mids[i:end]) + widths_ext / 2.0 - y_c[:, None]
            we_c = widths_ext if shared else bin_widths[i:end]
            zw_c = zero_width if shared else (we_c <= eps)
            al_c = ul_c.abs()
            ar_c = ur_c.abs()
            outside_c = (ul_c * ur_c) > 0            # y outside [a, b]
            big_c = torch.maximum(al_c, ar_c)
            small_c = torch.minimum(al_c, ar_c)
            big_pos_c = big_c > 0
            safe_big_c = torch.where(big_pos_c, big_c, torch.ones_like(big_c))
            r_c = (small_c - big_c) / safe_big_c     # ∈ [−1, 0]
            mid_c = mids_ext if shared else bin_mids[i:end]
            t1_cache.append((i, end, p_c, y_c, we_c, zw_c, al_c, ar_c, outside_c, 
                            safe_big_c, big_pos_c, r_c, mid_c))
        
        for beta in betas:
            # ---- Term 1: E|X - y|^beta  (chunked to avoid OOM on large grids) ----
            # Use pre-computed beta-independent quantities from t1_cache
            term1 = torch.empty(n_samples, dtype=probas.dtype, device=device)
            q_e = beta + 1.0
            for (i, end, p_c, y_c, we_c, zw_c, al_c, ar_c, outside_c, 
                 safe_big_c, big_pos_c, r_c, mid_c) in t1_cache:
                # (1/w) ∫_a^b |x − y|^β dx = [g(u_r) − g(u_l)] / w, where
                # u_l = a − y, u_r = b − y and g(d) = sign(d)|d|^{β+1}/(β+1) is
                # the antiderivative of |·|^β.  When y lies OUTSIDE the slab
                # (u_l, u_r same sign) this is a difference of two same-sign
                # powers |·|^{β+1} that catastrophically cancels for a narrow
                # bin far from y (relative loss ~ |offset| / w, up to ~1e-6 for
                # sub-micron widths at kilo-scale offsets — the exact hazard the
                # term-2 disjoint branch was rewritten to avoid).  Evaluate that
                # branch cancellation-free via expm1/log1p on the same-sign
                # magnitudes.  When y is INSIDE the slab the two terms have
                # opposite signs, so g(u_r) − g(u_l) = (|u_r|^q + |u_l|^q)/q is a
                # SUM (no cancellation) and the direct form is exact.
                out_num_c = safe_big_c.pow(q_e) * (-torch.expm1(q_e * torch.log1p(r_c)))
                out_num_c = torch.where(big_pos_c, out_num_c, torch.zeros_like(out_num_c))
                # Inside: |u_r|^q + |u_l|^q (kink at y splits the slab).
                in_num_c = ar_c.pow(q_e) + al_c.pow(q_e)
                numerator_c = torch.where(outside_c, out_num_c, in_num_c)
                integral_c = numerator_c / (we_c.clamp(min=eps) * q_e)
                if zw_c.any():
                    point_c = (mid_c - y_c[:, None]).abs().pow(beta)
                    expected_c = torch.where(zw_c, point_c, integral_c)
                else:
                    expected_c = integral_c
                term1[i:end] = (p_c * expected_c).sum(dim=-1)
                del numerator_c, integral_c, expected_c

            # ---- Term 2: 0.5 * E|X - X'|^beta ----
            if shared:
                # Build D for this β only, then free it (see note above).
                D = uniform_slab_pairwise_distance(bin_edges, beta, eps=eps)
                term2 = 0.5 * torch.einsum("si,ij,sj->s", probas, D, probas)
                del D
            else:
                # The per-sample pairwise term materialises a (chunk, n_bins,
                # n_bins) tensor; with fine grids this can be very large.
                # Size the chunk so that intermediate stays within elem_budget.
                term2 = torch.empty(n_samples, dtype=probas.dtype, device=device)
                for i in range(0, n_samples, chunk_size_t2):
                    end = min(i + chunk_size_t2, n_samples)
                    p_c = probas[i:end]
                    edges_c = bin_edges[i:end]           # (chunk, n_bins + 1)

                    # Exact uniform-slab pairwise distance (chunk, n_bins,
                    # n_bins) from the real per-sample edges — the primitive
                    # dispatches Dirac bins to the exact point↔slab form, so no
                    # reconstruction or midpoint fallback is needed here.
                    Dc = uniform_slab_pairwise_distance(edges_c, beta, eps=eps)

                    # einsum has no out=; copy the (chunk,) result into the
                    # preallocated buffer, then free Dc before the next chunk.
                    torch.mul(torch.einsum("ci,cij,cj->c", p_c, Dc, p_c),
                              0.5, out=term2[i:end])
                    del Dc

            # Average over samples (clamp per-sample: energy score / CRPS is
            # non-negative by definition; any sub-zero value is numerical error).
            results[f"energy_score_beta_{beta}"] = (term1 - term2).clamp(min=0).mean().item()

        return results

def compute_scoring_rules(dist: DistributionPrediction, y_true: np.ndarray) -> dict:
    """Compute all distributional scoring rules from a DistributionPrediction using PyTorch.

    Returns keys: crps, sharpness, dispersion,
                  coverage_90, interval_score_90,
                  coverage_95, interval_score_95,
                  crts_alpha_{1.01,1.2,1.5,2.0},
                  wcrps_left, wcrps_right, wcrps_center,
                  energy_score_beta_{0.5,1.0,1.5,2.0},
                  dpd_beta_{0.01,0.2,0.5,1.0}.

    Notes
    -----
    Before scoring, the bin grid is extended with zero-mass bins (via
    ``pad_to_common_grid``) so that every target y is interior to the
    integration domain.  This ensures that a model cannot improve its score
    by reporting a narrower support: the CDF values in the padded region are
    forced to 0 (left tail) and 1 (right tail), which is the correct value
    for a model that assigns zero mass there.
    """
    # Compute every scoring rule in float64 (enforced by @force_precision on
    # _compute_scoring_rules_torch).  Several rules form differences of large,
    # nearly-equal terms (variance E[X²] - E[X]², CRPS term1 - term2,
    # CDE ∫g² - 2g(y)); float32 cancellation there breaks guarantees such as
    # variance >= 0 / CRPS >= 0.
    # Build tensors on the compute device *first*, then let force_precision
    # upcast in-place
    _device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probas     = torch.as_tensor(dist.probas,        device=_device)
    bin_edges  = torch.as_tensor(dist.bin_edges,     device=_device)
    bin_mids   = torch.as_tensor(dist.bin_midpoints, device=_device)
    y          = torch.as_tensor(np.array(y_true, dtype=float), device=_device)  # np.array copies -> writable tensor
    shared     = bin_edges.ndim == 1

    logger.debug(
        "compute_scoring_rules: n_samples=%d  n_bins=%d  shared=%s",
        probas.shape[0], probas.shape[1], shared,
    )

    # Extend each sample's grid to cover target y with zero-mass bins.
    # This must happen *before* _compute_scoring_rules_torch so that every
    # scoring rule (DPD, CRTS, CRPS, energy score, interval scores, PIT)
    # sees the correct CDF values in the region beyond the model's reported
    # support.  The padding is model-independent: it depends only on y_true,
    # which is not known at train time.
    # gap_left/gap_right are ignored: CRTS derives the tail contribution from
    # the padded catch-all bins directly (no separate analytic correction).
    probas, bin_edges, bin_mids, shared, _gap_left, _gap_right = pad_to_common_grid(
        probas, bin_edges, bin_mids, y, shared
    )

    t0 = time.perf_counter()
    result = _compute_scoring_rules_torch(probas, bin_edges, bin_mids, y, shared)
    logger.debug("  torch backend      %.4fs (device=%s)",
                 time.perf_counter() - t0,
                 "cuda" if torch.cuda.is_available() else "cpu")
    return result



# ---------------------------------------------------------------------------
# PyTorch (GPU) implementation helpers
# ---------------------------------------------------------------------------

def pad_to_common_grid(
    probas: torch.Tensor,
    bin_edges: torch.Tensor,
    bin_mids: torch.Tensor,
    y: torch.Tensor,
    shared: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool, torch.Tensor, torch.Tensor]:
    """Extend the bin grid so that every target y is interior to the support.

    When a model's self-reported support does not cover a test target, the
    integration domain is too narrow: the CDF is stuck at 0 or 1 in the
    missing region, so the integrand there is evaluated at a certainty that is
    *wrong* for the target.  Padding with zero-mass bins restores the correct
    CDF values (0 to the left, 1 to the right) over the full domain that
    contains y, without changing the model's probability mass.

    **Memory-safe single-bin padding**: rather than adding one narrow bin per
    original bin-width step (which can require millions of bins when a target
    is far outside the support), we add at most ONE zero-mass catch-all bin on
    each side.  Its width spans from the grid edge to just beyond the farthest
    target, so y is guaranteed to be interior.

    Most scoring rules are invariant to this change:
    * **Energy score / CRPS / DPD integral**: zero-mass bins (p=0) contribute
      nothing to any mass-weighted sum, regardless of bin width.
    * **DPD point term**: f(y) = p/w = 0 for a zero-mass bin, which is the
      correct value for a model that assigns no density there.
    * **Interval / wCRPS**: quantile inversion uses the CDF, which is 0 in the
      left pad bin and 1 in the right pad bin — the same values that a wide
      uniform-width padding would produce.
    * **PIT**: already handles out-of-support targets explicitly via
      ``torch.where``; the pad bin merely ensures ``y_bin`` resolves correctly.

    **CRTS tail handling**: CRTS is *not* invariant to the number of zero-mass
    bins, because the integrand s_α(F(t), I(t≥y)) is non-zero in the region
    between the original grid edge and y where F(t)=1 (right tail) or F(t)=0
    (left tail) but the indicator q=I(t≥y) disagrees with the CDF:

    * Right tail (F=1, q=0, i.e. t ∈ [hi, y)):  s_α(1,0) = 1/(α-1)
    * Left tail  (F=0, q=1, i.e. t ∈ (y, lo]):  s_α(0,1) = 1/(α-1)

    Rather than adding a separate analytic term, the catch-all bins built here
    are sized so that y is *inside* the padded grid.  Over the catch-all bin the
    CDF is flat (F ≡ 1 or 0) and the bin is split at y, so ``compute_crts``'s
    ordinary slab integral already reproduces the exact tail contribution
    gap/(α−1) — the same way the β = 1 energy score treats its tails.  The
    ``gap_left``/``gap_right`` outputs are retained for diagnostics/back-compat
    but are no longer consumed by ``compute_crts``.

    Parameters
    ----------
    probas : (n_samples, n_bins) tensor
        PMF (probability mass per bin, sums to 1 per row).
    bin_edges : (n_bins+1,) or (n_samples, n_bins+1) tensor
        Bin edges.  After ``regrid_to_uniform`` the grid is uniform per row,
        so the bin width is constant and can be read off ``bin_edges[..., 1] -
        bin_edges[..., 0]``.
    bin_mids : (n_bins,) or (n_samples, n_bins) tensor
        Bin midpoints.  Recomputed from the padded edges on return.
    y : (n_samples,) tensor
        Test targets.
    shared : bool
        Whether the grid is shared across samples (1-D edges) or per-sample
        (2-D edges).

    Returns
    -------
    probas, bin_edges, bin_mids, shared, gap_left, gap_right
        Tensors with at most one zero-mass bin prepended and/or appended as
        needed.  If no padding is required the inputs are returned unchanged.
        ``gap_left`` and ``gap_right`` are (n_samples,) tensors giving the
        distance from each sample's grid edge to y (0 when y is interior),
        kept for diagnostics; the tail is now scored directly via the padded
        catch-all bins, so ``compute_crts`` no longer consumes them.

    Notes
    -----
    * The catch-all bin's width equals ``max(bw, y_extreme - grid_edge)`` so
      it always contains the farthest target.  Its zero mass means it
      contributes nothing to energy score, DPD, or CRPS.
    * For the non-shared case each row gets its own catch-all bin sized to
      contain that row's target.  The grid stays 2-D (per-sample).
    """
    device = probas.device
    dtype  = probas.dtype
    n_samples = probas.shape[0]
    zero_gaps = torch.zeros(n_samples, dtype=dtype, device=device)

    if shared:
        lo     = bin_edges[0].item()
        hi     = bin_edges[-1].item()
        bw     = (bin_edges[1] - bin_edges[0]).item()
        y_min  = y.min().item()
        y_max  = y.max().item()

        need_left  = y_min < lo
        need_right = y_max > hi

        # gap_left/right: per-sample distances from grid edge to y (0 when interior).
        gap_left  = (lo - y).clamp(min=0.0)   # (n_samples,)
        gap_right = (y  - hi).clamp(min=0.0)

        if not need_left and not need_right:
            return probas, bin_edges, bin_mids, shared, zero_gaps, zero_gaps

        # One catch-all bin on each side that needs padding.
        # Width = distance from grid edge to farthest target (at least bw so
        # the bin is non-degenerate and y is strictly interior).
        new_lo = (lo - max(bw, lo - y_min)) if need_left  else lo
        new_hi = (hi + max(bw, y_max - hi)) if need_right else hi

        n_left  = 1 if need_left  else 0
        n_right = 1 if need_right else 0

        # Build new edges: left catch-all edge → original edges → right catch-all edge.
        parts = []
        if need_left:
            parts.append(torch.tensor([new_lo], dtype=dtype, device=device))
        parts.append(bin_edges)
        if need_right:
            parts.append(torch.tensor([new_hi], dtype=dtype, device=device))
        new_edges = torch.cat(parts)
        new_mids  = 0.5 * (new_edges[:-1] + new_edges[1:])

        # Zero-pad PMF on both sides (at most 1 bin each side).
        pad_l = torch.zeros(probas.shape[0], n_left,  dtype=dtype, device=device)
        pad_r = torch.zeros(probas.shape[0], n_right, dtype=dtype, device=device)
        new_probas = torch.cat([pad_l, probas, pad_r], dim=1)

        return new_probas, new_edges, new_mids, True, gap_left, gap_right

    else:
        # Per-sample grids: one catch-all bin per row on each needed side.
        # bin_edges is (n_samples, n_bins+1).
        bw_per  = (bin_edges[:, 1] - bin_edges[:, 0])          # (n_samples,)
        lo_per  = bin_edges[:, 0]                               # (n_samples,)
        hi_per  = bin_edges[:, -1]                              # (n_samples,)

        # gap_left/right: per-sample distances from grid edge to y (0 when interior).
        gap_left  = (lo_per - y).clamp(min=0.0)                # (n_samples,)
        gap_right = (y - hi_per).clamp(min=0.0)

        need_left_per  = gap_left  > 0
        need_right_per = gap_right > 0

        if not need_left_per.any().item() and not need_right_per.any().item():
            return probas, bin_edges, bin_mids, shared, zero_gaps, zero_gaps

        n_samples, n_bins = probas.shape
        # Every row gets the same shape: 1 left pad + original + 1 right pad.
        # Rows that don't need padding on a given side get a dummy bin of
        # width bw (harmless: zero mass, y is not in it for those rows).
        n_bins_new = n_bins + 2  # always 1 left + 1 right catch-all

        new_probas = torch.zeros(n_samples, n_bins_new, dtype=dtype, device=device)
        new_probas[:, 1:n_bins + 1] = probas          # copy original mass into middle

        # Left catch-all: width = max(bw, gap_left) + bw*0.5 so y is strictly
        # interior when it falls on this side.
        w_left = torch.max(bw_per, gap_left + bw_per * 0.5)
        new_lo = lo_per - w_left

        # Right catch-all: symmetric.
        w_right = torch.max(bw_per, gap_right + bw_per * 0.5)
        new_hi  = hi_per + w_right

        # Build per-sample edge tensors: [new_lo, original edges..., new_hi]
        new_edges = torch.zeros(n_samples, n_bins_new + 1, dtype=dtype, device=device)
        new_edges[:, 0]            = new_lo
        new_edges[:, 1:n_bins + 2] = bin_edges
        new_edges[:, n_bins + 2]   = new_hi

        new_mids = 0.5 * (new_edges[:, :-1] + new_edges[:, 1:])
        return new_probas, new_edges, new_mids, False, gap_left, gap_right


@force_precision(torch.float64)
def _interval(alpha, cdf, bin_edges, y, n_samples, n_bins, device, shared, y_bin, ns_idx):
    """Compute interval score and coverage for a given alpha level.
    
    Parameters
    ----------
    alpha : float
        Confidence level (e.g., 0.10 for 90% CI, 0.05 for 95% CI)
    cdf : torch.Tensor
        Cumulative distribution function (n_samples, n_bins)
    bin_edges : torch.Tensor
        Bin edges (n_bins+1,) or (n_samples, n_bins+1)
    y : torch.Tensor
        Target values (n_samples,)
    n_samples, n_bins, device, shared, y_bin, ns_idx : context variables
    
    Returns
    -------
    interval_score : float
        Mean interval score across samples
    coverage : float
        Empirical coverage (fraction of samples where y is in interval)
    """
    lower_q, upper_q = alpha / 2.0, 1.0 - alpha / 2.0
    if shared:
        n_e = len(bin_edges)
        # uint8 (1 byte/elem) is bit-identical to long (8 bytes/elem) for
        # argmax on a 0/1 tensor; saves 8× on the (n_samples, n_bins) bool cast.
        idx_l = (cdf >= lower_q).to(torch.uint8).argmax(dim=1).clamp(max=n_e - 1)
        idx_u = ((cdf >= upper_q).to(torch.uint8).argmax(dim=1) + 1).clamp(max=n_e - 1)
        lows  = bin_edges[idx_l]
        highs = bin_edges[idx_u]
    else:
        # dtype=cdf.dtype is required, not cosmetic: `force_precision` upcasts the
        # *arguments*, but these probes are built here, so a bare `torch.full`
        # would take the global default (float32) and `searchsorted` would compare
        # the float64 `cdf` against e.g. 0.7 -> 0.699999988079071.  Rows whose CDF
        # crosses inside that ~1e-8 window then pick the neighbouring bin, which
        # is how the shared (argmax, exact-double) and per-sample branches drifted
        # apart on `interval_score_40` for numerically identical inputs.
        q_l = torch.full((n_samples, 1), lower_q, device=device, dtype=cdf.dtype)
        q_u = torch.full((n_samples, 1), upper_q, device=device, dtype=cdf.dtype)
        idx_l = torch.searchsorted(cdf.contiguous(), q_l).squeeze(1).clamp(0, n_bins - 1)
        idx_u = (torch.searchsorted(cdf.contiguous(), q_u).squeeze(1) + 1).clamp(0, n_bins - 1)
        lows  = bin_edges[ns_idx, idx_l]
        highs = bin_edges[ns_idx, idx_u]
    
    # `.float()` is float32 *by definition* -- it ignores the receiver's dtype --
    # so it would compute this mean in single precision on the float64 path.  The
    # count is exact either way, but the division is not, so cast to the working
    # dtype to keep the returned coverage double like every other output here.
    cov = ((y >= lows) & (y <= highs)).to(cdf.dtype).mean().item()
    sc  = ((highs - lows)
            + (2.0 / alpha) * (lows  - y).clamp(min=0)
            + (2.0 / alpha) * (y - highs).clamp(min=0))
    return sc.mean().item(), cov


@force_precision(torch.float64)
def compute_quantile_wcrps(cdf, bin_mids, y, n_samples, n_bins, device, shared):
    """Compute quantile-weighted CRPS with three weighting schemes.
    
    Quantile-Weighted CRPS (Gneiting & Ranjan 2011, Eq. 17):
        qwCRPS_v(F, y) = 2 ∫₀¹ ρ_α(y, q_α) v(α) dα
    
    where ρ_α(y, q) = (I[y ≤ q] − α)(q − y) is the pinball/check function.
    
    Weight functions (Table 1, Gneiting & Ranjan 2011):
        left-tail:  v(α) = (1−α)²      (emphasizes underprediction)
        right-tail: v(α) = α²          (emphasizes overprediction)
        center:     v(α) = α(1−α)      (balanced)
    
    Returns
    -------
    dict with keys: wcrps_left, wcrps_right, wcrps_center
    """
    # dtype=cdf.dtype matters twice over.  `force_precision` only upcasts the
    # arguments, so a bare `linspace` would be float32 here: (a) `searchsorted`
    # would probe the float64 `cdf` with e.g. 0.07 -> 0.070000000298..., shifting
    # the recovered quantile bin on rows that cross inside that window, and (b)
    # `alphas_qw` is subtracted from the indicator below, where float32 levels
    # would cap the pinball loss's precision regardless of the float64 inputs.
    alphas_qw = torch.linspace(0.01, 0.99, 99, device=device, dtype=cdf.dtype)   # (A,)

    # Invert the CDF: for each sample i and level α_j find the smallest bin k
    # with cdf[i, k] >= α_j.  Expand alphas to (n_samples, A) so searchsorted
    # can match the (n_samples, n_bins) cdf row-by-row.
    alphas_expanded = alphas_qw[None, :].expand(n_samples, -1).contiguous()  # (n_samples, A)
    idx_q = torch.searchsorted(cdf.contiguous(), alphas_expanded).clamp(0, n_bins - 1)

    if shared:
        q_a = bin_mids[idx_q]                    # (n_samples, A)
    else:
        q_a = torch.gather(bin_mids, 1, idx_q)   # (n_samples, A)

    # Pinball loss per sample and quantile level: 2(I[y ≤ q_α] − α)(q_α − y)
    # Cast the indicator to `alphas_qw.dtype` rather than `.float()`: `.float()`
    # is float32 by definition, which would silently drop this difference (and
    # hence the whole score) back to single precision.
    pinball = (
        2.0
        * ((y[:, None] <= q_a).to(alphas_qw.dtype) - alphas_qw[None, :])
        * (q_a - y[:, None])
    )                                                              # (n_samples, A)

    v_left   = (1.0 - alphas_qw).pow(2)                          # (A,)
    v_right  = alphas_qw.pow(2)
    v_center = alphas_qw * (1.0 - alphas_qw)

    # ∫₀¹ pinball(α)·v(α) dα via the shared uniform-grid quadrature.  The 99
    # levels are the *interior* points of a uniform grid on (0, 1), so the
    # "midpoint" rule (equal weight 1/(99+1) per sample, accounting for the two
    # open end intervals) reproduces the original Gneiting–Ranjan discretisation
    # exactly.
    def _qw_integral(v):
        return uniform_axis_integral(
            pinball * v[None, :], a=0.0, b=1.0, rule="midpoint", dim=-1
        ).mean().item()

    wcrps_left   = _qw_integral(v_left)
    wcrps_right  = _qw_integral(v_right)
    wcrps_center = _qw_integral(v_center)

    return {
        "wcrps_left":   wcrps_left,
        "wcrps_right":  wcrps_right,
        "wcrps_center": wcrps_center,
    }


@force_precision(torch.float64)
def compute_crts(
    cdf: torch.Tensor,
    bin_edges: torch.Tensor,
    y: torch.Tensor,
    y_bin: torch.Tensor,
    shared: bool,
    alphas: list[float] = CRTS_ALPHAS,
) -> dict[str, float]:
    """Continuous Ranked Tsallis Score (CRTS) for multiple Tsallis orders α.

    Integrates the strictly proper binary α-Tsallis loss of the threshold
    indicator I(t ≥ y) against the predicted CDF F(t).  With p = F(t) and
    q = I(t ≥ y) the (divergence-form) integrand is

        s_α(p, q) = [p^α + (1-p)^α]/α - [q p^{α-1} + (1-q)(1-p)^{α-1}]/(α-1)
                    - [1/α - 1/(α-1)].

    α → 1 recovers the log score (CRLS), α = 2 the Brier/quadratic score
    (so ``crts_alpha_2.0`` coincides with CRPS); useful values are 1 < α ≤ 2.
    The divergence form makes both tails vanish (s_α(0,0) = s_α(1,1) = 0), so
    the score is support-insensitive: models with different reported supports
    stay comparable and truncation cannot buy a better score.

    Discretisation: within each bin F is taken linear (uniform-slab law) and the
    bin containing y is split at y, matching the β=1 energy score so
    ``crts_alpha_2.0`` reproduces CRPS.  Each slab integral is closed form via
    ``_crts_slab_integral`` (exact, no quadrature error).

    Parameters
    ----------
    cdf : (n_samples, n_bins) tensor
        Predicted CDF values at the right edge of each bin.
    bin_edges : (n_bins+1,) or (n_samples, n_bins+1) tensor
        Bin edges; widths and left-edge coordinates are derived from these.
    y : (n_samples,) tensor
        Target values, used to split the containing bin exactly.
    y_bin : (n_samples,) int tensor
        Bin index of each target value.
    shared : bool
        Whether the bin grid is shared across samples.
    alphas : list[float]
        Tsallis orders, each > 1 + 1e-4 (α → 1 reintroduces support sensitivity).

    The caller pads the grid (``pad_to_common_grid``) so every y sits in a
    zero-mass catch-all bin; the flat-CDF slab integral over it already yields
    the exact tail contribution gap/(α-1), so no separate tail term is added.

    Returns
    -------
    dict[str, float]
        ``{"crts_alpha_{a}": value, ...}`` for each α in ``alphas``.
    """
    for alpha in alphas:
        if alpha <= 1.0 + 1e-4:
            raise ValueError(
                f"CRTS alpha must be > 1 + 1e-4 to avoid log-score support sensitivity; got {alpha}."
            )

    device = cdf.device
    n_samples, n_bins = cdf.shape

    # No analytic tail correction: the caller pads the grid so every y sits in
    # a zero-mass catch-all bin whose flat-CDF slab integral already yields the
    # exact tail contribution gap/(α-1) (see the "Tail handling" docstring).

    # Renormalise so the CDF *exactly* saturates at 1 on its final edge.
    # ``cdf = cumsum(probas)`` accumulates O(machine-eps) round-off, so its last
    # entry is 1 − r with r ~ ±1e-16 in ~40–50 % of samples.  On a saturated
    # tail slab (F ≡ 1, q = 0) the point term charges ``(1-F)^{α-1} = r^{α-1}``,
    # and r^{α-1} ≈ 0.71 at α = 1.01 — mischarging the out-of-support gap by up
    # to ~65 % of the whole score.  Dividing by the final CDF value pins
    # F(last edge) to exactly 1 (a normalised histogram integrates to 1 by
    # construction, so this only removes the round-off, never real mass).
    total = cdf[:, -1:].clamp(min=100.0 * torch.finfo(cdf.dtype).eps)
    cdf = cdf / total

    # Slab-aligned discretisation: within each bin F rises LINEARLY from its
    # left-edge value F_{k-1} to its right-edge value F_k = cdf_k (the same
    # uniform-slab assumption the β=1 energy score makes), and the bin
    # containing y is split at y exactly.  This keeps crts_alpha_2.0 consistent
    # with energy_score_beta_1.0 (= CRPS) to numerical precision.
    F_right = cdf                                             # (n_samples, n_bins)
    F_left = torch.zeros_like(cdf)
    F_left[:, 1:] = cdf[:, :-1]                               # F_{k-1}, F_{-1}=0

    # Bin widths and left-edge coordinates from the edges.
    if shared:
        widths  = torch.diff(bin_edges)[None, :].expand(n_samples, n_bins)
        edge_lo = bin_edges[:-1][None, :].expand(n_samples, n_bins)
    else:
        widths  = torch.diff(bin_edges, dim=-1)
        edge_lo = bin_edges[:, :-1]

    bin_ids  = torch.arange(n_bins, device=device)[None, :]   # (1, n_bins)
    is_below = bin_ids < y_bin[:, None]                       # q = 0 over whole bin
    is_above = bin_ids > y_bin[:, None]                       # q = 1 over whole bin
    is_split = bin_ids == y_bin[:, None]                      # bin contains y

    # Fraction of the containing bin lying below y (0 for a degenerate bin).
    # The only numerical guard CRTS needs is a positive *width* denominator so
    # a zero-width (degenerate) bin does not divide by 0; the CDF itself is
    # never clamped (see ``_crts_slab_integral``).
    w_floor = 100.0 * torch.finfo(widths.dtype).eps
    w_safe = widths.clamp(min=w_floor)
    frac_below = ((y[:, None] - edge_lo) / w_safe).clamp(0.0, 1.0)
    frac_below = torch.where(is_split, frac_below, torch.zeros_like(frac_below))
    F_at_y = F_left + frac_below * (F_right - F_left)         # linear interp at y

    w_lo = widths * frac_below                                # split sub-widths
    w_hi = widths * (1.0 - frac_below)

    results: dict[str, float] = {}
    for alpha in alphas:
        below = _crts_slab_integral(F_left, F_right, widths, 0.0, alpha)
        above = _crts_slab_integral(F_left, F_right, widths, 1.0, alpha)
        split_lo = _crts_slab_integral(F_left, F_at_y, w_lo, 0.0, alpha)
        split_hi = _crts_slab_integral(F_at_y, F_right, w_hi, 1.0, alpha)

        per_bin = torch.where(is_below, below,
                   torch.where(is_above, above, split_lo + split_hi))
        bin_score = per_bin.sum(dim=-1)   # (n_samples,)
        results[f"crts_alpha_{alpha}"] = bin_score.mean().item()

    return results


def _linear_power_integral(A: torch.Tensor, B: torch.Tensor,
                           p: float) -> torch.Tensor:
    r"""Exact ∫₀¹ (A + u·(B−A))^p du for A, B ∈ [0, 1] and p ≥ 0.

    The integrand is a power of a quantity that varies *linearly* across the
    unit interval, so it has the elementary antiderivative

        ∫₀¹ (A + u(B−A))^p du = (B^{p+1} − A^{p+1}) / ((p+1)(B−A))   (A ≠ B)
                              = A^p                                   (A = B).

    Evaluating the closed form *directly* is numerically hazardous: for
    ``B ≈ A`` the numerator ``B^{p+1} − A^{p+1}`` is a difference of two nearly
    equal quantities and its relative error blows up like ``A/(B−A)`` — a
    **relative** loss that no *absolute* "treat as flat" threshold can bound
    (an absolute floor ``|B−A| ≤ ε`` still leaves e.g. ``(B−A)/A ~ 1e-9`` cases
    just above the floor with ~1e-6 relative error, worst exactly where CRTS is
    most sensitive, α → 1).

    We instead evaluate the integral in a **cancellation-free** form.  The
    integral is symmetric under ``u → 1 − u`` (i.e. under swapping ``A ↔ B``)
    and non-negative, so write ``M = max(A, B)``, ``m = min(A, B)`` and let
    ``r = (M − m) / M ∈ [0, 1]`` be the *relative* rise (``M − m = |B − A|`` is
    an exact subtraction of the clamped inputs).  Factoring ``M^{p+1}`` out of
    the numerator turns the catastrophic subtraction into a well-conditioned
    ``1 − (m/M)^{p+1}`` evaluated through ``expm1``/``log1p``::

        I = M^p · (1 − e^x) / ((p+1) · r),   x = (p+1)·log1p(−r) ≤ 0.

    Every step is benign: ``log1p(−r)`` is accurate for small ``r`` (near the
    ``A ≈ B`` singularity) and equals ``−∞`` at ``r = 1`` (``m = 0``, i.e. an
    endpoint is ``0``), where ``1 − e^x → 1`` recovers ``I = M^p/(p+1)``.  As
    ``r → 0`` the ratio ``(1 − e^x)/((p+1) r) → 1`` recovers the removable
    limit ``I = M^p`` smoothly — so no ``A == B`` branch or epsilon threshold is
    needed; the degeneracy falls out as an analytic limit.

    ``A`` and ``B`` are clamped to ``[0, 1]`` to absorb the O(machine-eps)
    round-off in the convex interpolation; a fractional power of a
    marginally-negative base would otherwise return NaN.  Verified to ≤ 4e-16
    relative error against 50-digit mpmath references across the full parameter
    range, including ``A = B``, ``A = 0``, ``B = 0`` and ``(B−A)/A`` down to
    machine epsilon (vs. up to ~1e-2 for the naive closed form).
    """
    A = A.clamp(0.0, 1.0)
    B = B.clamp(0.0, 1.0)
    M = torch.maximum(A, B)
    m = torch.minimum(A, B)
    # ``r = (M - m)/M`` is the relative rise; ``M - m == |B - A|`` is exact.
    # Guard the division by ``M`` so ``M == 0`` (⇒ A == B == 0) yields r = 0.
    M_pos = M > 0
    safe_M = torch.where(M_pos, M, torch.ones_like(M))
    r = torch.where(M_pos, (M - m) / safe_M, torch.zeros_like(M))
    # x = (p+1)·log1p(-r) ≤ 0; log1p(-1) = -inf at r = 1 (an endpoint is 0),
    # where -expm1(x) = 1 - e^x -> 1 and the ratio recovers M^p/(p+1).
    x = (p + 1.0) * torch.log1p(-r)
    one_minus_ex = -torch.expm1(x)            # 1 - e^x ∈ [0, 1], well conditioned
    r_pos = r > 0
    safe_r = torch.where(r_pos, r, torch.ones_like(r))
    generic = M.pow(p) * one_minus_ex / ((p + 1.0) * safe_r)
    # ``r == 0`` (⇒ A == B, incl. the zero-width / saturated-slab limit): I = M^p.
    return torch.where(r_pos, generic, M.pow(p))


def _crts_slab_integral(F_lo: torch.Tensor, F_hi: torch.Tensor,
                        width: torch.Tensor, q: float, alpha: float) -> torch.Tensor:
    """∫ s_α(F(u), q) du over a slab where F rises linearly F_lo→F_hi.

    ``s_α`` is the divergence-form binary α-Tsallis integrand with a *constant*
    indicator q on the slab:

        s_α(F, q) = [F^α + (1-F)^α]/α - [q F^{α-1} + (1-q)(1-F)^{α-1}]/(α-1)
                    - [1/α - 1/(α-1)].

    Because F (and hence 1−F) is *linear* across the slab, every term is a power
    of a linear function and integrates in closed form via
    ``_linear_power_integral`` — the result is **exact** for every admissible α,
    with no quadrature truncation error.  This matters at the endpoints: on a
    saturated tail slab (F ≡ 1, q = 0) the point term ``(1−F)^{α-1}`` must
    evaluate to ``0^{α-1} = 0`` to give the exact gap value ``s_α(1,0) =
    1/(α-1)``.  A fixed Gauss–Legendre rule samples F strictly *inside* the slab
    and, when the endpoint derivative diverges as α → 1, mischarges the tail by
    O(1e-3) even at 16 nodes; the closed form has no such error.  All tensors
    broadcast over (n_samples, n_bins); a zero ``width`` contributes 0.

    F is used on its *natural* range ``[0, 1]`` — never the biased
    ``[eps, 1-eps]``.  For every order α > 1 all four powers ``F^α``,
    ``(1-F)^α``, ``F^{α-1}`` and ``(1-F)^{α-1}`` are finite and *vanish* (not
    diverge) at F = 0 and F = 1, so the endpoints must stay reachable.  An
    ``[eps, 1-eps]`` clamp would instead force ``(1-F)^{α-1} = eps^{α-1}``,
    which for α near 1 is O(1) (≈0.73 at α = 1.01) and would mischarge the whole
    out-of-support gap term.  The caller is responsible for handing in a CDF
    whose final value is *exactly* 1 (``compute_crts`` renormalises); otherwise
    a residual r = 1 − F on a saturated slab would be charged
    ``r^{α-1}/(α-1)`` ≈ 70 at α = 1.01 for r ~ 1e-16.
    """
    offset = 1.0 / alpha - 1.0 / (alpha - 1.0)
    # Integral term  ∫ [F^α + (1-F)^α]/α du
    int_F = _linear_power_integral(F_lo, F_hi, alpha)
    int_1mF = _linear_power_integral(1.0 - F_lo, 1.0 - F_hi, alpha)
    integral_term = (int_F + int_1mF) / alpha
    # Point term  ∫ [q F^{α-1} + (1-q)(1-F)^{α-1}]/(α-1) du
    if q == 1.0:
        pt = _linear_power_integral(F_lo, F_hi, alpha - 1.0)
    else:
        pt = _linear_power_integral(1.0 - F_lo, 1.0 - F_hi, alpha - 1.0)
    point_term = pt / (alpha - 1.0)
    return (integral_term - point_term - offset) * width


def _require_positive_widths(widths: torch.Tensor, eps: float, where: str) -> torch.Tensor:
    """Assert the strictly-positive-width contract of the density path.

    Every histogram reaching the density-based rules (``unified_bin_density``
    and the ``density_integral is None`` fall-backs of DPD / CDE, plus the PIT
    CDF interpolation) is produced by ``wrappers.base``, whose sanitiser
    (``regrid_to_uniform`` / ``_extended_span``) guarantees each bin width is
    strictly greater than ``EPS``.  A width ``<= eps`` here therefore signals a
    broken invariant upstream, not a legitimate input: silently clamping it
    would turn a collapsed bin into a huge-but-finite score that quietly
    dominates a benchmark average.  We fail loudly instead so the bug is caught
    at its source.

    (Zero-width bins *are* valid for the energy score, where they encode a Dirac
    point mass and are handled explicitly; that path does not call this guard.)

    Returns ``widths`` unchanged when the contract holds.
    """
    bad = widths <= eps
    if bool(bad.any()):
        w_min = float(widths[bad].min())
        raise ValueError(
            f"{where}: bin width {w_min:.3e} <= eps ({eps:.3e}); the density "
            "path requires strictly positive widths. This should have been "
            "guaranteed by regrid_to_uniform in wrappers.base -- a non-positive "
            "width here means the histogram was not sanitised upstream."
        )
    return widths


@force_precision(torch.float64)
def compute_pit_ks(probas, cdf, bin_edges, bin_widths, y_bin, y, shared, ns_idx):
    """Compute PIT values and the Kolmogorov-Smirnov p-value vs. Uniform(0, 1).

    Probability Integral Transform (Dawid 1984; Diebold et al. 1998):
        p_t = F_t(x_t)
    If the predictive distributions F_t are ideal and continuous, the PIT
    values {p_t} are i.i.d. Uniform(0, 1).  We test that null hypothesis with
    a one-sample Kolmogorov-Smirnov test.

    For a histogram predictive density we treat each bin as having a uniform
    density (piecewise-linear CDF), so for y in bin k_y:
        F(y) = cdf_{k_y - 1} + p_{k_y} * (y - left_edge_{k_y}) / w_{k_y}
    Values outside the support are clamped to [0, 1].

    Returns
    -------
    dict with keys:
        pit_ks_stat : float    KS statistic (sup |F_emp(p) - p|)
        pit_ks_pvalue : float  Two-sided p-value vs. Uniform(0, 1)
    """
    eps = 100 * torch.finfo(probas.dtype).eps

    # Probability mass and width of the bin containing each y
    p_y = probas.gather(1, y_bin.unsqueeze(1)).squeeze(1)
    if shared:
        w_y = bin_widths[y_bin]
        left_y = bin_edges[y_bin]
        support_lo = bin_edges[0]
        support_hi = bin_edges[-1]
    else:
        w_y = bin_widths.gather(1, y_bin.unsqueeze(1)).squeeze(1)
        left_y = bin_edges.gather(1, y_bin.unsqueeze(1)).squeeze(1)
        support_lo = bin_edges[:, 0]
        support_hi = bin_edges[:, -1]

    # Cumulative mass strictly below the y-bin
    cdf_prev = cdf[ns_idx, y_bin] - p_y
    _require_positive_widths(w_y, eps, "compute_pit_ks")
    frac = ((y - left_y) / w_y).clamp(0.0, 1.0)
    pit = (cdf_prev + p_y * frac).clamp(0.0, 1.0)

    # y outside support -> clamp PIT to 0 / 1.
    # Scalar constants avoid allocating zeros_like / ones_like tensors.
    pit = torch.where(y <= support_lo, 0.0, pit)
    pit = torch.where(y >= support_hi, 1.0, pit)

    pit_np = pit.detach().cpu().numpy().astype(np.float64)
    ks = stats.kstest(pit_np, "uniform")
    return {
        "pit_ks_stat":   float(ks.statistic),
        "pit_ks_pvalue": float(ks.pvalue),
    }


@force_precision(torch.float64)
def unified_bin_density(probas, bin_widths, shared, eps):
    """Piecewise-constant density on bin grid: f_k = p_k / w_k.
    
    Returns
    -------
    f_bins : (n_samples, n_bins) density value of each bin.
    w_eff : (1, n_bins) or (n_samples, n_bins) width of each bin.
    """
    w_eff = bin_widths[None, :] if shared else bin_widths    # (1|n, n_bins)
    _require_positive_widths(w_eff, eps, "unified_bin_density")
    f_bins = probas / w_eff

    # Renormalise so the density integrates to one: with f_k = p_k / w_k the
    # mass in bin k is f_k * w_k = p_k, so the normaliser is exactly sum_k p_k.
    # Reading it straight from ``probas`` (rather than reconstructing
    # f_bins * w_eff) is the exact mass and avoids re-deriving it from the
    # density.  Widths are guaranteed > eps by the guard above, so the only
    # clamp left is on Z, which guards against an all-zero PMF row.
    Z = probas.sum(dim=-1, keepdim=True)
    return f_bins / Z.clamp(min=eps), w_eff


def _density_terms(probas, cdf, bin_edges, bin_widths, bw, y, y_bin, shared, eps):
    """Build (g_y, density_integral) pair from unified_bin_density.
    
    Returns
    -------
    g_y : (n_samples,) pointwise density f(y).
    density_integral : callable(power) -> (n_samples,) integral of f^power.
    """
    f_bins, w_eff = unified_bin_density(probas, bin_widths, shared, eps)
    g_y = f_bins.gather(1, y_bin.unsqueeze(1)).squeeze(1)

    def density_integral(power):
        return (f_bins.pow(power) * w_eff).sum(dim=-1)

    return g_y, density_integral


@force_precision(torch.float64)
def compute_cde_loss(probas, bin_widths, g_y, bw, shared, density_integral=None):
    """Compute Continuous Density Estimation (CDE) Loss.
    
    From Izbicki and Lee (2016): "Nonparametric Conditional Density Estimation..."
    First derived 1980 (Rudemo): "Empirical Choice of Histograms and Kernel Density Estimators"
    
    General proper scoring rule for density comparison:
        L(f, g) = ∫∫ (f(z|x) - g(z|x))² dP(x) dz
                = ∫∫ f² dP(x)dz - 2∫∫ f·g dP(x)dz + ∫∫ g² dP(x)dz
    
    For scoring rules, drop constants independent of g:
        L_CDE(f, g) = ∫∫ g² dP(x)dz - 2∫∫ f·g dP(x)dz
    
    With empirical target f (point mass at y):
        ∫ g² dz  = ∫ [g(z)]² dz        (second moment of g over support)
        ∫ f·g dz = g(y)                 (density of g evaluated at y)
    
    Discretized form (on grid with bin widths w_k and grid PMF p_k):
        ∫ g² dz  ≈  ∑_k (p_k/w_k)² · w_k = ∑_k p_k² / w_k    (exact bin masses/widths)
        g(y)     ≈  p_{k_y} / w_{k_y}                        (forward difference)

    Relationship to DPD
    -------------------
    The CDE (integrated-squared-error / L²) loss is *identical* to the Density
    Power Divergence score at β = 1:
        S_{β=1}(g, y) = ∫ g(z)^{1+β} dz - (1 + 1/β) g(y)^β
                      = ∫ g² dz - 2 g(y).
    The production path therefore does not call this function at all — it reads
    ``cde_loss`` straight off ``dpd_beta_1.0`` so the two cannot drift apart.
    Kept as the standalone, directly readable form of the rule.

    Parameters
    ----------
    probas : torch.Tensor
        Probability masses per bin (n_samples, n_bins)
    bin_widths : torch.Tensor
        Bin widths (n_bins,) or (n_samples, n_bins)
    g_y : torch.Tensor
        Pointwise predictive density at the target, f(y) (n_samples,).
    bw : torch.Tensor
        Broadcast-ready bin widths
    shared : bool
        Whether grid is shared or per-sample
    density_integral : callable(power) -> (n_samples,) tensor, optional
        ``∫ f(t)^{power} dt`` per sample; must come from the same density as
        ``g_y``.  Defaults to the histogram density ``∑_k p_k² / w_k``.

    Returns
    -------
    float
        Mean CDE loss across samples
    """
    eps = 100*torch.finfo(probas.dtype).eps
    if density_integral is None:
        _require_positive_widths(bw, eps, "compute_cde_loss")
        term1 = (probas.pow(2) / bw).sum(dim=-1)                 # ∫ g² dz
    else:
        term1 = density_integral(2.0)                            # ∫ g² dz
    term2 = 2.0 * g_y.clamp(min=eps)                         # 2·g(y)
    cde_loss = (term1 - term2).mean().item()
    return cde_loss


@force_precision(torch.float64)
def _compute_scoring_rules_torch(probas, bin_edges, bin_mids, y, shared):
    """All scoring rules computed on GPU (or CPU) via PyTorch tensors.

    Note: `probas` are PMF values (probability mass per bin), i.e. for each
    sample the entries satisfy ∑_k p_k = 1 and represent P(z ∈ bin_k).
    To obtain a density at a bin midpoint divide by the bin width:
    density_k = p_k / w_k. Integrating densities over the grid then
    recovers 1: ∑_k density_k * w_k = 1.

    Inputs are float64 tensors (upcast by ``@force_precision``); here we only
    move them onto the compute device.
    """
    # Tensors are already on the target device (moved before the
    # force_precision upcast in compute_scoring_rules); .to(device) is a
    # no-op here but kept for safety when the function is called directly.
    device = probas.device

    n_samples, n_bins = probas.shape
    ns_idx = torch.arange(n_samples, device=device)

    bin_widths = torch.diff(bin_edges, dim=-1)           # (n_bins,) or (n_samples, n_bins)
    bw = bin_widths[None, :] if shared else bin_widths   # broadcast-ready

    cdf = torch.cumsum(probas, dim=-1)                   # (n_samples, n_bins)
    eps = 100*torch.finfo(probas.dtype).eps

    mids = bin_mids[None, :] if shared else bin_mids     # broadcast-ready

    # ---- bin index of each y (reused by CRTS, DPD) ----
    if shared:
        y_bin = torch.searchsorted(bin_edges[1:].contiguous(), y).clamp(0, n_bins - 1)
    else:
        y_bin = torch.searchsorted(
            bin_edges[:, 1:].contiguous(), y.unsqueeze(1)
        ).squeeze(1).clamp(0, n_bins - 1)

    # ---- Quantile-Weighted CRPS (Gneiting & Ranjan 2011, Eq. 17) ----
    qwcrps_result = compute_quantile_wcrps(cdf, bin_mids, y, n_samples, n_bins, device, shared)
    wcrps_left   = qwcrps_result["wcrps_left"]
    wcrps_right  = qwcrps_result["wcrps_right"]
    wcrps_center = qwcrps_result["wcrps_center"]

    # ---- DPD scores, incl. dpd_beta_0.01 and cde_loss (β=1) ----
    # Reading all off one call keeps them exactly consistent.  The density
    # terms (``f_bins`` / ``g_y`` / the ``density_integral`` closure) are each a
    # full (n_samples, n_bins) tensor; build and consume them inside this scope
    # so they are freed on return instead of being pinned alive through the
    # interval, energy-score and PIT sections below.
    #
    # log_score (β=0) is intentionally excluded: it is sensitive to the support
    # of the predicted distribution (yields -log(ε) for out-of-support y) and
    # is therefore not suitable for benchmarking.  Use dpd_beta_0.01 instead.
    def _dpd_block():
        # One piecewise-constant density on the bin grid supplies both terms, so
        # the two-term rules (cde_loss, dpd_beta_*) are self-consistent and hence
        # proper for it.  See ``unified_bin_density``.  The SAME density feeds
        # the pseudospherical scores (Good 1971) so ratio and difference rules
        # share one f and stay mutually consistent.
        g_y, density_integral = _density_terms(
            probas, cdf, bin_edges, bin_widths, bw, y, y_bin, shared, eps
        )
        dpd = compute_dpd_scores(probas, bin_widths, g_y,
                                 betas=sorted({*DPD_BETAS, 1.0}),
                                 shared=shared,
                                 density_integral=density_integral)
        pseudos = compute_pseudospherical_scores(
            g_y, PSEUDOS_ALPHAS,
            density_integral=density_integral, eps=eps,
        )
        return dpd, pseudos

    all_dpd, pseudos_scores = _dpd_block()
    dpd_scores = {f"dpd_beta_{b}": all_dpd[f"dpd_beta_{b}"] for b in DPD_BETAS}
    cde_loss = all_dpd["dpd_beta_1.0"]

    # ---- Sharpness & Dispersion (Tran et al. 2020) ----
    # Sharpness: mean of per-sample predictive std.  Dispersion: std of the
    # per-sample predictive std.  Scoped so the (n_samples, n_bins) products
    # ``probas * mids`` / ``probas * mids²`` are freed as soon as the two scalars
    # are read out.
    def _sharpness_dispersion():
        mean_  = (probas * mids).sum(dim=-1)
        var_   = ((probas * mids.pow(2)).sum(dim=-1) - mean_.pow(2)).clamp(min=0)
        std_per_sample = var_.sqrt()                          # (n_samples,)
        # Use unbiased=False to avoid torch warning when n_samples is small
        return std_per_sample.mean().item(), std_per_sample.std(unbiased=False).item()

    sharpness, dispersion = _sharpness_dispersion()

    # ---- Interval scores (shared path: vectorised; non-shared: searchsorted) ----
    # Coverage levels (%) from COVERAGE_LEVELS; significance alpha = 1 - level/100.
    interval_results = {}

    for cov_level in COVERAGE_LEVELS:
        alpha = 1.0 - cov_level / 100.0
        is_alpha, cov_alpha = _interval(alpha, cdf, bin_edges, y, n_samples, n_bins, device, shared, y_bin, ns_idx)
        interval_results[f"coverage_{cov_level}"] = cov_alpha
        interval_results[f"interval_score_{cov_level}"] = is_alpha

    # Every beta is computed independently inside
    # ``compute_energy_score_histogram_corrected`` (its per-beta result does not
    # depend on which other betas are requested), so a single batched call is
    # bit-identical to the per-beta calls -- and CRPS is exactly the β=1.0 energy
    # score, so we read it off here instead of paying for a second full pass over
    # the (chunk, n_bins, n_bins) distance matrices.
    energy_all = compute_energy_score_histogram_corrected(
        probas, bin_edges, y, betas=ENERGY_BETAS
    )
    energy_scores = [energy_all[f"energy_score_beta_{beta}"] for beta in ENERGY_BETAS]
    crps = energy_all["energy_score_beta_1.0"]

    # ---- CRTS (replaces CRLS; uses binary α-Tsallis integrand vs. log-score) ----
    # No gap args: the grid is already padded so tails fall inside catch-all bins.
    crts_scores = compute_crts(cdf, bin_edges, y, y_bin, shared)

    # ---- PIT KS test (Dawid 1984; Diebold et al. 1998) ----
    pit_ks = compute_pit_ks(probas, cdf, bin_edges, bin_widths, y_bin, y, shared, ns_idx)

    return {
        "crps":              crps,
        "sharpness":         sharpness,
        "dispersion":        dispersion,
        **interval_results,
        **crts_scores,
        "cde_loss":          cde_loss,
        "pit_ks_stat":       pit_ks["pit_ks_stat"],
        "pit_ks_pvalue":     pit_ks["pit_ks_pvalue"],
        "wcrps_left":        wcrps_left,
        "wcrps_right":       wcrps_right,
        "wcrps_center":      wcrps_center,
        **{f"energy_score_beta_{b}": v for b, v in zip(ENERGY_BETAS, energy_scores)},
        **dpd_scores,
        **pseudos_scores,
    }
