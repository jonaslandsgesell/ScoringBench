"""Scoring rules and point metrics for tabular regression.

All functions work on numpy arrays. PyTorch is used internally for GPU acceleration
when available; falls back to CPU otherwise.

Public API
----------
compute_metrics(dist, y_true) -> dict
    All metrics: MAE, RMSE, R², CRPS, log-score, sharpness, dispersion,
    90%/95% coverage and interval scores, energy scores β∈{0.5,1,1.5,2},
    CRLS.

compute_point_metrics(y_true, y_pred) -> dict
    MAE, RMSE, R².

compute_scoring_rules(dist, y_true) -> dict
    CRPS, log-score, sharpness, dispersion, coverage and interval scores,
    energy scores, CRLS, wCRPS_left, wCRPS_right, wCRPS_center.
    dist is a DistributionPrediction from scoringbench.wrappers.
    bin_edges / bin_midpoints may be 1-D (shared grid) or 2-D (per-sample).
    Uses PyTorch on GPU when available; falls back to CPU otherwise.
"""

import logging
import time

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .wrappers import DistributionPrediction

logger = logging.getLogger(__name__)

# Energy score β values reported as additional metrics
ENERGY_BETAS = [0.2, 0.5, 1.0, 1.5, 2.0]


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


def compute_scoring_rules(dist: DistributionPrediction, y_true: np.ndarray) -> dict:
    """Compute all distributional scoring rules from a DistributionPrediction using PyTorch.

    Returns keys: crps, log_score, sharpness, dispersion,
                  coverage_90, interval_score_90,
                  coverage_95, interval_score_95,
                  crls,
                  wcrps_left, wcrps_right, wcrps_center,
                  energy_score_beta_{0.5,1.0,1.5,2.0}.
    """
    probas     = dist.probas.astype(np.float32)
    bin_edges  = dist.bin_edges.astype(np.float32)
    bin_mids   = dist.bin_midpoints.astype(np.float32)
    y          = np.asarray(y_true, dtype=np.float32)
    shared     = bin_edges.ndim == 1

    logger.debug(
        "compute_scoring_rules: n_samples=%d  n_bins=%d  shared=%s",
        probas.shape[0], probas.shape[1], shared,
    )

    t0 = time.perf_counter()
    result = _compute_scoring_rules_torch(probas, bin_edges, bin_mids, y, shared)
    logger.debug("  torch backend      %.4fs (device=%s)",
                 time.perf_counter() - t0,
                 "cuda" if torch.cuda.is_available() else "cpu")
    return result


# ---------------------------------------------------------------------------
# PyTorch (GPU) implementation
# ---------------------------------------------------------------------------

def _compute_scoring_rules_torch(probas_np, bin_edges_np, bin_mids_np, y_np, shared):
    """All scoring rules computed on GPU (or CPU) via PyTorch tensors.

    Note: `probas` are PMF values (probability mass per bin), i.e. for each
    sample the entries satisfy ∑_k p_k = 1 and represent P(z ∈ bin_k).
    To obtain a density at a bin midpoint divide by the bin width:
    density_k = p_k / w_k. Integrating densities over the grid then
    recovers 1: ∑_k density_k * w_k = 1.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    probas    = torch.as_tensor(probas_np,    dtype=torch.float32, device=device)
    bin_edges = torch.as_tensor(bin_edges_np, dtype=torch.float32, device=device)
    bin_mids  = torch.as_tensor(bin_mids_np,  dtype=torch.float32, device=device)
    y         = torch.as_tensor(y_np,         dtype=torch.float32, device=device)

    n_samples, n_bins = probas.shape
    ns_idx = torch.arange(n_samples, device=device)

    bin_widths = torch.diff(bin_edges, dim=-1)           # (n_bins,) or (n_samples, n_bins)
    bw = bin_widths[None, :] if shared else bin_widths   # broadcast-ready

    cdf = torch.cumsum(probas, dim=-1)                   # (n_samples, n_bins)
    eps = torch.finfo(probas.dtype).eps

    mids = bin_mids[None, :] if shared else bin_mids     # broadcast-ready

    # ---- bin index of each y (reused by log_score, CRLS) ----
    if shared:
        y_bin = torch.searchsorted(bin_edges[1:].contiguous(), y).clamp(0, n_bins - 1)
    else:
        y_bin = torch.searchsorted(
            bin_edges[:, 1:].contiguous(), y.unsqueeze(1)
        ).squeeze(1).clamp(0, n_bins - 1)

    # ---- Quantile-Weighted CRPS (Gneiting & Ranjan 2011, Eq. 17) ----
    # qwCRPS_v(F, y) = 2 ∫₀¹ ρ_α(y, q_α) v(α) dα
    # where ρ_α(y, q) = (I[y ≤ q] − α)(q − y) is the pinball/check function.
    # Weight functions following Table 1 of Gneiting & Ranjan (2011):
    #   left-tail:  v(α) = (1−α)²
    #   right-tail: v(α) = α²
    #   center:     v(α) = α(1−α)
    alphas_qw = torch.linspace(0.01, 0.99, 99, device=device)   # (A,)
    d_alpha   = 1.0 / (len(alphas_qw) + 1)                       # ≈ 0.01

    # Invert the CDF: for each sample i and level α_j find the smallest bin k
    # with cdf[i, k] >= α_j.  Expand alphas to (n_samples, A) so searchsorted
    # can match the (n_samples, n_bins) cdf row-by-row.
    alphas_expanded = alphas_qw[None, :].expand(n_samples, -1).contiguous()  # (n_samples, A)
    idx_q = torch.searchsorted(cdf.contiguous(), alphas_expanded).clamp(0, n_bins - 1)
    # idx_q: (n_samples, A)

    if shared:
        q_a = bin_mids[idx_q]                    # (n_samples, A)
    else:
        q_a = torch.gather(bin_mids, 1, idx_q)   # (n_samples, A)

    # Pinball loss per sample and quantile level: 2(I[y ≤ q_α] − α)(q_α − y)
    pinball = (
        2.0
        * ((y[:, None] <= q_a).float() - alphas_qw[None, :])
        * (q_a - y[:, None])
    )                                                              # (n_samples, A)

    v_left   = (1.0 - alphas_qw).pow(2)                          # (A,)
    v_right  = alphas_qw.pow(2)
    v_center = alphas_qw * (1.0 - alphas_qw)

    wcrps_left   = (pinball * v_left[None, :]).sum(dim=-1).mean().item() * d_alpha
    wcrps_right  = (pinball * v_right[None, :]).sum(dim=-1).mean().item() * d_alpha
    wcrps_center = (pinball * v_center[None, :]).sum(dim=-1).mean().item() * d_alpha

    # ---- Log score ----
    sel_p = probas[ns_idx, y_bin]
    sel_w = bin_widths[y_bin] if shared else bin_widths[ns_idx, y_bin]
    density  = sel_p / sel_w.clamp(min=1e-10)
    log_score = -torch.log(density.clamp(min=1e-10)).mean().item()

    # ---- Sharpness & Dispersion (Tran et al. 2020) ----
    # Sharpness: mean of per-sample predictive std.
    # Dispersion: std of per-sample predictive std.
    mean_  = (probas * mids).sum(dim=-1)
    var_   = ((probas * mids.pow(2)).sum(dim=-1) - mean_.pow(2)).clamp(min=0)
    std_per_sample = var_.sqrt()                          # (n_samples,)
    sharpness  = std_per_sample.mean().item()
    # Use unbiased=False to avoid torch warning when n_samples is small
    dispersion = std_per_sample.std(unbiased=False).item()

    # ---- Interval scores (shared path: vectorised; non-shared: searchsorted) ----
    def _interval(alpha):
        lower_q, upper_q = alpha / 2.0, 1.0 - alpha / 2.0
        if shared:
            n_e = len(bin_edges)
            idx_l = (cdf >= lower_q).long().argmax(dim=1).clamp(max=n_e - 1)
            idx_u = ((cdf >= upper_q).long().argmax(dim=1) + 1).clamp(max=n_e - 1)
            lows  = bin_edges[idx_l]
            highs = bin_edges[idx_u]
        else:
            q_l = torch.full((n_samples, 1), lower_q, device=device)
            q_u = torch.full((n_samples, 1), upper_q, device=device)
            idx_l = torch.searchsorted(cdf.contiguous(), q_l).squeeze(1).clamp(0, n_bins - 1)
            idx_u = (torch.searchsorted(cdf.contiguous(), q_u).squeeze(1) + 1).clamp(0, n_bins - 1)
            lows  = bin_edges[ns_idx, idx_l]
            highs = bin_edges[ns_idx, idx_u]
        cov = ((y >= lows) & (y <= highs)).float().mean().item()
        sc  = ((highs - lows)
               + (2.0 / alpha) * (lows  - y).clamp(min=0)
               + (2.0 / alpha) * (y - highs).clamp(min=0))
        return sc.mean().item(), cov

    is_90, cov_90 = _interval(0.10)
    is_95, cov_95 = _interval(0.05)



    def compute_energy_score_histogram_corrected(
        probas: torch.Tensor, 
        bin_mids: torch.Tensor, 
        bin_widths: torch.Tensor, 
        y: torch.Tensor, 
        betas: list = [0.2, 0.5, 1.0, 1.5, 2.0]
    ) -> dict:
        """
        Computes the Energy Score with exact uniform interval-correction.
        At beta=1.0, this mathematically equals the exact continuous CRPS.
        """
        device = probas.device
        n_samples, n_bins = probas.shape
        shared = (bin_mids.ndim == 1)
        
        mids_ext = bin_mids[None, :] if shared else bin_mids
        widths_ext = bin_widths[None, :] if shared else bin_widths
        
        # Define bin edges for the exact integral
        left_edges = mids_ext - widths_ext / 2.0
        right_edges = mids_ext + widths_ext / 2.0
        
        # Distance from edges to target y
        u_l = left_edges - y[:, None]
        u_r = right_edges - y[:, None]

        results = {}

        for beta in betas:
            # ---- Term 1: E|X - y|^beta ----
            # Exact integral for intra-bin uniform distribution.
            # When a bin has zero width (degenerate bin), both the numerator and
            # denominator are 0 (0/0 → NaN).  We clamp the denominator so that
            # 0-width bins contribute 0 to Term 1, which is the correct limit.
            numerator = u_r * u_r.abs().pow(beta) - u_l * u_l.abs().pow(beta)
            expected_d = numerator / (widths_ext.clamp(min=1e-10) * (beta + 1.0))
            term1 = (probas * expected_d).sum(dim=-1)

            # ---- Term 2: 0.5 * E|X - X'|^beta ----
            if shared:
                D = (bin_mids[:, None] - bin_mids[None, :]).abs()
                if beta != 1.0:
                    D = D.pow(beta)
                
                # Diagonal Correction (The Histogram Spirit Fix)
                diag_corr = (2.0 * bin_widths.pow(beta)) / ((beta + 1.0) * (beta + 2.0))
                D.diagonal().copy_(diag_corr)
                
                term2 = 0.5 * torch.einsum("si,ij,sj->s", probas, D, probas)
            else:
                chunk_size = 256
                term2_parts = []
                for i in range(0, n_samples, chunk_size):
                    end = min(i + chunk_size, n_samples)
                    p_c = probas[i:end]      
                    m_c = bin_mids[i:end]    
                    w_c = bin_widths[i:end]  
                    
                    Dc = (m_c.unsqueeze(2) - m_c.unsqueeze(1)).abs()
                    if beta != 1.0:
                        Dc = Dc.pow(beta)
                    
                    d_corr = (2.0 * w_c.pow(beta)) / ((beta + 1.0) * (beta + 2.0))
                    idx = torch.arange(n_bins, device=device)
                    Dc[:, idx, idx] = d_corr
                    
                    term2_parts.append(0.5 * torch.einsum("ci,cij,cj->c", p_c, Dc, p_c))
                term2 = torch.cat(term2_parts)

            # Average over samples
            results[f"energy_score_beta_{beta}"] = (term1 - term2).mean().item()

        return results
 
    energy_scores = []
    for beta in ENERGY_BETAS:
        energy_scores.append(compute_energy_score_histogram_corrected(probas, bin_mids, bin_widths, y, betas=[beta])[f"energy_score_beta_{beta}"])

    # ---- CRLS (Continuous Ranked Logarithmic Score) ----
    # = -sum_k w_k * [I(k>=target)*log(CDF_k) + I(k<target)*log(1-CDF_k)]
    # This is bin-width-weighted cross-entropy between predicted CDF and the
    # target step-function CDF.  Formula from finetuned_regressor.py.
    bin_idx    = torch.arange(n_bins, device=device)[None, :]  # (1, n_bins)
    target_cdf = (bin_idx >= y_bin[:, None]).float()           # (n_samples, n_bins)
    cdf_c      = cdf.clamp(eps, 1 - eps)
    crls_bins  = target_cdf * (-torch.log(cdf_c)) + (1 - target_cdf) * (-torch.log1p(-cdf_c))
    crls       = (crls_bins * bw).sum(dim=-1).mean().item()

    crps = compute_energy_score_histogram_corrected(probas, bin_mids, bin_widths, y, betas=[1.0])[f"energy_score_beta_{1.0}"]

    # ---- CDE Loss (Continuous Density Estimation Loss) ----
    # From Izbicki and Lee (2016): "Nonparametric Conditional Density Estimation..."
    # First derived 1980 https://www.jstor.org/stable/4615859 Empirical Choice of Histograms and Kernel Density Estimators Mats Rudemo
    #
    # General proper scoring rule for density comparison:
    # L(f, g) = ∫∫ (f(z|x) - g(z|x))² dP(x) dz
    #         = ∫∫ f² dP(x)dz - 2∫∫ f·g dP(x)dz + ∫∫ g² dP(x)dz
    #
    # For scoring rules, we drop constants independent of g:
    #   L_CDE(f, g) = ∫∫ g² dP(x)dz - 2∫∫ f·g dP(x)dz
    #
    # With empirical target f (point mass at y):
    #   ∫ g² dz  = ∫ [g(z)]² dz        (second moment of g over support)
    #   ∫ f·g dz = g(y)                 (density of g evaluated at y)
    #
    # Discretized form (on grid with bin widths w_k and grid PMF p_k):
    #   where g_density_k = p_k / w_k
    #   ∫ g² dz  ≈  ∑_k (p_k/w_k)² · w_k = ∑_k p_k² / w_k
    #   g(y)     ≈  p_ky / w_ky  (where y_bin = k_y finds bin containing y)
    #
    # Grid-stable form (converges as w_k → 0 uniformly):
    #   L_CDE ≈ ∑_k p_k²/w_k − 2 p_ky/w_ky
    term1 = (probas.pow(2) / bw.clamp(min=1e-10)).sum(dim=-1)  # ∫ g² dz
    p_at_y = probas.gather(1, y_bin.unsqueeze(1)).squeeze(1)
    if shared:
        dz_at_y = bin_widths[y_bin]
    else:
        dz_at_y = bin_widths.gather(1, y_bin.unsqueeze(1)).squeeze(1)
    term2 = 2.0 * p_at_y / dz_at_y.clamp(min=1e-10)           # 2·g(y)
    cde_loss = (term1 - term2).mean().item()

    return {
        "crps":              crps,
        "log_score":         log_score,
        "sharpness":         sharpness,
        "dispersion":        dispersion,
        "coverage_90":       cov_90,
        "interval_score_90": is_90,
        "coverage_95":       cov_95,
        "interval_score_95": is_95,
        "crls":              crls,
        "cde_loss":          cde_loss,
        "wcrps_left":        wcrps_left,
        "wcrps_right":       wcrps_right,
        "wcrps_center":      wcrps_center,
        **{f"energy_score_beta_{b}": v for b, v in zip(ENERGY_BETAS, energy_scores)},
    }
