"""Purely sample-based multivariate scoring rules.

Every rule is a Monte-Carlo functional of a :class:`MultivariateSamplePrediction`
(an ``(n_test, m, d)`` block of draws) against the observed ``(n_test, d)``
targets.  There is no grid, PMF, CDF, or density anywhere — the draws are the
forecast.

Rules (see the "Preliminary: Multivariate ScoringBench" note)
-------------------------------------------------------------
* **Energy score** ``ES_β(F, y) = E‖Y − y‖^β − ½ E‖Y − Y'‖^β`` with ``β = 1``
  (and a small β family reported as extra columns).  Term 2 uses the *fair*
  estimator ``1/(m(m−1)) Σ_{i≠j}`` so the estimate is unbiased for finite m.
* **Variogram score of order p** ``VS_p = Σ_{a,b} w_{ab} (|y_a − y_b|^p − E|Y_a − Y_b|^p)²``
  with ``p = 0.5`` and uniform weights ``w_{ab} = 1``.
* **Dawid–Sebastiani score** ``DSS = (y − μ)ᵀ Σ⁻¹ (y − μ) + log det Σ`` where
  ``μ, Σ`` are the ensemble sample mean and covariance.

Univariate rules with NO multivariate sample-based analogue (and therefore not
reported here): ``dpd_beta_*``, ``pseudospherical_*``, ``cde_loss``,
``crts_alpha_*``, ``pit_ks*`` — all of these are defined through a density/PMF
on a grid, which this package deliberately does not construct.

Output contract
---------------
:func:`compute_metrics` returns a flat ``dict[str, float]`` of *lower-is-better*
numeric metrics plus point metrics (``mae``, ``rmse`` — vector versions).  The
keys are plain numeric columns so ``autorank_leaderboard.py`` treats them all as
ascending (lower = better), exactly like the univariate energy-score columns.
"""

from __future__ import annotations

import numpy as np
import torch

from .estimators import (
    cross_norm_expectation,
    force_precision,
    pairwise_abs_pow_expectation,
    pairwise_norm_expectation,
)
from .prediction import MultivariateSamplePrediction

# Energy-score exponents reported (β = 1 is the classic energy score).
ENERGY_BETAS = [0.5, 1.0, 1.5]
# Variogram orders reported (p = 0.5 is the benchmark default).
VARIOGRAM_ORDERS = [0.5, 1.0]
# Small ridge added to the ensemble covariance before inversion / log-det so the
# Dawid–Sebastiani score stays finite when a marginal is (near-)degenerate.
_DSS_RIDGE = 1e-6


def _fmt(x: float) -> str:
    """Format a β/p value for a metric-key suffix (1.0 -> '1.0')."""
    return f"{float(x):g}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_metrics(pred: MultivariateSamplePrediction, y_true: np.ndarray) -> dict:
    """All multivariate metrics from a sample prediction.

    Parameters
    ----------
    pred : MultivariateSamplePrediction
        Draws of shape ``(n_test, m, d)`` and mean ``(n_test, d)``.
    y_true : (n_test, d) array
        Observed target vectors.
    """
    return {
        **compute_point_metrics(y_true, pred.mean),
        **compute_scoring_rules(pred, y_true),
    }


def compute_point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Vector MAE / RMSE (mean Euclidean error and root-mean-squared error).

    ``mae`` = mean over instances of ``‖y − ŷ‖`` (Euclidean).
    ``rmse`` = sqrt(mean over instances of ``‖y − ŷ‖²``).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]
    err = np.linalg.norm(y_true - y_pred, axis=-1)  # (n_test,)
    return {
        "mae": float(np.mean(err)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
    }


@force_precision(torch.float64)
def _energy_scores(samples: torch.Tensor, y: torch.Tensor, betas: list[float]) -> dict:
    """Energy score for each β: ``E‖Y−y‖^β − ½ E‖Y−Y'‖^β`` (per-instance mean)."""
    out = {}
    for beta in betas:
        term1 = cross_norm_expectation(samples, y, beta)          # (n_test,)
        term2 = pairwise_norm_expectation(samples, beta)          # (n_test,)
        es = term1 - 0.5 * term2
        # Energy score is non-negative for a proper forecast; tiny negatives are
        # Monte-Carlo noise -> clamp at 0 for a clean leaderboard column.
        es = torch.clamp(es, min=0.0)
        out[f"energy_score_beta_{_fmt(beta)}"] = float(es.mean())
    return out


@force_precision(torch.float64)
def _variogram_scores(samples: torch.Tensor, y: torch.Tensor, orders: list[float]) -> dict:
    """Variogram score for each order p with uniform weights w_{ab}=1."""
    out = {}
    # observed |y_a - y_b|^p : (n_test, d, d)
    y_a = y[:, :, None]
    y_b = y[:, None, :]
    for p in orders:
        observed = (y_a - y_b).abs() ** p                          # (n_test, d, d)
        expected = pairwise_abs_pow_expectation(samples, p)        # (n_test, d, d)
        vs = ((observed - expected) ** 2).sum(dim=(1, 2))          # (n_test,)
        out[f"variogram_score_p_{_fmt(p)}"] = float(vs.mean())
    return out


@force_precision(torch.float64)
def _dawid_sebastiani(samples: torch.Tensor, y: torch.Tensor) -> dict:
    """Dawid–Sebastiani score from ensemble mean μ and covariance Σ.

    ``DSS = (y − μ)ᵀ Σ⁻¹ (y − μ) + log det Σ`` per instance, averaged.
    A small ridge is added to Σ so the inverse and log-det stay finite when a
    coordinate is (near-)degenerate in the ensemble.
    """
    n_test, m, d = samples.shape
    mu = samples.mean(dim=1)                                       # (n_test, d)
    centered = samples - mu[:, None, :]                            # (n_test, m, d)
    # Unbiased sample covariance (divide by m-1).
    cov = torch.einsum("ima,imb->iab", centered, centered) / max(m - 1, 1)
    eye = torch.eye(d, dtype=cov.dtype, device=cov.device)
    cov = cov + _DSS_RIDGE * eye[None, :, :]

    diff = (y - mu)[:, :, None]                                    # (n_test, d, 1)
    sol = torch.linalg.solve(cov, diff)                            # Σ⁻¹ (y-μ)
    quad = (diff * sol).sum(dim=(1, 2))                            # (n_test,)
    logdet = torch.logdet(cov)                                     # (n_test,)
    dss = quad + logdet
    return {"dawid_sebastiani": float(dss.mean())}


def compute_scoring_rules(pred: MultivariateSamplePrediction, y_true: np.ndarray) -> dict:
    """Energy, variogram, and Dawid–Sebastiani scores (all lower-is-better).

    Scoring is a lightweight, numerically sensitive post-processing step run on
    the CPU in float64: the differences ``term1 − term2`` / ``(obs − exp)²`` /
    the DSS quadratic-form-plus-logdet demand double precision, and keeping them
    off the GPU sidesteps device-kernel incompatibilities (the heavy model
    *sampling* is what runs on the GPU, inside the wrappers).
    """
    samples = torch.as_tensor(pred.samples, dtype=torch.float64)
    y = np.asarray(y_true, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    y_t = torch.as_tensor(y, dtype=torch.float64)

    metrics: dict = {}
    metrics.update(_energy_scores(samples, y_t, ENERGY_BETAS))
    metrics.update(_variogram_scores(samples, y_t, VARIOGRAM_ORDERS))
    metrics.update(_dawid_sebastiani(samples, y_t))
    return metrics


# Metric keys produced by compute_scoring_rules (used by cv.py to null-fill on
# point-only fallback, mirroring the univariate cv.py).
SCORING_RULE_KEYS = (
    *[f"energy_score_beta_{_fmt(b)}" for b in ENERGY_BETAS],
    *[f"variogram_score_p_{_fmt(p)}" for p in VARIOGRAM_ORDERS],
    "dawid_sebastiani",
)
