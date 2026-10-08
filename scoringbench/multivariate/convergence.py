"""Multivariate potential scale reduction factor (MPSRF) for Monte-Carlo draws.

This module provides a *sampling-side* convergence diagnostic for the purely
sample-based multivariate benchmark.  Where :mod:`scoringbench.multivariate.metrics`
scores a *fixed* ensemble of draws, this module answers the orthogonal question
**"have we drawn enough samples yet?"** — i.e. is the Monte-Carlo ensemble large
enough that the multivariate scoring rules (energy / variogram / Dawid–Sebastiani)
are stable, and that the *covariance* of the predictive distribution is pinned
down with enough certainty?

Not MCMC
--------
The draws scored here are **i.i.d. Monte-Carlo samples** from a generative
predictive model (TabPFN sampling, a copula, a diffusion head, …), *not* a
Markov chain.  There is therefore **no burn-in / warm-up** to discard and **no
autocorrelation** to deflate the effective sample size: every draw is kept.  We
reuse the Gelman–Rubin *machinery* (splitting an ensemble into groups and
comparing between-group to within-group spread) purely as a well-understood,
dimension-aware "is the spread estimate stable?" statistic.

Literature
----------
Vats & Knudson (2020), *Revisiting the Gelman–Rubin Diagnostic*,
https://arxiv.org/html/1812.09384v3.  We adopt their notation and both their
multivariate statistics:

* :math:`m` groups (chains), :math:`n` samples per group, :math:`p` dimensions;
* within-group covariance :math:`S = \\frac1m \\sum_i S_i` (``S_i`` the sample
  covariance of group ``i``, ddof=1);
* between-group covariance :math:`\\mathbf{B}/n = \\frac{1}{m-1} \\sum_i
  (\\bar{x}_i - \\bar{x})(\\bar{x}_i - \\bar{x})^T`;

**Original MPSRF (their Eq. 8, Brooks–Gelman):**

.. math::

    \\hat{R}^p = \\frac{n-1}{n} + \\frac{m+1}{m}\\,\\lambda_{\\max}(S^{-1}\\mathbf{B}/n),

the worst-case (largest-eigenvalue) scale reduction over all linear
combinations of the ``p`` coordinates.

**New MPSRF (their Eq. 10), determinant / generalized-variance form:**

.. math::

    \\hat{R}^p_{\\det} = \\sqrt[p]{\\frac{\\det(\\hat{\\Sigma})}{\\det(S)}},
    \\qquad \\hat{\\Sigma} = \\frac{n-1}{n} S + \\frac{m+1}{m}\\,\\frac{\\mathbf{B}}{n},

the ``p``-th root of the ratio of generalized variances (geometric mean of the
generalized eigenvalues).  This is more stable than the largest eigenvalue
because it aggregates *all* directions rather than only the worst one.  Both
statistics tend to 1 from above as the groups mix.

Public API
----------
compute_multivariate_rhat(groups, *, method="det", ridge=...) -> float
assess_sample_prediction(pred, *, ...) -> SampleDiagnostic
adaptive_sample_until_converged(sample_dgp, p, *, ...) -> (groups, info)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from scipy.linalg import eigh

# Default ridge added to S when it is singular / ill-conditioned.  Kept tiny so
# it never materially shifts a well-conditioned S.
_DEFAULT_RIDGE = 1e-8

# Condition number above which S is treated as ill-conditioned and the ridge is
# applied pre-emptively.
_COND_THRESHOLD = 1e12

Method = Literal["det"]

# ---------------------------------------------------------------------------
# Diagnostic defaults (used by assess_sample_prediction and the adaptive loop)
# ---------------------------------------------------------------------------
# 4 groups gives 3 df for B/n — the minimum for a stable between-group
# covariance estimate.  2 groups (rank-1 B/n) is too noisy in practice.
DEFAULT_N_GROUPS: int = 4
# Vats & Knudson (2020) recommend R̂ < 1.01 as the strict convergence criterion.
DEFAULT_EPSILON: float = 1.01
# Subsample at most this many test instances for the per-instance summary;
# the diagnostic is a summary statistic so a subsample is sufficient.
DEFAULT_MAX_INSTANCES: int = 512
# Hard cap on total draws per instance in the adaptive top-up loop.
DEFAULT_MAX_SAMPLES: int = 10_000
# Draws added per convergence-check iteration in the adaptive top-up loop.
DEFAULT_BATCH_SIZE: int = 500


# ---------------------------------------------------------------------------
# Core statistic
# ---------------------------------------------------------------------------

def compute_multivariate_rhat(
    groups: np.ndarray,
    *,
    method: Method = "det",
    ridge: float = _DEFAULT_RIDGE,
) -> float:
    """Multivariate potential scale reduction factor :math:`\\hat{R}^p_{\\det}`.

    Implements the determinant / generalized-variance statistic of Vats &
    Knudson (2020), Eq. 10 — the ``p``-th root of the ratio of generalized
    variances, which aggregates *all* directions rather than only the worst one
    and is therefore more stable than the largest-eigenvalue (Brooks–Gelman)
    form.

    Parameters
    ----------
    groups : (m, n, p) array
        ``m`` independent groups (``m >= 2``) of ``n`` i.i.d. draws each of a
        ``p``-dimensional vector.  ``p == 1`` collapses to the classic scalar
        Gelman–Rubin statistic.  No warm-up is discarded — every draw is used.
    method : {"det"}, default "det"
        Only ``"det"`` (Vats & Knudson Eq. 10) is supported.  The old
        largest-eigenvalue statistic (Brooks–Gelman) has been removed.
    ridge : float, default 1e-8
        Ridge added to the within-group covariance ``S`` (``S + ridge * I``)
        when ``S`` is singular or ill-conditioned.

    Returns
    -------
    float
        :math:`\\hat{R}^p_{\\det} \\gtrsim 1`.  Values close to 1 (e.g.
        ``< 1.01``) indicate the ensemble spread is stable along every linear
        combination of the ``p`` coordinates.

    Raises
    ------
    ValueError
        If ``groups`` is not 3-D, has ``m < 2`` or ``n < 2``, contains
        non-finite values, or ``method`` is unknown.
    """
    groups = np.asarray(groups, dtype=np.float64)
    if groups.ndim != 3:
        raise ValueError(f"groups must be 3-D (m, n, p); got shape {groups.shape}")
    m, n, p = groups.shape
    if m < 2:
        raise ValueError(f"need at least m=2 groups; got m={m}")
    if n < 2:
        raise ValueError(f"need at least n=2 samples per group; got n={n}")
    if not np.all(np.isfinite(groups)):
        raise ValueError("groups contains non-finite values (nan/inf)")
    if method != "det":
        raise ValueError(f"method must be 'det'; got {method!r}")

    S, B_over_n = _within_between(groups)

    # p-th root of det(Sigma_hat) / det(S), where
    #   Sigma_hat = (n-1)/n * S + (m+1)/m * B/n.
    Sigma_hat = ((n - 1) / n) * S + ((m + 1) / m) * B_over_n
    ratio = _det_ratio(Sigma_hat, S, ridge=ridge)
    ratio = max(ratio, 1.0)  # Sigma_hat >= S (PSD), so ratio >= 1 up to noise
    return float(ratio ** (1.0 / p))


def _within_between(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Within-group covariance ``S`` and between-group covariance ``B/n``."""
    m, n, _ = groups.shape

    group_means = groups.mean(axis=1)              # x_bar_i, (m, p)
    overall_mean = group_means.mean(axis=0)        # x_bar,   (p,)

    # S = (1 / m) Σ_i S_i with S_i the ddof=1 sample covariance of group i.
    #   = 1 / (m (n-1)) ΣΣ (x_{i,t}-x̄_i)(x_{i,t}-x̄_i)^T.
    centered = groups - group_means[:, None, :]    # (m, n, p)
    S = np.einsum("itk,itl->kl", centered, centered) / (m * (n - 1))

    # B/n = ddof=1 covariance of the group means.
    mean_dev = group_means - overall_mean          # (m, p)
    B_over_n = (mean_dev.T @ mean_dev) / (m - 1)    # (p, p)
    return S, B_over_n


def _regularized(S: np.ndarray, ridge: float) -> np.ndarray:
    """Return ``S`` with a ridge added iff it is ill-conditioned."""
    p = S.shape[0]
    try:
        if np.linalg.cond(S) > _COND_THRESHOLD:
            return S + ridge * np.eye(p)
    except np.linalg.LinAlgError:
        return S + ridge * np.eye(p)
    return S


def _det_ratio(Sigma_hat: np.ndarray, S: np.ndarray, *, ridge: float) -> float:
    """``det(Sigma_hat) / det(S)`` via generalized eigenvalues of (Sigma_hat, S).

    The product of the generalized eigenvalues of ``(Sigma_hat, S)`` equals the
    determinant ratio, and solving the symmetric GEVP is numerically more stable
    than forming each determinant separately when ``S`` is small/elongated.
    """
    p = S.shape[0]
    S_solve = _regularized(S, ridge)
    try:
        eigvals = eigh(Sigma_hat, S_solve, eigvals_only=True)
    except np.linalg.LinAlgError:
        eigvals = eigh(Sigma_hat, S_solve + ridge * np.eye(p), eigvals_only=True)
    eigvals = np.clip(eigvals, 1e-300, None)       # guard log/product underflow
    return float(np.prod(eigvals))


# ---------------------------------------------------------------------------
# Wiring: diagnose a MultivariateSamplePrediction's ensemble of draws
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SampleDiagnostic:
    """Convergence diagnostic for an ensemble of predictive draws.

    Attributes
    ----------
    rhat : float
        Median (over test instances) of the multivariate :math:`\\hat{R}^p`.
    rhat_max : float
        Worst-case (max over test instances) :math:`\\hat{R}^p`.
    frac_converged : float
        Fraction of test instances with :math:`\\hat{R}^p < \\varepsilon`.
    n_draws : int
        Number of draws per test instance that were available (``m``).
    method : str
        Which statistic was used (always ``"det"``).
    epsilon : float
        The threshold used for ``frac_converged``.
    """

    rhat: float
    rhat_max: float
    frac_converged: float
    n_draws: int
    method: str
    epsilon: float


def assess_sample_prediction(
    pred,
    *,
    n_groups: int = 4,
    method: Method = "det",
    epsilon: float = 1.01,
    ridge: float = _DEFAULT_RIDGE,
    max_instances: int | None = 512,
    rng: np.random.Generator | int | None = None,
) -> SampleDiagnostic:
    """MPSRF diagnostic for a :class:`MultivariateSamplePrediction`.

    Splits each test instance's ``m`` i.i.d. draws into ``n_groups`` equal
    groups and computes :math:`\\hat{R}^p` per instance, then aggregates over
    instances.  Because the draws are i.i.d. (not a Markov chain) the split is a
    *random* partition — order carries no information — so we shuffle the draw
    axis before splitting.

    Parameters
    ----------
    pred : MultivariateSamplePrediction
        Draws of shape ``(n_test, m, p)``.
    n_groups : int, default 4
        Number of groups to split the ``m`` draws into (``>= 2``).
    method : {"det"}, default "det"
        Statistic passed to :func:`compute_multivariate_rhat`.
    epsilon : float, default 1.01
        Convergence threshold for ``frac_converged``.
    ridge : float, default 1e-8
        Ridge forwarded to :func:`compute_multivariate_rhat`.
    max_instances : int or None, default 512
        If set and ``n_test`` exceeds it, evaluate on a random subset of this
        many instances (the diagnostic is a summary statistic, so a subsample
        is sufficient and keeps the cost negligible).  ``None`` uses all.
    rng : Generator, int, or None
        Randomness for shuffling draws and subsampling instances.

    Returns
    -------
    SampleDiagnostic
    """
    samples = np.asarray(pred.samples, dtype=np.float64)   # (n_test, m, p)
    n_test, m, p = samples.shape
    if n_groups < 2:
        raise ValueError(f"need at least n_groups=2; got {n_groups}")
    n_per = m // n_groups
    if n_per < 2:
        raise ValueError(
            f"not enough draws to form {n_groups} groups of >=2 from m={m}; "
            f"reduce n_groups or increase the number of draws"
        )

    generator = np.random.default_rng(rng)

    idx = np.arange(n_test)
    if max_instances is not None and n_test > max_instances:
        idx = generator.choice(n_test, size=max_instances, replace=False)

    rhats = np.empty(idx.size, dtype=np.float64)
    usable = n_groups * n_per
    for out_i, i in enumerate(idx):
        draws = samples[i]                                 # (m, p)
        perm = generator.permutation(m)[:usable]
        grouped = draws[perm].reshape(n_groups, n_per, p)  # (M, n, p)
        rhats[out_i] = compute_multivariate_rhat(grouped, method=method, ridge=ridge)

    return SampleDiagnostic(
        rhat=float(np.median(rhats)),
        rhat_max=float(np.max(rhats)),
        frac_converged=float(np.mean(rhats < epsilon)),
        n_draws=m,
        method=method,
        epsilon=epsilon,
    )


# ---------------------------------------------------------------------------
# Adaptive sampling loop (dynamic stopping) — no warm-up, keep all samples
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConvergenceInfo:
    """Outcome of :func:`adaptive_sample_until_converged`.

    Attributes
    ----------
    converged : bool
        Whether ``rhat < epsilon`` was reached before ``max_samples``.
    n_samples : int
        Final number of samples per group used for the last statistic.
    rhat : float
        The final :math:`\\hat{R}^p` value.
    method : str
        Which statistic was used (always ``"det"``).
    history : list[tuple[int, float]]
        ``(n_samples, rhat)`` recorded at every evaluation checkpoint.
    """

    converged: bool
    n_samples: int
    rhat: float
    method: str
    history: list[tuple[int, float]]


def adaptive_sample_until_converged(
    sample_dgp: Callable[[int, int, int], np.ndarray],
    p: int,
    *,
    n_groups: int = 4,
    n_start: int = 1000,
    batch_size: int = 500,
    method: Method = "det",
    epsilon: float = 1.01,
    max_samples: int = 100_000,
    ridge: float = _DEFAULT_RIDGE,
    verbose: bool = True,
) -> tuple[np.ndarray, ConvergenceInfo]:
    """Draw from ``sample_dgp`` until :math:`\\hat{R}^p < \\varepsilon`.

    Dynamic stopping: start with ``n_start`` draws per group, then draw
    ``batch_size`` more draws for every group each iteration and re-evaluate
    :math:`\\hat{R}^p` until it drops below ``epsilon`` (or ``max_samples`` is
    exhausted).  Since the draws are i.i.d. Monte-Carlo samples, **no warm-up is
    discarded** — every draw contributes to the estimate.

    Parameters
    ----------
    sample_dgp : callable ``(n_groups, batch_size, p) -> (n_groups, batch_size, p)``
        The data-generating process.  Must return a fresh block of
        ``batch_size`` i.i.d. draws for each of the ``n_groups`` groups.  The
        groups must be drawn *independently*.
    p : int
        Target vector dimension.
    n_groups : int, default 4
        Number of parallel groups (``>= 2``; 4 recommended for a stable
        between-group covariance estimate).
    n_start : int, default 1000
        Initial draws per group before the first check.
    batch_size : int, default 500
        ``Δn`` draws added per group per iteration.
    method : {"det"}, default "det"
        Statistic passed to :func:`compute_multivariate_rhat`.
    epsilon : float, default 1.01
        Stopping threshold on :math:`\\hat{R}^p` (1.01 strict, 1.05 loose).
    max_samples : int, default 100_000
        Hard cap on total draws per group (safety valve).
    ridge : float, default 1e-8
        Ridge forwarded to :func:`compute_multivariate_rhat`.
    verbose : bool, default True
        Print ``(n, rhat)`` at every checkpoint.

    Returns
    -------
    groups : (m, n, p) array
        All draws from every group at the stopping point.
    info : ConvergenceInfo
    """
    if n_groups < 2:
        raise ValueError(f"need at least n_groups=2; got {n_groups}")

    groups = np.asarray(sample_dgp(n_groups, n_start, p), dtype=np.float64)
    _validate_dgp_block(groups, n_groups, n_start, p)

    history: list[tuple[int, float]] = []
    rhat = float("inf")

    while True:
        n = groups.shape[1]
        rhat = compute_multivariate_rhat(groups, method=method, ridge=ridge)
        history.append((n, rhat))
        if verbose:
            print(f"  n={n:>7d}  Rhat^p({method})={rhat:.6f}")

        if rhat < epsilon:
            return groups, ConvergenceInfo(
                converged=True, n_samples=n, rhat=rhat,
                method=method, history=history,
            )

        if n >= max_samples:
            if verbose:
                print(f"  reached max_samples={max_samples} without "
                      f"Rhat^p < {epsilon} (last={rhat:.6f})")
            return groups, ConvergenceInfo(
                converged=False, n_samples=n, rhat=rhat,
                method=method, history=history,
            )

        extra = np.asarray(sample_dgp(n_groups, batch_size, p), dtype=np.float64)
        _validate_dgp_block(extra, n_groups, batch_size, p)
        groups = np.concatenate([groups, extra], axis=1)


def _validate_dgp_block(block: np.ndarray, m: int, n: int, p: int) -> None:
    """Validate the shape of a block returned by ``sample_dgp``."""
    if block.shape != (m, n, p):
        raise ValueError(
            f"sample_dgp must return shape (n_groups, batch_size, p)="
            f"({m}, {n}, {p}); got {block.shape}"
        )
    if not np.all(np.isfinite(block)):
        raise ValueError("sample_dgp returned non-finite values (nan/inf)")
