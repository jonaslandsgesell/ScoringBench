"""Model-free conditional-dependence screen for multi-target datasets.

A joint model can only beat a product of marginals when the targets stay
dependent *after* conditioning on the inputs.  The screen therefore tests

    H0:  Q(Y_1, ..., Y_d | X) = prod_j Q(Y_j | X)

via the chain of block hypotheses ``H0_j: Y_j ⊥ (Y_1..Y_{j-1}) | X`` for
``j = 2..d`` (jointly equivalent to H0; pairwise tests would miss a purely
three-way interaction), using the kernel conditional independence test (KCI,
Zhang et al. 2011).  The ``d-1`` block p-values are Bonferroni-combined into one
task p-value.  Because ``H0_j`` only involves the first ``j`` targets, the block
p-values computed once for the longest target prefix serve every evaluated
dimension: ``p_task(d) = task_pvalue(p[:d-1])``.

Following omnibus significance, the mean pairwise residual distance correlation
serves as a robust non-parametric effect-size proxy to guard against large-n
p-value saturation.

Screening runs on a fixed curation split that the benchmark must not evaluate on.
"""

from __future__ import annotations

import numpy as np

ALPHA = 0.05
EFFECT_MIN = 0.175
CURATION_FRACTION = 0.2
CURATION_CAP = 1000
SCALE_RATIO_MAX = 500.0


def curation_split(n: int, seed: int = 0) -> np.ndarray:
    """Sorted row indices of the fixed curation split: 20 % of rows, capped at 1000."""
    k = min(int(round(CURATION_FRACTION * n)), CURATION_CAP)
    return np.sort(np.random.default_rng(seed).choice(n, size=k, replace=False))


def kci_pvalue(y: np.ndarray, block: np.ndarray, x: np.ndarray) -> float:
    """KCI p-value for ``y ⊥ block | x`` (Gaussian kernels, causal-learn defaults).

    Calibration can fail on wide conditioning sets. This is a documented
    limitation, not an exclusion rule: all variable input features are used.
    """
    from causallearn.utils.KCI.KCI import KCI_CInd

    return float(KCI_CInd().compute_pvalue(_2d(y), _2d(block), _2d(x))[0])


def block_pvalues(Y: np.ndarray, X: np.ndarray, test=kci_pvalue) -> list[float]:
    """``[p_2, ..., p_d]`` with ``p_j`` testing ``Y_j ⊥ Y_{<j} | X`` in column order of ``Y``."""
    return [test(Y[:, j], Y[:, :j], X) for j in range(1, Y.shape[1])]


def task_pvalue(block_p) -> float:
    """Bonferroni-combined task p-value ``min(1, (d-1) * min_j p_j)``."""
    if not block_p:
        return 1.0
    return float(min(1.0, len(block_p) * min(block_p)))


def leaky_features(X: np.ndarray, Y: np.ndarray, r2: float = 0.999) -> np.ndarray:
    """Mask of features that are (near-)linear functions of the targets, e.g. their sum."""
    if len(Y) < max(20, 5 * (Y.shape[1] + 1)):
        return np.zeros(X.shape[1], dtype=bool)
    A = np.c_[np.ones(len(Y)), Y]
    resid = X - A @ np.linalg.lstsq(A, X, rcond=None)[0]
    var = X.var(axis=0)
    return (var > 0) & (resid.var(axis=0) <= (1 - r2) * var)


def _2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return a.reshape(len(a), -1)


# ---------------------------------------------------------------------------
# Model-free effect size: mean off-diagonal residual distance correlation
# ---------------------------------------------------------------------------

def _u_centered(v: np.ndarray) -> np.ndarray:
    a = np.abs(v[:, None] - v[None, :])
    n = a.shape[0]
    A = (a - a.sum(1, keepdims=True) / (n - 2) - a.sum(0, keepdims=True) / (n - 2)
         + a.sum() / ((n - 1) * (n - 2)))
    np.fill_diagonal(A, 0.0)
    return A


def distance_correlation(a: np.ndarray, b: np.ndarray, max_rows: int = 1500, seed: int = 0) -> float:
    """Sample distance correlation between 1D arrays ``a`` and ``b`` (Székely & Rizzo 2014)."""
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    if a.size > max_rows:
        idx = np.random.default_rng(seed).choice(a.size, max_rows, replace=False)
        a, b = a[idx], b[idx]
    n = a.size
    if n < 5 or a.std() == 0 or b.std() == 0:
        return 0.0
    A, B = _u_centered(a), _u_centered(b)
    dvar = lambda M, N: float((M * N).sum() / (n * (n - 3)))
    va, vb = dvar(A, A), dvar(B, B)
    return float(np.sqrt(max(dvar(A, B) / np.sqrt(va * vb), 0.0))) if va > 0 and vb > 0 else 0.0


def residualize_oof(y: np.ndarray, X: np.ndarray, seed: int = 0) -> np.ndarray:
    """Cross-fitted residuals of ``y`` on ``X`` (RidgeCV for wide X, else gradient boosting)."""
    from sklearn.model_selection import KFold

    y = np.asarray(y, float).ravel()
    X = np.asarray(X, float)
    if X.shape[1] == 0:
        return y - y.mean()
    resid = np.empty_like(y)
    n_splits = min(5, max(2, len(y) // 10))
    for tr, te in KFold(n_splits, shuffle=True, random_state=seed).split(X):
        if X.shape[1] > 50:
            from sklearn.linear_model import RidgeCV
            m = RidgeCV(alphas=np.logspace(-3, 3, 7)).fit(X[tr], y[tr])
            resid[te] = y[te] - m.predict(X[te])
        else:
            from sklearn.ensemble import HistGradientBoostingRegressor
            m = HistGradientBoostingRegressor(max_iter=100, random_state=seed).fit(X[tr], y[tr])
            resid[te] = y[te] - m.predict(X[te])
    return resid


def residual_distance_correlation(Y: np.ndarray, X: np.ndarray, max_rows: int = 1500, seed: int = 0) -> float:
    """Mean off-diagonal residual distance correlation across all pairs of targets in ``Y``."""
    Y = np.asarray(Y, float)
    X = np.asarray(X, float)
    d = Y.shape[1]
    if d < 2:
        return 0.0
    resids = [residualize_oof(Y[:, j], X, seed=seed + j) for j in range(d)]
    corrs = [
        distance_correlation(resids[j], resids[k], max_rows=max_rows, seed=seed)
        for j in range(d) for k in range(j + 1, d)
    ]
    return float(np.mean(corrs)) if corrs else 0.0


def screen_kci(X: np.ndarray, Y: np.ndarray, alpha: float = ALPHA,
               effect_min: float = EFFECT_MIN,
               scale_max: float = SCALE_RATIO_MAX, seed: int = 0) -> dict:
    """Screen explicit curation data using KCI, residual dcorr, and target scale.

    ``selected`` marks the core pool (standard-deviation ratio <= 500);
    ``scale_ablation`` marks dependent cases above that ratio. No rows are
    sampled here: callers must keep curation data separate from evaluation.
    Target dimension is Y.shape[1]; the number of conditioning features is not capped.
    """
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    if X.ndim != 2 or Y.ndim != 2 or len(X) != len(Y):
        raise ValueError("X and Y must be aligned two-dimensional arrays")
    if not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise ValueError("screening requires finite curation data")
    X = X[:, X.std(axis=0) > 0]
    result = dict(selected=False, scale_ablation=False, kci_p=None,
                  residual_dcor=None, scale_ratio=None, n_features=X.shape[1],
                  n_targets=Y.shape[1], block_pvalues=None, reason=None)
    if len(Y) < 20 or Y.shape[1] < 2:
        result["reason"] = "insufficient_curation_data"
        return result
    scales = Y.std(axis=0, ddof=1)
    if np.any(scales == 0):
        result["reason"] = "constant_target"
        return result
    result["scale_ratio"] = float(scales.max() / scales.min())
    if X.shape[1] == 0:
        result["reason"] = "no_conditioning_features"
        return result
    blocks = block_pvalues(Y, X, test=kci_pvalue)
    if not np.isfinite(blocks).all():
        raise ValueError("KCI returned a non-finite p-value")
    result.update(kci_p=task_pvalue(blocks), block_pvalues=blocks)
    if result["kci_p"] >= alpha:
        result["reason"] = "kci_not_significant"
        return result
    result["residual_dcor"] = residual_distance_correlation(Y, X, seed=seed)
    if not np.isfinite(result["residual_dcor"]):
        raise ValueError("residual dcorr returned a non-finite effect size")
    if result["residual_dcor"] < effect_min:
        result["reason"] = "small_effect"
        return result
    result["scale_ablation"] = result["scale_ratio"] > scale_max
    result["selected"] = not result["scale_ablation"]
    result["reason"] = "scale_ratio" if result["scale_ablation"] else "selected"
    return result
