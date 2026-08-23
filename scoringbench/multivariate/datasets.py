"""Multivariate dataset construction (Source 1: feature promotion).

We reuse the univariate ScoringBench dataset plumbing (download, cache,
preprocess, validation) *only* to obtain a raw ``(X, y_1d)`` regression
problem. We then turn it into a ``d``-dimensional target problem by
**promoting the (d-1) feature columns that carry the strongest conditional
(residual) cross-target dependence into targets as well**. The original 1-D
target becomes the first target dimension, and the promoted feature columns
become the remaining ``d-1`` dimensions; those columns are removed from the
feature matrix ``X``.

Rationale
---------
Standard tabular regression benchmarks are univariate. To obtain *dependent*
multivariate targets we must move feature columns into the target block — but
*which* columns matters. Merely promoting the features most *marginally*
correlated with ``y`` is a poor proxy: two targets can each correlate with
``y`` yet be *conditionally independent* given ``X``, in which case an X-only
marginal (independent) model already captures everything and the joint models
have nothing left to recover.

Instead we select the promoted columns to maximise the **residual dependence**
of the target block. For a candidate target set ``S`` we residualise every
target against the *remaining* features ``X \\ S`` (linear least squares) and
measure the mean absolute **Spearman** rank correlation among those residuals.
Greedily maximising this quantity picks targets whose dependence survives
conditioning on ``X`` — exactly the structure a copula (via the vine on the
PITs) or a chain (via ``p(y1|x) p(y2|x,y1)``) can recover but an independent
product-of-marginals model structurally cannot. This makes the benchmark
actually reward multivariate modelling.

Spearman (rank) correlation is used rather than Pearson so the criterion is
invariant to monotone marginal transforms — consistent with the copula view,
where dependence lives in the ranks (the PITs) and not the marginals.

Only the *loading* is borrowed from :mod:`scoringbench.univariate.datasets`.
No model / wrapper / metric code is shared between the two subpackages.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from scipy.stats import rankdata

# Raw-data plumbing only (download / cache / preprocess / validate).
from scoringbench.univariate.datasets import (
    get_DATASETS_CONFIG as _uni_get_DATASETS_CONFIG,
    load_dataset as _uni_load_dataset,
    validate_datasets as _uni_validate_datasets,
)

from . import config as cfg


# ---------------------------------------------------------------------------
# Config passthrough
# ---------------------------------------------------------------------------

def get_DATASETS_CONFIG():
    """Return the shared ScoringBench dataset configuration list."""
    return _uni_get_DATASETS_CONFIG()


def validate_datasets(datasets_config):
    """Filter out non-regression datasets (delegates to univariate validation)."""
    return _uni_validate_datasets(datasets_config)


# ---------------------------------------------------------------------------
# Feature promotion (residual-Spearman conditional-dependence selection)
# ---------------------------------------------------------------------------

def _numeric_matrix(X: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """Coerce ``X`` to a float64 matrix, dropping non-numeric/constant/all-NaN
    columns and replacing any residual NaNs with the column mean.

    Returns ``(usable_column_names, matrix)`` where ``matrix`` has one column
    per usable name in the same order.
    """
    cols: list[str] = []
    arrays: list[np.ndarray] = []
    for col in X.columns:
        arr = pd.to_numeric(X[col], errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(arr)
        if finite.sum() < 3:
            continue
        fin_vals = arr[finite]
        if np.std(fin_vals) == 0.0:
            continue
        if not finite.all():
            arr = arr.copy()
            arr[~finite] = float(np.mean(fin_vals))
        cols.append(col)
        arrays.append(arr)
    mat = np.column_stack(arrays) if arrays else np.empty((len(X), 0))
    return cols, mat


# Per-selection-run memoization of residuals.  The greedy selector trials many
# (target, feature-set) combinations, and the same combination recurs across
# candidate loops within and across steps; caching the residual vector avoids
# re-fitting the (relatively expensive) XGBoost regressor for identical inputs.
# Keyed by (SHA1 of target bytes, SHA1 of X_rest bytes, X_rest shape); cleared
# at the start of every ``_select_residual_spearman_features`` call so it never
# leaks state between datasets.
_RESID_CACHE: dict[tuple, np.ndarray] = {}


def _resid_cache_key(target: np.ndarray, X_rest: np.ndarray) -> tuple:
    ht = hashlib.sha1(np.ascontiguousarray(target, dtype=np.float64).tobytes()).hexdigest()
    hx = hashlib.sha1(np.ascontiguousarray(X_rest, dtype=np.float64).tobytes()).hexdigest()
    return (ht, hx, X_rest.shape)


def _residualize_linear(target: np.ndarray, X_rest: np.ndarray) -> np.ndarray:
    """Fast OLS residualisation (intercept included). Used when p is large."""
    A = np.column_stack([np.ones(len(target)), X_rest])
    coef, _, _, _ = np.linalg.lstsq(A, target, rcond=None)
    return target - A @ coef


def _residualize(target: np.ndarray, X_rest: np.ndarray) -> np.ndarray:
    """Residuals of ``target`` after removing its conditional mean given ``X_rest``.

    **Fast path (linear):** when the number of conditioning features exceeds
    ``cfg.RESIDUALIZER_LINEAR_FALLBACK_THRESHOLD`` (default 50) we fall back to
    plain OLS.  The greedy loop already pre-screens candidates (see
    ``_select_residual_spearman_features``), so by the time we reach the greedy
    stage the feature matrix is already small; the linear path is mainly a
    safety net for unusually wide datasets.

    **Row subsampling:** when ``n > cfg.RESIDUALIZER_MAX_ROWS`` (default 2000)
    the model is fit on a random subsample but residuals are computed for *all*
    rows (predict on full data), keeping the Spearman ranks representative.

    With no conditioning features the residual is just the mean-centred target
    (there is nothing to regress on).  Results are memoized per selection run
    via :data:`_RESID_CACHE`.
    """
    if X_rest.shape[1] == 0:
        return target - float(np.mean(target))

    key = _resid_cache_key(target, X_rest)
    cached = _RESID_CACHE.get(key)
    if cached is not None:
        return cached

    n_features = X_rest.shape[1]
    linear_threshold = int(getattr(cfg, "RESIDUALIZER_LINEAR_FALLBACK_THRESHOLD", 50))
    use_linear = n_features > linear_threshold

    if use_linear:
        resid = _residualize_linear(target, X_rest)
    else:
        from xgboost import XGBRegressor

        n = len(target)
        max_rows = int(getattr(cfg, "RESIDUALIZER_MAX_ROWS", 2000))
        if n > max_rows:
            # Deterministic per-input subsample seed: derived from the content
            # hash of (target, X_rest) so the same combination always draws the
            # same rows (reproducible) but different combinations differ.  Uses
            # the same SHA1 as the cache key, so no NaN/overflow sensitivity.
            seed = int(_resid_cache_key(target, X_rest)[0][:8], 16)
            rng = np.random.default_rng(seed=seed)
            idx = rng.choice(n, size=max_rows, replace=False)
            X_fit, y_fit = X_rest[idx], target[idx]
        else:
            X_fit, y_fit = X_rest, target

        model = XGBRegressor(
            n_estimators=int(getattr(cfg, "RESIDUALIZER_N_ESTIMATORS", 100)),
            max_depth=int(getattr(cfg, "RESIDUALIZER_MAX_DEPTH", 4)),
            learning_rate=float(getattr(cfg, "RESIDUALIZER_LEARNING_RATE", 0.3)),
            subsample=float(getattr(cfg, "RESIDUALIZER_SUBSAMPLE", 1.0)),
            tree_method="hist",
            # nthread=1: the residualizer is called in a tight greedy loop;
            # spawning a full thread pool per fit is far more expensive than
            # running each fit single-threaded (benchmarked: ~40x slower with
            # nthread=-1 on this machine due to thread-pool startup overhead).
            nthread=1,
            verbosity=0,
        )
        model.fit(X_fit, y_fit)
        pred = np.asarray(model.predict(X_rest), dtype=np.float64).reshape(-1)
        resid = target - pred

    _RESID_CACHE[key] = resid
    return resid


def _abs_spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Absolute Spearman rank correlation between two vectors (0 if degenerate)."""
    ra = rankdata(a)
    rb = rankdata(b)
    if np.std(ra) == 0.0 or np.std(rb) == 0.0:
        return 0.0
    c = np.corrcoef(ra, rb)[0, 1]
    return 0.0 if not np.isfinite(c) else abs(float(c))


def _mean_offdiag_residual_spearman(
    targets: list[np.ndarray], X_rest: np.ndarray
) -> float:
    """Mean absolute off-diagonal Spearman among residuals of ``targets``.

    Each target is residualised against ``X_rest`` (the features *not* in the
    candidate target set) before ranks are taken.
    """
    resids = [_residualize(t, X_rest) for t in targets]
    m = len(resids)
    if m < 2:
        return 0.0
    vals = [
        _abs_spearman(resids[i], resids[j])
        for i in range(m)
        for j in range(i + 1, m)
    ]
    return float(np.mean(vals)) if vals else 0.0


def _select_residual_spearman_features(
    X: pd.DataFrame, y: pd.Series, n_promote: int
) -> list[str]:
    """Greedily choose ``n_promote`` feature columns maximising conditional
    (residual) cross-target Spearman dependence.

    ``target_0`` is fixed to ``y``. At each step every not-yet-selected usable
    column is trialled as the next target; the target block is residualised
    against all remaining (non-selected) features and scored by the mean
    absolute off-diagonal residual Spearman. The best-scoring column is added.

    **Pre-screening (scalability):** when the number of usable feature columns
    exceeds ``cfg.RESIDUALIZER_PRESCREENING_KEEP`` (default
    ``max(20, 3 * n_promote)``), the candidate pool is first narrowed to the
    top-``k`` columns by *marginal* absolute Spearman correlation with ``y``.
    This is a cheap O(p) pass that keeps the expensive O(k² · n_promote)
    greedy XGBoost loop tractable even for p > 100.

    Falls back to original column order for any padding needed when fewer than
    ``n_promote`` usable columns exist (guarded by the caller, but kept safe).
    """
    # Clear memoization cache so residuals from a previous dataset never leak.
    _RESID_CACHE.clear()

    y_arr = pd.to_numeric(y, errors="coerce").to_numpy(dtype=np.float64)
    usable_cols, mat = _numeric_matrix(X)
    col_index = {c: k for k, c in enumerate(usable_cols)}

    # --- Pre-screening: reduce candidate pool for wide datasets ---------------
    prescreening_keep = int(
        getattr(cfg, "RESIDUALIZER_PRESCREENING_KEEP", max(20, 3 * n_promote))
    )
    if len(usable_cols) > prescreening_keep:
        ry = rankdata(y_arr)
        marginal_scores = [
            abs(float(np.corrcoef(ry, rankdata(mat[:, col_index[c]]))[0, 1]))
            if np.std(mat[:, col_index[c]]) > 0 else 0.0
            for c in usable_cols
        ]
        top_k_idx = np.argsort(marginal_scores)[::-1][:prescreening_keep]
        usable_cols = [usable_cols[i] for i in top_k_idx]
        # Rebuild col_index for the reduced set (indices still point into mat).
        col_index = {c: col_index[c] for c in usable_cols}
    # --------------------------------------------------------------------------

    # Indices (into mat) of columns excluded by pre-screening.  These are never
    # candidates but must stay in X_rest so residualisation conditions on the
    # full feature matrix, not just the screened subset.  Computed once here to
    # avoid calling _numeric_matrix inside the hot inner loop.
    usable_set = set(usable_cols)
    all_cols_ordered, _ = _numeric_matrix(X)
    screened_out_idx = [k for k, c2 in enumerate(all_cols_ordered) if c2 not in usable_set]

    selected: list[str] = []
    while len(selected) < n_promote and len(selected) < len(usable_cols):
        best_col: str | None = None
        best_score = -np.inf
        for c in usable_cols:
            if c in selected:
                continue
            cand_set = set(selected) | {c}
            # Screened-in cols not in the candidate target set stay in X_rest.
            rest_idx = [col_index[u] for u in usable_cols if u not in cand_set]
            full_rest_idx = rest_idx + screened_out_idx
            X_rest = mat[:, full_rest_idx] if full_rest_idx else np.empty((len(y_arr), 0))
            targets = [y_arr] + [mat[:, col_index[s]] for s in selected] + [mat[:, col_index[c]]]
            score = _mean_offdiag_residual_spearman(targets, X_rest)
            # Stable tie-break: keep the earliest column (loop order == X order).
            if score > best_score:
                best_score = score
                best_col = c
        if best_col is None:
            break
        selected.append(best_col)
    return selected


def promote_features_to_targets(
    X: pd.DataFrame,
    y: pd.Series,
    target_dim: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a ``target_dim``-dimensional target block from a 1-D problem.

    The original target ``y`` becomes column ``0`` of ``Y``. The ``target_dim-1``
    feature columns carrying the strongest *conditional (residual) Spearman*
    cross-target dependence are moved out of ``X`` and become the remaining
    target columns (in greedy-selection order). Returns ``(X_reduced, Y)`` where
    ``Y`` has exactly ``target_dim`` columns.

    Raises ``ValueError`` if ``X`` does not have enough columns to promote.
    """
    d = int(target_dim)
    if d < 1:
        raise ValueError(f"target_dim must be >= 1, got {d}")

    X = pd.DataFrame(X).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    y.name = "target_0"

    n_promote = d - 1
    # Promotion moves ``n_promote`` feature columns out of X into Y. Require at
    # least one feature to REMAIN in X afterwards, otherwise models get an empty
    # design matrix (shape (n, 0)) and crash. Using ``>=`` (instead of ``>``)
    # reserves that residual feature: a dataset needs >= d columns to qualify.
    if n_promote >= X.shape[1]:
        raise ValueError(
            f"Cannot build a {d}-dimensional target: need {n_promote} feature "
            f"columns to promote and >=1 to keep in X, but X only has "
            f"{X.shape[1]}."
        )

    if n_promote == 0:
        Y = y.to_frame()
        return X, Y

    promoted_cols = _select_residual_spearman_features(X, y, n_promote)
    if len(promoted_cols) < n_promote:
        raise ValueError(
            f"Cannot build a {d}-dimensional target: only {len(promoted_cols)} "
            f"usable (numeric, non-constant) feature columns available to "
            f"promote but {n_promote} required."
        )

    Y = pd.DataFrame({"target_0": y.to_numpy(dtype=np.float64)})
    for j, col in enumerate(promoted_cols, start=1):
        Y[f"target_{j}"] = pd.to_numeric(X[col], errors="coerce").to_numpy(dtype=np.float64)

    X_reduced = X.drop(columns=promoted_cols).reset_index(drop=True)
    return X_reduced, Y


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_multivariate_dataset(
    dataset_config: dict,
    target_dim: int = int(cfg.TARGET_DIM),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a dataset and return ``(X, Y)`` with a ``target_dim``-dim target.

    ``X`` is a DataFrame of remaining features; ``Y`` is a DataFrame with
    ``target_dim`` numeric columns (``target_0`` is the original 1-D target).
    """
    X, y = _uni_load_dataset(dataset_config)
    X = pd.DataFrame(X)
    y = pd.Series(y)

    # Drop rows with a NaN in the (numeric) target before promotion.
    y = pd.to_numeric(y, errors="coerce")
    mask = y.notna().to_numpy()
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)

    return promote_features_to_targets(X, y, target_dim)
