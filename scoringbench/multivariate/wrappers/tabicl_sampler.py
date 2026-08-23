"""Per-dimension conditional sampler backed by ``tabicl.TabICLRegressor``.

TabICL exposes predictive *quantiles* at levels ``alphas``.  The
``(alpha, quantile)`` pairs are exactly ``(cdf, value)`` CDF nodes, which we
hand to :class:`BaseSampler`; the base class builds a single monotone
piecewise-linear CDF per row and derives ``sample`` / ``cdf`` / ``quantile``
from it via batched PyTorch interpolation (consistent inverse-CDF sampling,
flat tails past the extreme levels).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .base_sampler import BaseSampler

# Prefer a local checkout of `tabicl` when present (mirrors univariate wrapper).
_repo_root = Path(__file__).resolve().parents[3]
_local_tabicl = _repo_root / "tabicl" / "src"
if _local_tabicl.exists() and str(_local_tabicl) not in sys.path:
    sys.path.insert(0, str(_local_tabicl))


class TabICLSampler(BaseSampler):
    """Conditional sampler over TabICL's predictive quantile function."""

    def __init__(self, alphas=None, **kwargs):
        from tabicl import TabICLRegressor

        self._model = TabICLRegressor(**kwargs)
        self._alphas = (
            np.asarray(alphas, dtype=np.float64)
            if alphas is not None
            else np.linspace(0.005, 0.995, 200)
        )
        # Single-entry cache for the (expensive) TabICL forward pass.  The
        # predictive quantiles depend only on X, but ``cdf`` / ``quantile`` /
        # ``sample`` all recompute them for the *same* X.  Memoise the most
        # recent result keyed on a cheap content fingerprint; bounded to ONE
        # entry to keep memory small.
        self._quant_cache: tuple | None = None   # (key, quantiles)

    def fit(self, X, y) -> "TabICLSampler":
        self._model.fit(X, np.asarray(y).ravel())
        self._quant_cache = None                 # invalidate on refit
        return self

    def predict_mean(self, X) -> np.ndarray:
        X_arr = np.asarray(X.values if hasattr(X, "values") else X)
        return np.asarray(self._model.predict(X_arr, output_type="mean"), dtype=np.float64).ravel()

    @staticmethod
    def _cache_key(X_arr: np.ndarray) -> tuple:
        """Cheap, collision-resistant fingerprint of ``X`` for the quantile cache."""
        import hashlib

        arr = np.ascontiguousarray(np.asarray(X_arr, dtype=np.float64))
        h = hashlib.sha1(arr.view(np.uint8)).hexdigest()
        return (arr.shape, h)

    def _predictive_quantiles(self, X) -> np.ndarray:
        """Return sorted ``(n_test, n_alphas)`` predicted quantiles.

        Memoised on a single-entry content-keyed cache so repeated calls on the
        same feature matrix reuse the TabICL forward pass (the dominant cost).
        """
        X_arr = np.asarray(X.values if hasattr(X, "values") else X)
        key = self._cache_key(X_arr)
        if self._quant_cache is not None and self._quant_cache[0] == key:
            return self._quant_cache[1]

        raw_q = self._model.predict(
            X_arr, output_type="quantiles", alphas=self._alphas.tolist()
        )
        if isinstance(raw_q, dict):
            raw_q = list(raw_q.values())[0]
        if isinstance(raw_q, list):
            q = np.vstack([np.asarray(r).ravel() for r in raw_q])
        else:
            q = np.asarray(raw_q, dtype=np.float64)
        if q.ndim == 1:
            q = q[None, :]
        if q.shape[1] != self._alphas.size and q.shape[0] == self._alphas.size:
            q = q.T
        q = np.sort(q, axis=1)                                          # monotone CDF
        self._quant_cache = (key, q)             # bounded to one entry
        return q

    # -- CDF grid for the base class (alphas are cumulative probs) -------
    def _row_cdf_grid(self, X):
        q = self._predictive_quantiles(X)                              # (n_test, K)
        n_test = q.shape[0]
        alphas = self._alphas                                          # (K,)
        c_rows = [alphas for _ in range(n_test)]
        v_rows = [q[i] for i in range(n_test)]
        return c_rows, v_rows
