"""Per-dimension conditional sampler backed by ``tabpfn.TabPFNRegressor``.

TabPFN exposes a *bar distribution*: a piecewise-constant density on
``criterion.borders`` with per-bar probabilities ``softmax(logits)``.  Its CDF
is exactly the cumulative bar mass at each right edge — a set of strictly
increasing ``(cdf, value)`` nodes on the bar grid.  We hand only those nodes to
:class:`BaseSampler`, which builds a single monotone-PCHIP quantile function
``Q_i`` per row and derives *all* of ``sample`` / ``cdf`` / ``quantile`` from
it.  There is deliberately **no** ``sample`` override here: keeping one PCHIP
inverse-CDF path guarantees draws, PITs, and inverse-PITs stay mutually
consistent (a separate within-bar sampler would draw from a piecewise-uniform
law that disagrees with the PCHIP ``cdf``/``quantile``).
"""

from __future__ import annotations

import sys

import numpy as np

from .base_sampler import BaseSampler


class TabPFNSampler(BaseSampler):
    """Conditional sampler over TabPFN's native bar distribution."""

    def __init__(self, device: str | None = None, **kwargs):
        import torch

        # Ensure the pip-installed tabpfn is used (mirrors the univariate wrapper).
        for k in list(sys.modules.keys()):
            if k.startswith("tabpfn"):
                sys.modules.pop(k)
        from tabpfn import TabPFNRegressor

        kwargs.pop("device", None)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        self._model = TabPFNRegressor(device=self._device, **kwargs)
        # Single-entry cache for the (expensive) TabPFN forward pass.  The bar
        # distribution depends only on the feature matrix X, but ``cdf`` /
        # ``quantile`` / ``sample`` / ``predict_mean`` all recompute it for the
        # *same* X (e.g. the copula wrapper PITs the training X and then, at
        # predict time, inverts the marginals on the replicated test X).  On CPU
        # one forward pass over a few hundred rows costs tens of seconds, so we
        # memoise the most recent result keyed on a cheap content fingerprint.
        # Bounded to ONE entry to keep memory small.
        self._bars_cache: tuple | None = None   # (key, (edges, probas))

    def fit(self, X, y) -> "TabPFNSampler":
        self._model.fit(X, np.asarray(y).ravel())
        self._bars_cache = None                 # invalidate on refit
        return self

    def predict_mean(self, X) -> np.ndarray:
        return np.asarray(self._model.predict(X), dtype=np.float64).ravel()

    @staticmethod
    def _cache_key(X) -> tuple:
        """Cheap, collision-resistant fingerprint of ``X`` for the bars cache.

        Uses shape + dtype + a SHA1 of the raw bytes of a contiguous float64
        view.  Hashing the bytes (rather than array identity) means a freshly
        built ``np.repeat(...)`` block that is value-identical still hits the
        cache, while any change in values misses.
        """
        import hashlib

        arr = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        h = hashlib.sha1(arr.view(np.uint8)).hexdigest()
        return (arr.shape, h)

    def _predictive_bars(self, X):
        """Return ``(edges (n_bins+1,), probas (n_test, n_bins))`` for the bar dist.

        Memoised on a single-entry cache keyed on the content of ``X`` so
        repeated calls on the same feature matrix reuse the TabPFN forward pass
        (the dominant CPU cost).
        """
        import torch

        key = self._cache_key(X)
        if self._bars_cache is not None and self._bars_cache[0] == key:
            return self._bars_cache[1]

        with torch.no_grad():
            pred_full = self._model.predict(X, output_type="full")
        logits = pred_full["logits"]
        if not isinstance(logits, torch.Tensor):
            logits = torch.as_tensor(logits)
        probas = torch.softmax(logits.float(), dim=-1).cpu().numpy()   # (n_test, n_bins)
        edges = pred_full["criterion"].borders.cpu().numpy().astype(np.float64)
        # Normalise per-row masses (softmax already sums to 1, but guard).
        totals = probas.sum(axis=1, keepdims=True)
        probas = np.where(totals > 0, probas / np.where(totals > 0, totals, 1.0),
                          1.0 / probas.shape[1])
        result = (edges, probas.astype(np.float64))
        self._bars_cache = (key, result)         # bounded to one entry
        return result

    # -- CDF grid for the base class (values vs cumulative prob) ---------
    def _row_cdf_grid(self, X):
        edges, probas = self._predictive_bars(X)          # edges: (n_bins+1,)
        cum = np.cumsum(probas, axis=1)                   # (n_test, n_bins)
        n_test = cum.shape[0]
        # CDF nodes on the bar grid: 0 at the left border, cumulative mass at
        # each right edge (values are the bar edges).
        c_rows, v_rows = [], []
        for i in range(n_test):
            c_rows.append(np.concatenate([[0.0], cum[i]]))
            v_rows.append(edges)
        return c_rows, v_rows

    # -- shared-grid fast path (edges are identical for every row) -------
    def _shared_grid_cdf(self, X):
        """Vectorized draw path: the bar ``edges`` are shared by all rows.

        Only the per-row cumulative masses differ, so we return the single
        shared ``edges`` grid plus an ``(n_test, n_bins+1)`` CDF (0 at the left
        border, cumulative mass at each right edge, clamped to 1).  This lets
        :class:`BaseSampler` invert every row's CDF with one batched
        piecewise-linear ``torch.searchsorted`` interpolation on GPU instead of
        a per-row Python loop — the dominant cost when sampling ``n_test * m``
        replicated rows.  Piecewise-linear inversion is the exact inverse of
        TabPFN's piecewise-linear bar CDF.
        """
        edges, probas = self._predictive_bars(X)          # edges: (n_bins+1,)
        cum = np.cumsum(probas, axis=1)                   # (n_test, n_bins)
        cdf = np.concatenate(
            [np.zeros((cum.shape[0], 1), dtype=np.float64), cum], axis=1
        )
        cdf = np.clip(cdf, 0.0, 1.0)
        cdf[:, -1] = 1.0
        cdf = np.maximum.accumulate(cdf, axis=1)
        return edges.astype(np.float64), cdf

    # NOTE: cdf (the PIT), quantile and sample are all inherited from
    # BaseSampler and share one batched piecewise-linear interpolation over the
    # shared bar grid, so the PIT and its inverse are exact inverses by
    # construction (round-trip ``cdf(quantile(u)) == u``).
