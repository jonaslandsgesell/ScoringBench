"""Abstract per-dimension conditional sampler + shared linear-interp CDF machinery.

Every multivariate wrapper composes *univariate* conditional samplers, one per
target dimension.  A sampler exposes a small, model-agnostic interface::

    sampler.fit(X, y)                    # y is 1-D
    draws = sampler.sample(X_test, n)    # -> (n_test, n) float64
    u     = sampler.cdf(X_test, y)       # PIT: F_{Y|X_i}(y_i) -> (n_test,)
    y     = sampler.quantile(X_test, u)  # inverse CDF F^{-1}_{Y|X_i}(u_i)

The ``cdf`` / ``quantile`` pair is what makes the *copula* wrapper base-model
agnostic: it only needs conditional PITs and their inverse, regardless of how a
concrete model represents its predictive law.

Single source of truth: one piecewise-linear CDF per row
--------------------------------------------------------
Concrete samplers implement one primitive, :meth:`BaseSampler._row_cdf_grid`,
returning per-row CDF nodes ``(cdf, value)``.  From these the base class builds
**one** monotone *piecewise-linear* CDF ``F_i`` per row (nodes on
``cdf <-> value``).  Linear interpolation is used in **both** directions, which
makes ``quantile`` (``cdf -> value``) and ``cdf`` (``value -> cdf``) *exact
inverses of each other*: for a piecewise-linear monotone map, linear inversion
is the true analytic inverse, so the round-trip identity
``cdf(quantile(u)) == u`` holds by construction — no bisection, no PCHIP.

Everything is done in **PyTorch**, batched over rows, on CUDA when available:

* :meth:`quantile` — interpolate ``F_i^{-1}`` at the given levels.
* :meth:`sample`   — interpolate ``F_i^{-1}`` at uniform draws (inverse-CDF).
* :meth:`cdf`      — interpolate ``F_i`` at the given values.

All tails are clamped: query values are capped to the row's ``[v_min, v_max]``
support and probabilities to ``[0, 1]`` (flat CDF outside the node range).
"""

from __future__ import annotations

import numpy as np
import torch


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaseSampler:
    """Abstract univariate conditional sampler with a piecewise-linear CDF.

    Subclasses implement :meth:`fit`, :meth:`predict_mean`, and
    :meth:`_row_cdf_grid`.  Everything else (``cdf``, ``quantile``, ``sample``)
    is derived here from a single per-row monotone piecewise-linear CDF, built
    and evaluated in PyTorch (CUDA if available) so the three operations are
    mutually consistent and fast on large ``n_test * m`` batches.
    """

    #: dtype used for the interpolation math.  float64 keeps the copula PIT
    #: pseudo-observations numerically well-behaved (tail dependence is
    #: sensitive to rounding).
    _DTYPE = torch.float64

    @property
    def _torch_device(self) -> torch.device:
        """Device for the interpolation math (CUDA if available, else CPU)."""
        dev = getattr(self, "_device", None)
        if isinstance(dev, torch.device):
            return dev
        if isinstance(dev, str):
            return torch.device(dev)
        return _default_device()

    # -- interface a subclass must implement ----------------------------
    def fit(self, X, y) -> "BaseSampler":
        raise NotImplementedError

    def predict_mean(self, X) -> np.ndarray:
        raise NotImplementedError

    def _row_cdf_grid(self, X):
        """Return per-row CDF nodes as two lists/arrays ``(c_rows, v_rows)``.

        For each of the ``n_test`` rows, ``c_rows[i]`` is a vector of cumulative
        probabilities and ``v_rows[i]`` the matching values.  Subclasses build
        these from their native predictive representation (bar distribution,
        quantiles, …).  Rows may have *different* node counts; the base class
        pads them into a rectangular ``(n_test, K)`` tensor by repeating the
        last node (a repeated node is a zero-width segment, never selected by
        interpolation, so padding is inert).
        """
        raise NotImplementedError

    # -- optional shared-grid fast path (all rows share one value grid) --
    def _shared_grid_cdf(self, X):
        """Optional fast-path hook for samplers with a *shared* value grid.

        Return ``(values, cdf)`` where ``values`` is a single 1-D array of
        length ``K`` shared by every row and ``cdf`` is ``(n_test, K)`` (0 at
        the left edge, 1 at the right edge per row).  TabPFN's bar distribution
        fits this (the bar ``edges`` are identical for all rows), letting us
        skip per-row node assembly entirely.  Return ``None`` to fall back to
        :meth:`_row_cdf_grid`.
        """
        return None

    # -- build the padded (values, cdf) node tensors for a batch ---------
    def _node_tensors(self, X):
        """Return ``(values, cdf)`` tensors of shape ``(n_test, K)`` on device.

        Both are monotone non-decreasing along the last axis, with each row's
        CDF pinned to 0 at the first node and 1 at the last.  Flat CDF segments
        are left in place (zero-width in the ``cdf`` axis); the interpolation
        helper treats them as inert.
        """
        device, dtype = self._torch_device, self._DTYPE

        grid = self._shared_grid_cdf(X)
        if grid is not None:
            values, cdf = grid
            v = torch.as_tensor(np.asarray(values), dtype=dtype, device=device)
            c = torch.as_tensor(np.asarray(cdf), dtype=dtype, device=device)
            if v.ndim == 1:                       # (K,) shared -> (n_test, K)
                v = v.unsqueeze(0).expand(c.shape[0], -1).contiguous()
            c = self._sanitize_cdf(c)
            return v, c

        # General per-row path: pad ragged rows to a common width.
        c_rows, v_rows = self._row_cdf_grid(X)
        n_test = len(c_rows)
        K = max(len(c) for c in c_rows)
        v = torch.empty((n_test, K), dtype=dtype, device=device)
        c = torch.empty((n_test, K), dtype=dtype, device=device)
        for i in range(n_test):
            ci = np.asarray(c_rows[i], dtype=np.float64)
            vi = np.asarray(v_rows[i], dtype=np.float64)
            k = ci.shape[0]
            if k < K:                              # pad by repeating last node
                ci = np.concatenate([ci, np.full(K - k, ci[-1])])
                vi = np.concatenate([vi, np.full(K - k, vi[-1])])
            c[i] = torch.as_tensor(ci, dtype=dtype, device=device)
            v[i] = torch.as_tensor(vi, dtype=dtype, device=device)
        c = self._sanitize_cdf(c)
        return v, c

    @staticmethod
    def _sanitize_cdf(c: torch.Tensor) -> torch.Tensor:
        """Clip to ``[0,1]``, pin endpoints (0 left, 1 right), make monotone.

        Pinning *both* endpoints is a correctness requirement: some samplers
        report a CDF grid whose first level is > 0 (e.g. TabICL's quantile grid
        starts at ``alpha = 0.005``).  Leaving the first node at ``0.005`` while
        forcing the last to ``1.0`` makes the tails asymmetric — the PIT is
        floored at ``0.005`` on the left but reaches ``1.0`` on the right —
        which biases the vine's lower-tail dependence.  Pinning the first node
        to 0 restores left/right symmetry.
        """
        c = torch.clamp(c, 0.0, 1.0)
        c = c.clone()
        c[:, 0] = 0.0
        c[:, -1] = 1.0
        c = torch.cummax(c, dim=1).values          # enforce non-decreasing
        return c

    # -- batched piecewise-linear interpolation (the whole engine) -------
    @staticmethod
    def _interp_rows(xp: torch.Tensor, fp: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Per-row 1-D linear interpolation with clamped (flat) tails.

        Parameters
        ----------
        xp : (n_test, K) monotone non-decreasing abscissae per row.
        fp : (n_test, K) ordinates per row.
        x  : (n_test, m) query points per row.

        Returns ``(n_test, m)`` with ``x`` clamped to ``[xp[:,0], xp[:,-1]]``
        (both tails flat / clamped), then linearly interpolated between the two
        surrounding nodes.  Zero-width segments (padding or flat CDF steps) fall
        back to the left node.
        """
        K = xp.shape[1]
        lo = xp[:, :1]
        hi = xp[:, -1:]
        xc = torch.clamp(x, lo, hi)                # flat tails
        # Right node index j with xp[:, j-1] <= x < xp[:, j].
        idx = torch.searchsorted(xp, xc, right=True)
        idx = torch.clamp(idx, 1, K - 1)
        j0 = idx - 1
        x0 = torch.gather(xp, 1, j0)
        x1 = torch.gather(xp, 1, idx)
        f0 = torch.gather(fp, 1, j0)
        f1 = torch.gather(fp, 1, idx)
        denom = x1 - x0
        safe = torch.where(denom > 0, denom, torch.ones_like(denom))
        w = torch.where(denom > 0, (xc - x0) / safe, torch.zeros_like(denom))
        return f0 + w * (f1 - f0)

    # -- quantile / sample / cdf, all driven by the same linear F_i ------
    def quantile(self, X, u) -> np.ndarray:
        """Inverse CDF ``F_i^{-1}(u_i)`` at one level per row -> ``(n_test,)``."""
        v, c = self._node_tensors(X)
        u_t = torch.as_tensor(np.asarray(u, dtype=np.float64).ravel(),
                              dtype=self._DTYPE, device=self._torch_device)
        out = self._interp_rows(c, v, u_t.unsqueeze(1))[:, 0]   # cdf -> value
        return out.detach().cpu().numpy()

    def quantile_batch(self, X, U) -> np.ndarray:
        """Inverse CDF ``F_i^{-1}`` at *many* levels per row -> ``(n_test, m)``.

        Builds each unique row's conditional CDF **once** (a single model
        forward pass over the ``n_test`` distinct covariate rows), then inverts
        an ``(n_test, m)`` grid of levels against it with one vectorised
        interpolation.  This is the efficient way to draw ``m`` copula samples
        per test row: the predictive law depends only on the covariates, so
        there is no need to replicate rows ``m`` times through the model.

        Parameters
        ----------
        X : ``(n_test, p)`` distinct covariate rows.
        U : ``(n_test, m)`` uniform levels (one column per draw).

        Returns ``(n_test, m)`` inverted values, using the *same* per-row
        piecewise-linear CDF as :meth:`quantile` / :meth:`cdf` / :meth:`sample`.
        """
        v, c = self._node_tensors(X)                             # one fwd pass
        U_arr = np.asarray(U, dtype=np.float64)
        if U_arr.ndim == 1:
            U_arr = U_arr[:, None]
        u_t = torch.as_tensor(U_arr, dtype=self._DTYPE, device=self._torch_device)
        out = self._interp_rows(c, v, u_t)                       # cdf -> value
        return out.detach().cpu().numpy()

    def sample(self, X, n_samples: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Draw ``(n_test, n_samples)`` by piecewise-linear inverse-CDF sampling.

        Uses the *same* per-row CDF as :meth:`quantile` / :meth:`cdf`, so draws,
        PITs and inverse-PITs are all mutually consistent.  Uniforms are drawn
        with the provided numpy RNG (reproducible), then inverted on device.
        """
        v, c = self._node_tensors(X)
        n_test = c.shape[0]
        rng = np.random.default_rng() if rng is None else rng
        u_np = rng.random((n_test, n_samples))
        u_t = torch.as_tensor(u_np, dtype=self._DTYPE, device=self._torch_device)
        out = self._interp_rows(c, v, u_t)                       # cdf -> value
        return out.detach().cpu().numpy()

    def cdf(self, X, y) -> np.ndarray:
        """PIT ``F_i(y_i)`` at one value per row -> ``(n_test,)`` in ``[0,1]``.

        The *exact* inverse of :meth:`quantile`: both interpolate the same
        monotone piecewise-linear node set (one direction each), so
        ``cdf(quantile(u)) == u`` on the support by construction.
        """
        v, c = self._node_tensors(X)
        y_t = torch.as_tensor(np.asarray(y, dtype=np.float64).ravel(),
                             dtype=self._DTYPE, device=self._torch_device)
        out = self._interp_rows(v, c, y_t.unsqueeze(1))[:, 0]    # value -> cdf
        out = torch.clamp(out, 0.0, 1.0)
        return out.detach().cpu().numpy()
