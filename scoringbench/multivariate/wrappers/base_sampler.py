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

from typing import NamedTuple

import numpy as np
import torch


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CDFNodes(NamedTuple):
    """Per-row predictive law as a monotone piecewise-linear CDF on a grid.

    This is the one representation every sampler is reduced to (TabPFN bars,
    TabICL / Causilo quantile grids, …), so the shared ``cdf`` / ``quantile`` /
    ``sample`` / ``crps`` machinery is model-agnostic.  A piecewise-linear CDF on
    a grid is exactly a **histogram** (piecewise-constant density), so the two
    stored arrays carry the whole law and the mass views follow from them:

    * :attr:`edges` — ``(n_test, K)`` grid locations where the CDF is pinned, i.e.
      the histogram **bin edges** (non-decreasing along the last axis).  (These
      are TabPFN's bar borders / a quantile sampler's support points.)
    * :attr:`cdf` — ``(n_test, K)`` cumulative probability *at each edge*, with
      ``cdf[:, 0] = 0`` and ``cdf[:, -1] = 1``.
    * :attr:`pmf` — ``(n_test, K-1)`` probability **mass per bin**
      ``F(edgeᵢ₊₁) − F(edgeᵢ)`` (the TabPFN bar masses, on this grid).
    * :attr:`density` — ``(n_test, K-1)`` piecewise-constant density
      ``pmf / bin_width`` (0-width bins report 0).

    Fields are ``(edges, cdf)``, so it unpacks positionally as
    ``edges, cdf = nodes``.
    """

    edges: torch.Tensor
    cdf: torch.Tensor

    @property
    def pmf(self) -> torch.Tensor:
        """Per-bin probability mass ``F(edgeᵢ₊₁) − F(edgeᵢ)`` -> ``(n_test, K-1)``."""
        return self.cdf[:, 1:] - self.cdf[:, :-1]

    @property
    def density(self) -> torch.Tensor:
        """Piecewise-constant density ``pmf / bin_width`` -> ``(n_test, K-1)``."""
        width = self.edges[:, 1:] - self.edges[:, :-1]
        safe = torch.where(width > 0, width, torch.ones_like(width))
        return torch.where(width > 0, self.pmf / safe, torch.zeros_like(width))


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

    # -- build the padded node tensors for a batch ----------------------
    def _node_tensors(self, X) -> CDFNodes:
        """Return a :class:`CDFNodes` (``edges`` + ``cdf``) for ``X``.

        Both ``edges`` and ``cdf`` are ``(n_test, K)`` and monotone
        non-decreasing along the last axis, with each row's CDF pinned to 0 at
        the first edge and 1 at the last.  Flat CDF segments are left in place
        (zero-width in the ``cdf`` axis); the interpolation helper treats them as
        inert.  The result also exposes ``.pmf`` / ``.density`` views (see
        :class:`CDFNodes`), and still unpacks as ``edges, cdf = nodes``.
        """
        device, dtype = self._torch_device, self._DTYPE

        # Fast path: every row shares one edge grid (e.g. TabPFN's bar borders).
        grid = self._shared_grid_cdf(X)
        if grid is not None:
            grid_edges, grid_cdf = grid
            edges = torch.as_tensor(np.asarray(grid_edges), dtype=dtype, device=device)
            cdf = torch.as_tensor(np.asarray(grid_cdf), dtype=dtype, device=device)
            if edges.ndim == 1:                   # (K,) shared -> (n_test, K)
                edges = edges.unsqueeze(0).expand(cdf.shape[0], -1).contiguous()
            cdf = self._sanitize_cdf(cdf)
            return CDFNodes(edges=edges, cdf=cdf)

        # General per-row path: ragged (cdf, edge) node lists -> padded to a
        # common node count by repeating each row's last node (a zero-width bin,
        # inert under interpolation).
        cdf_rows, edge_rows = self._row_cdf_grid(X)
        n_test = len(cdf_rows)
        max_nodes = max(len(row) for row in cdf_rows)
        edges = torch.empty((n_test, max_nodes), dtype=dtype, device=device)
        cdf = torch.empty((n_test, max_nodes), dtype=dtype, device=device)
        for row in range(n_test):
            cdf_row = np.asarray(cdf_rows[row], dtype=np.float64)
            edge_row = np.asarray(edge_rows[row], dtype=np.float64)
            n_nodes = cdf_row.shape[0]
            if n_nodes < max_nodes:               # pad by repeating the last node
                pad = max_nodes - n_nodes
                cdf_row = np.concatenate([cdf_row, np.full(pad, cdf_row[-1])])
                edge_row = np.concatenate([edge_row, np.full(pad, edge_row[-1])])
            cdf[row] = torch.as_tensor(cdf_row, dtype=dtype, device=device)
            edges[row] = torch.as_tensor(edge_row, dtype=dtype, device=device)
        cdf = self._sanitize_cdf(cdf)
        return CDFNodes(edges=edges, cdf=cdf)

    @staticmethod
    def _sanitize_cdf(cdf: torch.Tensor) -> torch.Tensor:
        """Clip to ``[0,1]``, pin endpoints (0 left, 1 right), make monotone.

        Pinning *both* endpoints is a correctness requirement: some samplers
        report a CDF grid whose first level is > 0 (e.g. TabICL's quantile grid
        starts at ``alpha = 0.005``).  Leaving the first node at ``0.005`` while
        forcing the last to ``1.0`` makes the tails asymmetric — the PIT is
        floored at ``0.005`` on the left but reaches ``1.0`` on the right —
        which biases the vine's lower-tail dependence.  Pinning the first node
        to 0 restores left/right symmetry.
        """
        cdf = torch.clamp(cdf, 0.0, 1.0)
        cdf = cdf.clone()
        cdf[:, 0] = 0.0
        cdf[:, -1] = 1.0
        cdf = torch.cummax(cdf, dim=1).values      # enforce non-decreasing
        return cdf

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
        edges, cdf = self._node_tensors(X)
        u_t = torch.as_tensor(np.asarray(u, dtype=np.float64).ravel(),
                              dtype=self._DTYPE, device=self._torch_device)
        out = self._interp_rows(cdf, edges, u_t.unsqueeze(1))[:, 0]   # cdf -> edge
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
        edges, cdf = self._node_tensors(X)                       # one fwd pass
        U_arr = np.asarray(U, dtype=np.float64)
        if U_arr.ndim == 1:
            U_arr = U_arr[:, None]
        u_t = torch.as_tensor(U_arr, dtype=self._DTYPE, device=self._torch_device)
        out = self._interp_rows(cdf, edges, u_t)                 # cdf -> edge
        return out.detach().cpu().numpy()

    def sample(self, X, n_samples: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Draw ``(n_test, n_samples)`` by piecewise-linear inverse-CDF sampling.

        Uses the *same* per-row CDF as :meth:`quantile` / :meth:`cdf`, so draws,
        PITs and inverse-PITs are all mutually consistent.  Uniforms are drawn
        with the provided numpy RNG (reproducible), then inverted on device.
        """
        edges, cdf = self._node_tensors(X)
        n_test = cdf.shape[0]
        rng = np.random.default_rng() if rng is None else rng
        u_np = rng.random((n_test, n_samples))
        u_t = torch.as_tensor(u_np, dtype=self._DTYPE, device=self._torch_device)
        out = self._interp_rows(cdf, edges, u_t)                 # cdf -> edge
        return out.detach().cpu().numpy()

    def cdf(self, X, y) -> np.ndarray:
        """PIT ``F_i(y_i)`` at one value per row -> ``(n_test,)`` in ``[0,1]``.

        The *exact* inverse of :meth:`quantile`: both interpolate the same
        monotone piecewise-linear node set (one direction each), so
        ``cdf(quantile(u)) == u`` on the support by construction.
        """
        edges, cdf = self._node_tensors(X)
        y_t = torch.as_tensor(np.asarray(y, dtype=np.float64).ravel(),
                             dtype=self._DTYPE, device=self._torch_device)
        out = self._interp_rows(edges, cdf, y_t.unsqueeze(1))[:, 0]    # edge -> cdf
        return out.detach().cpu().numpy()

    def crps(self, X, y) -> np.ndarray:
        """Exact CRPS of the conditional law ``F_{Y|X_i}`` at one value per row.

        Returns ``(n_test,)`` where entry ``i`` is

            ``CRPS(F_i, y_i) = ∫ (F_i(z) − 1{z ≥ y_i})² dz``

        evaluated **in closed form** on the *same* monotone piecewise-linear CDF
        ``F_i`` that drives :meth:`cdf` / :meth:`quantile` / :meth:`sample` (see
        :func:`crps_piecewise_linear`).  This is a *true* CRPS read off the
        predictive CDF — no Gaussian assumption, no Monte-Carlo sample estimate —
        the direct multivariate-sampler analogue of the univariate benchmark's
        grid CRPS.  It is what the chained wrapper sums over its sequential
        conditionals to form the conditional CRPS (cCRPS).
        """
        nodes = self._node_tensors(X)
        y_t = torch.as_tensor(np.asarray(y, dtype=np.float64).ravel(),
                              dtype=self._DTYPE, device=self._torch_device)
        return crps_piecewise_linear(nodes.edges, nodes.cdf, y_t).detach().cpu().numpy()


def crps_piecewise_linear(edges: torch.Tensor, cdf: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Closed-form CRPS for a per-row monotone piecewise-linear CDF on a grid.

    Parameters
    ----------
    edges : (n_test, K) tensor
        Per-row grid locations (histogram bin edges) where the CDF is pinned,
        non-decreasing along the last axis (``CDFNodes.edges``).
    cdf : (n_test, K) tensor
        Matching cumulative probabilities ``F`` at each edge, with ``F[:, 0] = 0``
        and ``F[:, -1] = 1`` (as produced by :meth:`BaseSampler._node_tensors`).
    y : (n_test,) tensor
        One observed value per row.

    Returns
    -------
    (n_test,) tensor
        ``CRPS(F_i, y_i) = ∫ (F_i(z) − 1{z ≥ y_i})² dz``.

    Notes
    -----
    ``F`` is piecewise linear between the edges and *clamped* flat outside
    ``[edge₀, edge_{K−1}]`` (0 to the left, 1 to the right), exactly matching the
    piecewise-linear interpolation used for sampling.  The integral splits as
    ``∫_{-∞}^{y} F² dz + ∫_{y}^{∞} (F−1)² dz``.  On a linear bin of width ``w``
    where ``F`` runs from ``a`` to ``b``, ``∫ (F − t)² dz`` has the exact value
    ``(w/3)·((a−t)² + (a−t)(b−t) + (b−t)²)``.  Each bin is split at
    ``split_point = clamp(y, edgeⱼ, edgeⱼ₊₁)`` so the left piece uses ``t = 0``
    (below ``y``) and the right piece ``t = 1`` (above ``y``); the two flat tails
    contribute ``relu(edge₀ − y)`` and ``relu(y − edge_{K−1})``.  The result is
    the analytic CRPS of the predictive CDF, not a sample approximation.
    """
    left_edge = edges[:, :-1]                             # (n_test, K-1)
    right_edge = edges[:, 1:]
    cdf_left = cdf[:, :-1]
    cdf_right = cdf[:, 1:]
    bin_width = right_edge - left_edge
    safe_width = torch.where(bin_width > 0, bin_width, torch.ones_like(bin_width))

    # Split point per bin: clamp the observation into the bin so the left piece
    # lies at or below y (indicator term = 0) and the right piece at or above y
    # (indicator term = 1).
    split_point = torch.clamp(y[:, None], left_edge, right_edge)
    split_frac = torch.where(bin_width > 0,
                             (split_point - left_edge) / safe_width,
                             torch.zeros_like(bin_width))
    cdf_split = cdf_left + (cdf_right - cdf_left) * split_frac   # F at split point

    left_width = split_point - left_edge                  # below-y piece width
    right_width = right_edge - split_point                # above-y piece width
    # ∫ (F-0)^2 over [left_edge, split]  and  ∫ (F-1)^2 over [split, right_edge].
    left_integral = left_width / 3.0 * (
        cdf_left * cdf_left + cdf_left * cdf_split + cdf_split * cdf_split)
    cdf_split_minus1 = cdf_split - 1.0
    cdf_right_minus1 = cdf_right - 1.0
    right_integral = right_width / 3.0 * (
        cdf_split_minus1 * cdf_split_minus1
        + cdf_split_minus1 * cdf_right_minus1
        + cdf_right_minus1 * cdf_right_minus1)
    bin_integral = (left_integral + right_integral).sum(dim=1)   # (n_test,)

    # Flat tails: F=0 left of the support (mass 1 for z in (y, edge0) when
    # y<edge0), F=1 right of it (mass 1 for z in (edge_last, y) when y>edge_last).
    left_tail = torch.clamp(edges[:, 0] - y, min=0.0)
    right_tail = torch.clamp(y - edges[:, -1], min=0.0)
    return bin_integral + left_tail + right_tail
