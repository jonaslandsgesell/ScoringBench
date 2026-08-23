"""Multivariate wrappers built by composing univariate conditional samplers.

Three composition *modes*, all parameterised by a per-dimension base sampler
(:class:`~.tabpfn_sampler.TabPFNSampler` or
:class:`~.tabicl_sampler.TabICLSampler`), and all sharing one base class so the
plumbing (fit bookkeeping, shape handling, RNG, ensemble assembly) lives in
exactly one place:

* **independent** (:class:`IndependentMultiOutputWrapper`, baseline **A**) — fit
  one base model per target dimension on the *original* covariates and sample
  each dimension independently, then stack into a joint ``(n_test, m, d)``
  ensemble.  The joint is a product of marginals: no learned cross-dimensional
  dependence.

* **copula** (:class:`CopulaMultiOutputWrapper`) — fit the *same* conditional
  marginals ``F_{Y_k|X}`` as the independent model, but glue them with a vine
  copula fit on the PIT pseudo-observations ``u_{ik}=F_{Y_k|X_i}(y_{ik})``.  The
  PITs retain the residual cross-target dependence *not* explained by ``X``;
  sampling draws correlated uniforms from the copula and inverts each
  conditional marginal.  This is the sampling analogue of
  ``ConditionalCopulaTabPFN``.

* **chained** (:class:`ChainedMultiOutputWrapper`, baseline **B**) — the product
  rule ``p(y_1..y_d | x) = Π_k p(y_k | x, y_{<k})``.  Fit dimension ``k`` on
  ``[X, Y_{<k}]`` (teacher forcing); draw the chain jointly, feeding each drawn
  coordinate back into the covariates.  This is the sampling analogue of
  ``ChainedTabPFN``.

All three emit a :class:`MultivariateSamplePrediction` of shape
``(n_test, m, d)``.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from ..config import N_DRAWS, SEED
from ..prediction import MultivariateSamplePrediction
from .base import MultivariateWrapper, as_2d_features, as_2d_targets

# A sampler factory: called with no args, returns a fresh per-dim sampler that
# implements .fit(X, y), .sample(X, n, rng) -> (n_test, n), and (for the copula
# mode) .cdf(X, y) / .quantile(X, u).
SamplerFactory = Callable[[], object]


# ---------------------------------------------------------------------------
# Shared base: per-dimension composition of univariate samplers
# ---------------------------------------------------------------------------

class _ComposedMultiOutputWrapper(MultivariateWrapper):
    """Common machinery for the independent / copula / chained wrappers.

    Subclasses implement :meth:`_fit_dims` (train the per-dimension samplers)
    and :meth:`_draw` (produce the ``(n_test, m, d)`` ensemble).  Everything
    else — coercion, seeding, dimension bookkeeping, and packaging into a
    :class:`MultivariateSamplePrediction` — is shared here.
    """

    #: Max number of *replicated* rows (``n_test * draws``) pushed through a
    #: single base-model forward pass.  Draw generation is chunked over the test
    #: rows so the model never sees more than this many rows at once, bounding
    #: peak memory (a 600-row test set × 300 draws would otherwise materialize a
    #: 180k-row forward pass + intermediate arrays in one shot).
    ROW_BATCH: int = 20_000

    def __init__(self, sampler_factory: SamplerFactory, n_draws: int = int(N_DRAWS),
                 seed: int = int(SEED)):
        self._make_sampler = sampler_factory
        self._n_draws = int(n_draws)
        self._seed = int(seed)
        self._samplers: list = []
        self._d: int | None = None

    # -- shared row-chunking helper -------------------------------------
    def _row_chunk_size(self, m: int) -> int:
        """Number of *test rows* per draw chunk so ``rows * m <= ROW_BATCH``."""
        m = max(int(m), 1)
        return max(1, int(self.ROW_BATCH) // m)

    # -- subclass hooks --------------------------------------------------
    def _fit_dims(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Train ``self._samplers`` (and any auxiliary state) from ``(X, Y)``."""
        raise NotImplementedError

    def _draw(self, X: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
        """Return an ``(n_test, m, d)`` ensemble of draws for the rows of ``X``."""
        raise NotImplementedError

    # -- shared fit / predict -------------------------------------------
    def fit(self, X, Y) -> "_ComposedMultiOutputWrapper":
        X = as_2d_features(X)
        Y = as_2d_targets(Y)
        self._d = Y.shape[1]
        self._samplers = []
        self._fit_dims(X, Y)
        return self

    def predict_ensemble(self, X) -> MultivariateSamplePrediction:
        X = as_2d_features(X)
        rng = np.random.default_rng(self._seed)
        samples = self._draw(X, self._n_draws, rng)
        return MultivariateSamplePrediction(samples=np.asarray(samples, dtype=np.float64))


# ---------------------------------------------------------------------------
# Mode A: independent per-dimension
# ---------------------------------------------------------------------------

class IndependentMultiOutputWrapper(_ComposedMultiOutputWrapper):
    """Predict each target dimension independently (baseline A)."""

    def _fit_dims(self, X, Y):
        for k in range(self._d):
            s = self._make_sampler()
            s.fit(X, Y[:, k])
            self._samplers.append(s)

    def _draw(self, X, m, rng):
        per_dim = []
        for k in range(self._d):
            # (n_test, m); independent stream per dimension.
            draws_k = self._samplers[k].sample(X, m, rng=rng)
            per_dim.append(np.asarray(draws_k, dtype=np.float64))
        # Stack along a new last axis -> (n_test, m, d).
        return np.stack(per_dim, axis=-1)


# ---------------------------------------------------------------------------
# Mode B: copula over conditional marginals
# ---------------------------------------------------------------------------

class CopulaMultiOutputWrapper(_ComposedMultiOutputWrapper):
    """Glue the independent conditional marginals with a vine copula.

    Marginals ``F_{Y_k|X}`` are fit exactly as in the independent wrapper (each
    conditioned on the *features only*).  The training PITs
    ``u_{ik}=F_{Y_k|X_i}(y_{ik})`` capture the residual cross-target dependence
    a per-dimension model cannot; a vine copula (``pyvinecopulib``) is fit on
    them.  To sample we draw correlated uniforms ``u* ~ copula`` (shared across
    dimensions for a given draw) and invert each conditional marginal via the
    sampler's monotone-PCHIP :meth:`quantile`.

    Robustness details
    ------------------
    * **Family set.**  ``family_set="parametric"`` uses a rich set that adds the
      two-parameter BB1/BB6/BB7/BB8 families and Tawn (asymmetric / tail
      dependence) on top of the one-parameter families, with rotations enabled.
      BIC selection falls back to simpler families when the extra flexibility is
      not warranted, so this is a strict improvement over a Gaussian/Archimedean
      set.  ``"nonparametric"`` fits TLL kernel pair-copulas; any other value
      uses ``pyvinecopulib`` defaults.
    * **PIT jitter.**  TabPFN's piecewise bar CDF (and degenerate constant rows)
      produce *tied* pseudo-observations, which make a vine see spurious
      independence.  A tiny reproducible uniform jitter (``pit_jitter``) breaks
      those ties before fitting without materially perturbing the ranks.
    """

    def __init__(self, sampler_factory, n_draws=int(N_DRAWS), seed=int(SEED),
                 family_set: str = "parametric", pit_jitter: float = 1e-4,
                 num_threads: int = 1):
        super().__init__(sampler_factory, n_draws=n_draws, seed=seed)
        self._family_set = family_set
        self._pit_jitter = float(pit_jitter)
        self._num_threads = int(num_threads)
        self._copula = None

    def _fit_dims(self, X, Y):
        import pyvinecopulib as pv

        n = X.shape[0]
        U = np.empty((n, self._d), dtype=np.float64)
        for k in range(self._d):
            s = self._make_sampler()
            s.fit(X, Y[:, k])                          # condition on FEATURES only
            # PIT with each row's own conditional CDF F_{Y_k|X_i}.
            u_col = s.cdf(X, Y[:, k])
            U[:, k] = np.clip(u_col, 1e-6, 1 - 1e-6)
            self._samplers.append(s)

        if self._d == 1:
            self._copula = None
            return

        # -- break PIT ties so the vine sees strictly-continuous pseudo-obs.
        # TabPFN's bar CDF is piecewise, and degenerate/constant conditional
        # rows collapse whole groups of PITs onto identical values; a vine fit
        # on tied ranks yields degenerate rank correlations and spurious
        # independence.  A tiny reproducible uniform jitter (rank-preserving in
        # expectation) restores continuity without materially moving the ranks.
        if self._pit_jitter > 0.0:
            jit_rng = np.random.default_rng(self._seed + 101)
            U = U + jit_rng.uniform(-self._pit_jitter, self._pit_jitter, size=U.shape)
            U = np.clip(U, 1e-6, 1 - 1e-6)

        if self._family_set == "parametric":
            # Include the two-parameter BB families (asymmetric / tail
            # dependence) and Tawn on top of the one-parameter set; rotations
            # are enabled by default so lower- *and* upper-tail dependence can
            # be captured.  This is a strict superset of the previous set, so
            # BIC selection falls back to the simpler families when the extra
            # flexibility is not warranted.
            fam = [pv.BicopFamily.indep, pv.BicopFamily.gaussian,
                   pv.BicopFamily.student, pv.BicopFamily.clayton,
                   pv.BicopFamily.gumbel, pv.BicopFamily.frank,
                   pv.BicopFamily.joe, pv.BicopFamily.bb1,
                   pv.BicopFamily.bb6, pv.BicopFamily.bb7,
                   pv.BicopFamily.bb8, pv.BicopFamily.tawn]
            controls = pv.FitControlsVinecop(
                family_set=fam, selection_criterion="bic",
                allow_rotations=True, num_threads=self._num_threads)
        elif self._family_set == "nonparametric":
            # Local-likelihood transformation kernel (TLL) pair-copulas: fully
            # nonparametric, captures dependence shapes no parametric family
            # can, at the cost of a heavier fit.
            controls = pv.FitControlsVinecop(
                family_set=[pv.BicopFamily.tll], num_threads=self._num_threads)
        else:
            controls = pv.FitControlsVinecop(num_threads=self._num_threads)
        self._copula = pv.Vinecop.from_data(U, controls=controls)

    def _draw(self, X, m, rng):
        n_test = X.shape[0]
        # One correlated uniform vector per (test row, draw).  Draw n_test*m
        # copula uniforms, then invert each dimension's conditional marginal at
        # the replicated covariates so every draw shares the same dependence.
        n = n_test * m
        if self._copula is None:
            U = rng.random((n, self._d))
        else:
            seed = int(rng.integers(0, 2**31 - 1))
            U = np.asarray(self._copula.simulate(n, seeds=[seed]), dtype=np.float64)
            U = np.clip(U, 1e-6, 1 - 1e-6)

        # Invert the conditional marginals.  The predictive law F_{Y_k|X_i}
        # depends only on the *covariates*, so each of the n_test distinct rows
        # needs exactly ONE model forward pass — not m replicated copies.  The
        # copula uniforms are laid out row-major (row r, draw j at r*m + j), so
        # reshaping U[:, k] to (n_test, m) groups every draw for a given test
        # row into one CDF-inversion batch.  quantile_batch builds each row's
        # conditional CDF once and inverts all m levels against it, collapsing
        # the old (n_test * m)-row forward pass down to n_test rows.
        out = np.empty((n_test, m, self._d), dtype=np.float64)
        for k in range(self._d):
            U_k = U[:, k].reshape(n_test, m)            # (n_test, m) levels
            out[:, :, k] = self._samplers[k].quantile_batch(X, U_k)
        return out


# ---------------------------------------------------------------------------
# Mode C: Bayes-chained (regressor chain / product rule)
# ---------------------------------------------------------------------------

class ChainedMultiOutputWrapper(_ComposedMultiOutputWrapper):
    """Chain dimensions via the product rule, sampling jointly (baseline B).

    Training: for a chain order ``π`` dimension ``π(t)`` is fit on
    ``[X, Y_{π(<t)}]`` using the *true* previous targets (teacher forcing).
    Prediction: draws are generated one coordinate at a time along ``π``,
    feeding each drawn coordinate back into the covariates for the next — so the
    joint ensemble reflects the learned dependence.

    Order robustness
    ----------------
    The chain-rule factorization ``p(y|x)=Π_t p(y_{π(t)}|x, y_{π(<t)})`` is exact
    for *any* permutation ``π``, but with imperfect conditional models the
    *sampled* joint is order-dependent: errors compound down the chain (a form
    of exposure bias, since we teacher-force on true targets at fit time but
    feed *drawn* targets at sample time).  A single arbitrary order can there­
    fore bias the dependence structure.  To desensitise the estimate we fit
    ``n_orders`` independent chains under different (seeded) random permutations
    and split the ``m`` draws across them, so the returned ensemble mixes the
    per-order joints.  ``n_orders=1`` recovers the classic single-chain
    behaviour; ``order`` pins an explicit permutation (and forces
    ``n_orders=1``).

    Parameters
    ----------
    n_orders:
        Number of random chain permutations to average over (draws are split
        as evenly as possible across them).
    order:
        Optional explicit permutation of ``range(d)``.  When given, a single
        chain is fit in exactly that order.
    """

    def __init__(self, sampler_factory, n_draws=int(N_DRAWS), seed=int(SEED),
                 n_orders: int = 3, order: list[int] | None = None):
        super().__init__(sampler_factory, n_draws=n_draws, seed=seed)
        self._order = None if order is None else [int(i) for i in order]
        self._n_orders = 1 if self._order is not None else max(1, int(n_orders))
        self._orders: list[np.ndarray] = []
        # One list of d samplers per chain order; self._samplers[o][t] is the
        # sampler for the t-th position of order o.
        self._chains: list[list] = []

    def _build_orders(self) -> list[np.ndarray]:
        """Seeded chain permutations (identity first for reproducible baseline)."""
        d = self._d
        if self._order is not None:
            return [np.asarray(self._order, dtype=int)]
        orders = [np.arange(d)]                         # identity: stable anchor
        perm_rng = np.random.default_rng(self._seed + 202)
        seen = {tuple(orders[0].tolist())}
        # Draw distinct permutations where feasible; fall back to duplicates
        # only when d is too small to supply n_orders unique orders.
        attempts = 0
        while len(orders) < self._n_orders and attempts < 50 * self._n_orders:
            cand = perm_rng.permutation(d)
            attempts += 1
            key = tuple(cand.tolist())
            if key in seen and len(seen) < math.factorial(d):
                continue
            seen.add(key)
            orders.append(cand)
        while len(orders) < self._n_orders:             # d tiny: allow repeats
            orders.append(perm_rng.permutation(d))
        return orders

    def _fit_dims(self, X, Y):
        self._orders = self._build_orders()
        self._chains = []
        for order in self._orders:
            chain = []
            for t, k in enumerate(order):
                prev = order[:t]                        # dims placed before k
                # Augment covariates with the TRUE earlier targets in this
                # order (teacher forcing), matching the sampling-time layout.
                X_aug = X if t == 0 else np.concatenate([X, Y[:, prev]], axis=1)
                s = self._make_sampler()
                s.fit(X_aug, Y[:, k])
                chain.append(s)
            self._chains.append(chain)
        # Keep a flat sampler list too so shared introspection still works.
        self._samplers = [s for chain in self._chains for s in chain]

    def _draw_one_order(self, X, order, chain, m, rng):
        """``(n_test, m, d)`` draws for a single chain order.

        The **first** chain dimension conditions on the covariates *only*, so
        all ``m`` draws of a given test row share one predictive law: we draw it
        with a single ``sample(X, m)`` forward pass over the ``n_test`` distinct
        rows (the fast independent-style path) instead of replicating rows.

        Every **subsequent** dimension is teacher-forced on the *drawn* earlier
        coordinates, so its augmented covariates ``[X, y_{<t}]`` genuinely
        differ across draws — the ``n_test * m`` forward pass is intrinsic to
        autoregressive sampling and cannot be collapsed.  Those steps are
        row-chunked so no single forward pass exceeds ``ROW_BATCH`` replicated
        rows; each chunk costs ``d - 1`` forward passes.
        """
        n_test = X.shape[0]
        drawn = np.empty((n_test, m, self._d), dtype=np.float64)

        # -- dim 0: covariates only -> one fwd pass over n_test rows, m draws.
        k0 = int(order[0])
        drawn[:, :, k0] = np.asarray(chain[0].sample(X, m, rng=rng),
                                     dtype=np.float64)
        if self._d == 1:
            return drawn

        # -- dims 1..d-1: autoregressive, teacher-forced on the drawn earlier
        # coordinates.  Replicate rows m times (row r, draw j at r*m + j) and
        # walk the remaining chain, feeding each new coordinate back in.
        rows_per_chunk = self._row_chunk_size(m)
        for r0 in range(0, n_test, rows_per_chunk):
            r1 = min(r0 + rows_per_chunk, n_test)
            X_rep = np.repeat(X[r0:r1], m, axis=0)      # ((r1-r0)*m, p)
            # Seed the augmented covariates with dim-0's already-drawn values
            # in the layout the chain expects (order[:1] placed first).
            y0 = drawn[r0:r1, :, k0].reshape(-1)        # ((r1-r0)*m,)
            X_aug = np.concatenate([X_rep, y0[:, None]], axis=1)
            for t in range(1, len(order)):
                k = int(order[t])
                draw_k = chain[t].sample(X_aug, 1, rng=rng)[:, 0]
                drawn[r0:r1, :, k] = draw_k.reshape(r1 - r0, m)
                X_aug = np.concatenate([X_aug, draw_k[:, None]], axis=1)
        return drawn

    def _draw(self, X, m, rng):
        if len(self._chains) == 1:
            return self._draw_one_order(X, self._orders[0], self._chains[0], m, rng)
        # Split the m draws as evenly as possible across the chain orders and
        # concatenate the per-order sub-ensembles along the draw axis.  Mixing
        # the per-order joints averages out any single order's exposure bias.
        n_orders = len(self._chains)
        base, extra = divmod(m, n_orders)
        counts = [base + (1 if o < extra else 0) for o in range(n_orders)]
        parts = [
            self._draw_one_order(X, order, chain, cnt, rng)
            for order, chain, cnt in zip(self._orders, self._chains, counts)
            if cnt > 0
        ]
        return np.concatenate(parts, axis=1)            # (n_test, m, d)
