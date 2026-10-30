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
import warnings
from typing import Callable

import numpy as np

from .. import config as _config
from ..config import N_DRAWS, SEED
from ..convergence import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPSILON,
    DEFAULT_MAX_INSTANCES,
    DEFAULT_MAX_SAMPLES,
    DEFAULT_N_GROUPS,
    assess_sample_prediction,
    log_sample_cap_hit,
)
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
        """Draw the initial ensemble, then adaptively top up until R̂ converges.

        After the initial ``_n_draws`` draws, if ``config.CONV_ENABLED`` is set
        the multivariate R̂ (Vats & Knudson 2020, det-form) is evaluated on a
        subsample of test instances.  Additional draws are collected in
        ``DEFAULT_BATCH_SIZE`` chunks until R̂ < ``DEFAULT_EPSILON`` or
        ``DEFAULT_MAX_SAMPLES`` is reached.  There is no wall-clock budget.  If
        the cap is hit without converging it is recorded via
        :func:`log_sample_cap_hit`.  No draws are discarded (i.i.d. MC, not MCMC).
        """
        X = as_2d_features(X)
        rng = np.random.default_rng(self._seed)
        samples = np.asarray(self._draw(X, self._n_draws, rng), dtype=np.float64)

        if not getattr(_config, "CONV_ENABLED", False):
            return MultivariateSamplePrediction(samples=samples)

        last_rhat: float | None = None
        while samples.shape[1] < DEFAULT_MAX_SAMPLES:
            pred_tmp = MultivariateSamplePrediction(samples=samples)
            try:
                diag = assess_sample_prediction(
                    pred_tmp,
                    n_groups=DEFAULT_N_GROUPS,
                    epsilon=DEFAULT_EPSILON,
                    max_instances=DEFAULT_MAX_INSTANCES,
                )
            except ValueError as exc:
                warnings.warn(
                    f"convergence check skipped (adaptive top-up): {exc}",
                    RuntimeWarning, stacklevel=2,
                )
                break
            last_rhat = diag.rhat_max
            if diag.rhat_max < DEFAULT_EPSILON:
                break
            extra = np.asarray(self._draw(X, DEFAULT_BATCH_SIZE, rng), dtype=np.float64)
            samples = np.concatenate([samples, extra], axis=1)
        else:
            # Cap reached without a convergence/exception break -> not converged.
            log_sample_cap_hit(
                model=type(self).__name__,
                n_samples=int(samples.shape[1]),
                rhat_max=last_rhat,
            )

        return MultivariateSamplePrediction(samples=samples)


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

    def conditional_crps(self, X, Y) -> float:
        """Conditional CRPS (cCRPS) of the independent model's joint.

        cCRPS factorizes the joint via the chain rule and sums the univariate
        CRPS of each sequential conditional ``p(y_{π(t)} | y_{π(<t)}, x)``.  The
        independent model *asserts* no cross-target dependence — each marginal is
        fit on the features only, so its conditional ``p(y_k | y_{<t}, x)`` equals
        the marginal ``p(y_k | x)`` regardless of the earlier coordinates.  The
        chain-rule conditionals therefore collapse to the marginals and cCRPS is
        exactly the **summed marginal CRPS** (order-independent):

            cCRPS(P̂_indep, y | x) = Σ_k CRPS( F_{Y_k|X}, y_k ).

        This needs **no distributional assumption** — the marginals are genuine
        conditional CDFs read in closed form via :meth:`BaseSampler.crps` — so,
        unlike a native joint sampler (e.g. ``forest_diffusion``, which exposes no
        per-coordinate conditional CDF), the independent baseline *can* report
        cCRPS.  Reporting it alongside the chained wrapper's cCRPS is a
        fair comparison: both are scored under the same strictly proper joint
        rule against the same realized targets, so the gap is precisely the value
        of conditioning on previous targets (``cCRPS`` ≠ summed marginal CRPS for
        a dependence-aware model; equal, by construction, for this one).

        Parameters
        ----------
        X : (n_test, p) array-like
            Test covariates (already imputed by the CV harness).
        Y : (n_test, d) array-like
            Observed test targets.

        Returns
        -------
        float
            Mean over test instances of ``Σ_k CRPS_k``.  Lower is better.
        """
        if not self._samplers:
            raise RuntimeError("conditional_crps requires a fitted wrapper.")
        X = as_2d_features(X)
        Y = as_2d_targets(Y)
        if Y.shape[1] != self._d:
            raise ValueError(
                f"Y has {Y.shape[1]} targets but the wrapper was fit on d={self._d}."
            )
        ccrps = np.zeros(X.shape[0], dtype=np.float64)
        for target_dim in range(self._d):
            ccrps += np.asarray(
                self._samplers[target_dim].crps(X, Y[:, target_dim]), dtype=np.float64)
        return float(ccrps.mean())


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
    * **Family set.**  The vine is fit with local-likelihood transformation
      kernel (TLL) pair-copulas: fully nonparametric, capturing dependence
      shapes no parametric family can, at the cost of a heavier fit.
    * **Out-of-fold PITs.**  For a TFM, ``fit(X, Y)`` loads ``(X, Y)`` into the
      in-context training set, so an *in-sample* PIT ``F_{Y_k|X_i}(y_{ik})`` is
      evaluated with row ``i`` still in the model's context.  Those PITs are
      biased (over-concentrated) and bias the copula fit on them.  We therefore
      cross-fit the pseudo-observations: the training rows are split into
      ``n_pit_folds`` internal folds, each fold's PITs are computed from a
      marginal fit on the *other* folds only, and the copula is fit on the
      pooled out-of-fold PITs.  The marginals used to invert at sampling time
      are still fit on the full training data -- cross-fitting only de-biases
      the dependence estimate, not the predictive marginals.
    * **PIT jitter.**  TabPFN's piecewise bar CDF (and degenerate constant rows)
      produce *tied* pseudo-observations, which make a vine see spurious
      independence.  A tiny reproducible uniform jitter (``pit_jitter``) breaks
      those ties before fitting without materially perturbing the ranks.
    """

    def __init__(self, sampler_factory, n_draws=int(N_DRAWS), seed=int(SEED),
                 pit_jitter: float = 1e-4, num_threads: int = 1,
                 n_pit_folds: int = 5):
        super().__init__(sampler_factory, n_draws=n_draws, seed=seed)
        self._pit_jitter = float(pit_jitter)
        self._num_threads = int(num_threads)
        self._n_pit_folds = int(n_pit_folds)
        self._copula = None

    def _oof_pits(self, X, Y):
        """Out-of-fold PIT pseudo-observations ``U`` of shape ``(n, d)``.

        Internal ``n_pit_folds``-fold cross-fitting: for each fold a fresh
        sampler is fit on the *other* folds and used to PIT the held-out rows,
        so no row is ever PITted by a model that had it in context.  Falls back
        to in-sample PITs (with a warning) when there are too few rows to split.
        """
        from sklearn.model_selection import KFold

        n = X.shape[0]
        U = np.empty((n, self._d), dtype=np.float64)

        n_folds = self._n_pit_folds
        if n_folds < 2 or n < 2 * n_folds:
            warnings.warn(
                f"copula: out-of-fold PITs disabled (n={n}, n_pit_folds="
                f"{self._n_pit_folds}); falling back to in-sample PITs, which "
                "bias the copula for in-context (TFM) marginals.",
                RuntimeWarning, stacklevel=2,
            )
            for k in range(self._d):
                U[:, k] = self._samplers[k].cdf(X, Y[:, k])
            return np.clip(U, 1e-6, 1 - 1e-6)

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=self._seed)
        for train_idx, test_idx in kf.split(X):
            X_tr, X_te = X[train_idx], X[test_idx]
            for k in range(self._d):
                s = self._make_sampler()
                s.fit(X_tr, Y[train_idx, k])           # fit WITHOUT the held-out rows
                U[test_idx, k] = s.cdf(X_te, Y[test_idx, k])
        return np.clip(U, 1e-6, 1 - 1e-6)

    def _fit_dims(self, X, Y):
        import pyvinecopulib as pv

        # Marginals used for inversion at sampling time: fit on the FULL training
        # data, conditioning on features only (as in the independent wrapper).
        for k in range(self._d):
            s = self._make_sampler()
            s.fit(X, Y[:, k])
            self._samplers.append(s)

        if self._d == 1:
            self._copula = None
            return

        # Dependence estimate uses OUT-OF-FOLD PITs so no row is PITted while in
        # its own marginal's (in-context) training set -- see _oof_pits.
        U = self._oof_pits(X, Y)

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

        # Local-likelihood transformation kernel (TLL) pair-copulas: fully
        # nonparametric, captures dependence shapes no parametric family can,
        # at the cost of a heavier fit.
        controls = pv.FitControlsVinecop(
            family_set=[pv.BicopFamily.tll], num_threads=self._num_threads)
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
            for position, target_dim in enumerate(order):
                preceding_dims = order[:position]       # dims placed before this one
                # Augment covariates with the TRUE earlier targets in this
                # order (teacher forcing), matching the sampling-time layout.
                X_aug = (X if position == 0
                         else np.concatenate([X, Y[:, preceding_dims]], axis=1))
                sampler = self._make_sampler()
                sampler.fit(X_aug, Y[:, target_dim])
                chain.append(sampler)
            self._chains.append(chain)
        # Keep a flat sampler list too so shared introspection still works.
        self._samplers = [sampler for chain in self._chains for sampler in chain]

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

    def conditional_crps(self, X, Y) -> float:
        """Conditional CRPS (cCRPS) of the chained joint predictive law.

        The chain-rule factorization ``p(y | x) = Π_t p(y_{π(t)} | y_{π(<t)}, x)``
        turns the joint into a sequence of *univariate* conditionals.  The cCRPS
        (Roordink & Hess 2023) is the sum of the univariate CRPS of each of those
        sequential conditionals, evaluated against the observed coordinate while
        conditioning on the **observed** earlier coordinates (teacher forcing on
        the true ``y_{π(<t)}``, not on drawn values)::

            cCRPS(P̂, y | x) = Σ_t CRPS( P̂(y_{π(t)} | y_{π(<t)}, x), y_{π(t)} ).

        This is a strictly proper scoring rule for the *joint* law and, unlike the
        sum of marginal CRPS, genuinely rewards the learned cross-target
        dependence (each conditional sees the true preceding targets).  Each
        per-coordinate CRPS is read off the base sampler's piecewise-linear
        predictive CDF in closed form via :meth:`BaseSampler.crps` — a true CRPS
        with no Gaussian assumption and no Monte-Carlo sampling, the direct
        analogue of the univariate benchmark's CRPS.

        Because the conditioning is on *observed* targets (available only at
        scoring time, with the realised ``y``), cCRPS is a property of the fitted
        conditional chain, not of the emitted sample ensemble — so it lives here
        on the chained wrapper rather than in the sample-based ``metrics`` module,
        and is reported **only** for chained models.  When the wrapper averages
        over ``n_orders`` permutations, the cCRPS is averaged over the orders
        (each order is an exact, equivalent factorization of the same joint).

        Parameters
        ----------
        X : (n_test, p) array-like
            Test covariates (already imputed by the CV harness).
        Y : (n_test, d) array-like
            Observed test targets.

        Returns
        -------
        float
            Mean over test instances of ``Σ_t CRPS_t``, averaged over chain
            orders.  Lower is better.
        """
        if not self._chains:
            raise RuntimeError("conditional_crps requires a fitted chained wrapper.")
        X = as_2d_features(X)
        Y = as_2d_targets(Y)
        if Y.shape[1] != self._d:
            raise ValueError(
                f"Y has {Y.shape[1]} targets but the wrapper was fit on d={self._d}."
            )
        n_test = X.shape[0]

        per_order = np.zeros(len(self._chains), dtype=np.float64)
        for order_index, (order, chain) in enumerate(zip(self._orders, self._chains)):
            ccrps = np.zeros(n_test, dtype=np.float64)
            for position, target_dim in enumerate(order):
                preceding_dims = order[:position]
                # Condition each sequential conditional on the TRUE earlier
                # targets (teacher forcing), matching the fit-time augmentation.
                X_aug = (X if position == 0
                         else np.concatenate([X, Y[:, preceding_dims]], axis=1))
                ccrps += np.asarray(
                    chain[position].crps(X_aug, Y[:, target_dim]), dtype=np.float64)
            per_order[order_index] = float(ccrps.mean())
        return float(per_order.mean())

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
