"""Distributional Random Forest (DRF) multivariate wrapper for ScoringBench.

Conditional generative regression via **Distributional Random Forests** (Cevid,
Michel, Meinshausen & Buhlmann 2022), using the ``drforest`` package (a Rust
optimisation core, ``pip install drforest``).  Instead of splitting tree nodes
on single-variable MSE, DRF splits on a Maximum-Mean-Discrepancy (MMD) criterion
in an RKHS, so each split reduces discrepancy between the *joint* ``d``-dimensional
target distributions of the child nodes.  Like ``forest_diffusion`` this is a
*native* multivariate model — the forest is grown on the joint target block
``Y`` — so the learned cross-target dependence is intrinsic, not composed per
dimension.  There is exactly one ``drforest`` entry; no chained / copula /
independent variants.

Sampling
--------
DRF yields, for a test row ``x``, a sparse vector of nonnegative weights
``w_i(x)`` over the *training* rows (``forest.weights(X)``), summing to one.
These weights define the conditional law as the weighted empirical measure
``sum_i w_i(x) * delta_{Y_i}``.  A sample-based draw is therefore a weighted
bootstrap: draw ``n`` training indices ``i ~ Categorical(w(x))`` and return the
corresponding training target vectors ``Y_i``.  This preserves the exact joint
dependence present among the selected training targets.

Note this replaces the earlier from-scratch Fourier-feature reimplementation
(and is unrelated to the rpy2->R bridge under ``additional_models/drf``): it is
the maintained native-Python/Rust ``drforest`` package.

Compute
-------
``drforest`` runs on CPU (Rust core); there is no CUDA path for the forest
itself.  The weighted-bootstrap sampling is a NumPy gather.  (The benchmark's
*scoring* still runs on the GPU regardless of how samples were produced.)

Reference
---------
Cevid, Michel, Nasecki, Buhlmann, Meinshausen. "Distributional Random Forests:
Heterogeneity Adjustment and Multivariate Distributional Regression." JMLR 2022.
Package: ``pip install drforest``.
"""

from __future__ import annotations

import numpy as np

from ..config import N_DRAWS, SEED
from .base import as_2d_features, as_2d_targets
from .sample_based import SampleBasedWrapper


class DRForestMultiOutputWrapper(SampleBasedWrapper):
    """Native multivariate Distributional Random Forest conditional sampler.

    The forest is grown with an MMD split criterion on the joint target block
    ``Y``.  At sample time the per-test-row leaf-membership weights define a
    weighted empirical distribution over the training targets, which we draw
    from (weighted bootstrap) to produce joint draws.  The
    :class:`SampleBasedWrapper` base accumulates them into a
    :class:`~..prediction.MultivariateSamplePrediction` of shape
    ``(n_test, m, d)``.

    Parameters
    ----------
    n_estimators : int
        Number of trees in the forest.
    min_samples_leaf : int
        Minimum training rows per leaf (controls weight sparsity / smoothing).
    subsample : float
        Fraction of rows used to grow each tree.
    bootstrap : bool
        Whether trees are grown on bootstrap resamples (``False`` = subsampling
        without replacement, the honest-forest default).
    honesty_fraction : float
        Fraction of each tree's sample held out for honest leaf estimation.
    colsample : float
        Fraction of features considered per split.
    criterion : str or None
        Split criterion name; ``None`` uses the package default (Gaussian MMD).
    n_draws : int
        Target joint draws per test instance (defaults to ``config.N_DRAWS``).
    sample_chunk : int
        Draws requested per :meth:`_draw_samples` call (weighted bootstrap is
        cheap, so this defaults to one-shot ``n_draws``).
    random_state : int
        Seed for both the forest and the weighted-bootstrap sampling.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        min_samples_leaf: int = 5,
        subsample: float = 0.5,
        bootstrap: bool = False,
        honesty_fraction: float = 0.5,
        colsample: float = 0.7,
        criterion: str | None = None,
        n_draws: int = int(N_DRAWS),
        sample_chunk: int = int(N_DRAWS),
        random_state: int = int(SEED),
    ):
        self.n_estimators = int(n_estimators)
        self.min_samples_leaf = int(min_samples_leaf)
        self.subsample = float(subsample)
        self.bootstrap = bool(bootstrap)
        self.honesty_fraction = float(honesty_fraction)
        self.colsample = float(colsample)
        self.criterion = criterion
        self.random_state = int(random_state)

        self.N_SAMPLES = int(n_draws)
        self.SAMPLE_CHUNK = int(sample_chunk) or int(n_draws)

        self._model = None
        self._Y_train: np.ndarray | None = None
        self._d: int | None = None
        self._n_features: int | None = None
        # Running draw counter so consecutive chunks (and adaptive top-up
        # batches) use a *distinct* RNG stream rather than repeating a block.
        # Reset per ``predict_ensemble`` call.
        self._draw_offset: int = 0

    @staticmethod
    def _sanitize_X(X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return np.nan_to_num(X, nan=0.0, posinf=1e7, neginf=-1e7)

    def fit(self, X, Y) -> "DRForestMultiOutputWrapper":
        from drforest import DistributionalRandomForest

        X = self._sanitize_X(as_2d_features(X))
        Y = as_2d_targets(Y)

        valid = np.isfinite(Y).all(axis=1)
        X, Y = X[valid], Y[valid]
        if len(Y) == 0:
            raise ValueError("No valid (finite) training samples after sanitization")

        self._n_features = X.shape[1]
        self._d = Y.shape[1]
        self._Y_train = np.ascontiguousarray(Y, dtype=np.float64)

        self._model = DistributionalRandomForest(
            criterion=self.criterion,
            n_estimators=self.n_estimators,
            subsample=self.subsample,
            bootstrap=self.bootstrap,
            min_samples_leaf=self.min_samples_leaf,
            honesty_fraction=self.honesty_fraction,
            colsample=self.colsample,
            random_state=self.random_state,
        )
        self._model.fit(X, self._Y_train)
        return self

    def _draw_samples(self, X, n_samples: int) -> np.ndarray:
        if self._model is None or self._Y_train is None:
            raise RuntimeError("DRForestMultiOutputWrapper must be fit before sampling.")

        X = self._sanitize_X(as_2d_features(X))
        n_test = X.shape[0]
        d = int(self._d)
        n = int(n_samples)
        n_train = self._Y_train.shape[0]

        # Per-test-row sparse weights over the training rows (rows sum to 1).
        W = self._model.weights(X)  # scipy csr (n_test, n_train)

        # Seed a private Generator from the wrapper seed + running draw offset so
        # the draw is reproducible yet each chunk uses a distinct stream.
        rng = np.random.default_rng(self.random_state + 1 + self._draw_offset)

        draws = np.empty((n_test, n, d), dtype=np.float64)
        for i in range(n_test):
            start, end = W.indptr[i], W.indptr[i + 1]
            idx = W.indices[start:end]
            w = W.data[start:end]
            if idx.size == 0 or not np.isfinite(w).all() or w.sum() <= 0:
                # Degenerate leaf weights -> fall back to a uniform draw over
                # all training targets (keeps the ensemble well-defined).
                chosen = rng.integers(0, n_train, size=n)
            else:
                p = w / w.sum()
                chosen = idx[rng.choice(idx.size, size=n, p=p)]
            draws[i, :, :] = self._Y_train[chosen, :d]

        self._draw_offset += n
        return draws

    def _collect_samples(self, X) -> np.ndarray:
        # Reset per-call offset so every predict_ensemble is reproducible yet
        # its internal chunks each advance to a distinct RNG stream.
        self._draw_offset = 0
        return super()._collect_samples(X)
