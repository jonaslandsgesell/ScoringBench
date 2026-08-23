"""PyMC-BART wrapper for ScoringBench.

Bayesian Additive Regression Trees via ``pymc_bart``. A sample-based model:
``fit`` runs MCMC to obtain the posterior over the sum-of-trees mean function,
and ``predict_distribution`` draws posterior-predictive samples for each test
row, which the :class:`SampleBasedWrapper` base converts into the standard PMF.

The posterior-predictive draw is a single batched call, so its cost is governed
by the trace size (``chains * draws``) chosen at fit time; the base class's
``MAX_SAMPLE_SECONDS`` budget still bounds repeated draws.
"""

from __future__ import annotations

import numpy as np

from .sample_based import SampleBasedWrapper


class BARTWrapper(SampleBasedWrapper):
    """Bayesian Additive Regression Trees (pymc_bart) with a Normal likelihood.

    Parameters
    ----------
    num_trees : int
        Number of trees in the BART sum-of-trees prior.
    draws : int
        Posterior draws per chain.
    tune : int
        Tuning (warm-up) iterations per chain.
    chains : int
        Number of MCMC chains. Total posterior-predictive samples per test row
        is ``chains * draws``.
    cores : int
        Number of cores for sampling.
    random_state : int
        Seed for MCMC and posterior-predictive sampling.
    """

    SAMPLE_CHUNK = 0  # one-shot posterior-predictive draw

    def __init__(
        self,
        num_trees: int = 50,
        draws: int = 150,
        tune: int = 200,
        chains: int = 2,
        cores: int = 1,
        random_state: int = 0,
    ):
        self.num_trees = num_trees
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.cores = cores
        self.random_state = random_state
        self.N_SAMPLES = int(chains * draws)

        self._model = None
        self._idata = None
        self._n_features = None

    @staticmethod
    def _sanitize_X(X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return np.nan_to_num(X, nan=0.0, posinf=1e7, neginf=-1e7)

    def fit(self, X, y) -> "BARTWrapper":
        import pymc as pm
        import pymc_bart as pmb

        X = self._sanitize_X(X)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        valid = np.isfinite(y)
        X, y = X[valid], y[valid]
        if len(y) == 0:
            raise ValueError("No valid (finite) training samples after sanitization")

        self._n_features = X.shape[1]

        with pm.Model() as model:
            X_data = pm.Data("X", X)
            mu = pmb.BART("mu", X_data, y, m=self.num_trees)
            sigma = pm.HalfNormal("sigma", sigma=float(np.std(y) + 1e-6))
            pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y, shape=mu.shape)

            self._idata = pm.sample(
                draws=self.draws,
                tune=self.tune,
                chains=self.chains,
                cores=self.cores,
                random_seed=self.random_state,
                progressbar=False,
            )
        self._model = model
        return self

    def _draw_samples(self, X, n_samples: int) -> np.ndarray:
        import pymc as pm

        X = self._sanitize_X(X)
        n_test = X.shape[0]
        with self._model:
            pm.set_data({"X": X})
            pp = pm.sample_posterior_predictive(
                self._idata,
                predictions=True,
                progressbar=False,
                random_seed=self.random_state,
            )
        # (chains, draws, n_test) -> (n_test, chains*draws)
        arr = np.asarray(pp.predictions["y_obs"].values, dtype=np.float64)
        samples = arr.reshape(-1, n_test).T
        if n_samples is not None and samples.shape[1] > n_samples:
            samples = samples[:, :n_samples]
        return samples
