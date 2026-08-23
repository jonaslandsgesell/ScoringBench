"""Surjectors (JAX) conditional normalizing-flow wrapper for ScoringBench.

Surjectors (https://surjectors.readthedocs.io) builds conditional normalizing
flows on top of JAX/haiku/distrax. For a 1-D target ``y`` conditioned on ``x``
we use a masked-autoregressive flow (MAF) whose conditioner is a MADE network;
stacking affine MAF layers with order-reversing permutations yields a flexible
conditional density. ``predict_distribution`` evaluates the trained log-density
on a shared ``y``-grid and discretizes it via
:func:`grid_density_to_distribution`.

The model trains on CPU (the JAX build here is CPU-only); only ScoringBench's
metric computation needs a GPU.
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np

from .base import DistributionPrediction, ProbabilisticWrapper
from .sample_based import grid_density_to_distribution

# Canonical default hyperparameters per flow type. Single source of truth shared
# by the benchmark's MODELS registration and the integration tests.
SURJECTORS_PRESETS = {
    "maf": dict(n_layers=3, hidden=(64, 64), n_iter=400, batch_size=100, lr=1e-3),
}


class SurjectorsWrapper(ProbabilisticWrapper):
    """Conditional masked-autoregressive flow (Surjectors/JAX) with grid PMF.

    Parameters
    ----------
    flow : str
        Flow family. Currently only ``"maf"`` (masked autoregressive flow).
    n_layers : int
        Number of affine MAF layers (each followed by an order permutation).
    hidden : tuple[int, ...]
        Hidden layer sizes of the MADE conditioner.
    n_iter : int
        Number of training epochs (full passes over the data).
    batch_size : int
        Minibatch size for the optax training loop.
    lr : float
        Adam learning rate.
    n_grid : int
        Number of points on the shared ``y``-grid used to discretize the
        predicted conditional density into the ScoringBench PMF.
    grid_pad : float
        Fraction of the (standardized) ``y`` range to extend the density grid
        beyond ``[y_min, y_max]`` on each side.
    eval_chunk : int
        Number of test rows whose (x, y-grid) log-densities are evaluated per
        JAX call, to bound peak memory.
    random_seed : int
        Seed for haiku/JAX PRNG.
    """

    def __init__(
        self,
        flow: str = "maf",
        n_layers: int | None = None,
        hidden=None,
        n_iter: int | None = None,
        batch_size: int | None = None,
        lr: float | None = None,
        n_grid: int = 200,
        grid_pad: float = 0.15,
        eval_chunk: int = 64,
        random_seed: int = 0,
    ):
        if flow not in SURJECTORS_PRESETS:
            raise ValueError(
                f"Unsupported Surjectors flow {flow!r}; "
                f"choose from {sorted(SURJECTORS_PRESETS)}"
            )
        preset = SURJECTORS_PRESETS[flow]
        self.flow = flow
        self.n_layers = int(n_layers if n_layers is not None else preset["n_layers"])
        self.hidden = tuple(hidden if hidden is not None else preset["hidden"])
        self.n_iter = int(n_iter if n_iter is not None else preset["n_iter"])
        self.batch_size = int(batch_size if batch_size is not None else preset["batch_size"])
        self.lr = float(lr if lr is not None else preset["lr"])
        self.n_grid = int(n_grid)
        self.grid_pad = float(grid_pad)
        self.eval_chunk = int(eval_chunk)
        self.random_seed = int(random_seed)

        self._fn = None
        self._params = None
        self._x_mean = self._x_std = None
        self._y_mean = self._y_std = None
        self._grid_s = None  # standardized y-grid
        self._grid_o = None  # original-scale y-grid

    @staticmethod
    def _sanitize_X(X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return np.nan_to_num(X, nan=0.0, posinf=1e7, neginf=-1e7)

    def _build_flow(self, n_dim: int):
        import haiku as hk
        import distrax
        from jax import numpy as jnp
        from surjectors import (
            MaskedAutoregressive,
            Permutation,
            Chain,
            TransformedDistribution,
        )
        from surjectors.nn import MADE
        from surjectors.util import unstack

        hidden = list(self.hidden)
        n_layers = self.n_layers

        def flow(**kwargs):
            def bijector_fn(params):
                means, log_scales = unstack(params, -1)
                return distrax.Inverse(distrax.ScalarAffine(means, jnp.exp(log_scales)))

            layers = []
            order = jnp.arange(n_dim)
            for _ in range(n_layers):
                layers.append(
                    MaskedAutoregressive(
                        conditioner=MADE(n_dim, hidden, 2),
                        bijector_fn=bijector_fn,
                    )
                )
                order = order[::-1]
                layers.append(Permutation(order, 1))
            transform = Chain(layers)
            base = distrax.Independent(
                distrax.Normal(jnp.zeros(n_dim), jnp.ones(n_dim)), 1
            )
            return TransformedDistribution(base, transform)(**kwargs)

        return hk.transform(flow)

    def fit(self, X, y) -> "SurjectorsWrapper":
        try:
            import jax
            from jax import numpy as jnp
            import haiku as hk
            import optax
            from surjectors.util import as_batch_iterator
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "Failed to import surjectors/jax. Install surjectors to use this wrapper."
            ) from exc

        X = self._sanitize_X(X)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        valid = np.isfinite(y)
        X, y = X[valid], y[valid]
        if len(y) == 0:
            raise ValueError("No valid (finite) training samples after sanitization")

        # Standardize inputs/targets — essential for stable flow training.
        self._x_mean = X.mean(0, keepdims=True)
        self._x_std = X.std(0, keepdims=True)
        self._x_std[self._x_std == 0] = 1.0
        self._y_mean = float(y.mean())
        self._y_std = float(y.std()) or 1.0

        Xs = ((X - self._x_mean) / self._x_std).astype("float32")
        ys = ((y - self._y_mean) / self._y_std).astype("float32").reshape(-1, 1)

        lo, hi = float(ys.min()), float(ys.max())
        pad = self.grid_pad * (hi - lo if hi > lo else 1.0)
        self._grid_s = np.linspace(lo - pad, hi + pad, self.n_grid).astype("float32")
        self._grid_o = self._grid_s * self._y_std + self._y_mean

        self._fn = self._build_flow(n_dim=1)
        named = namedtuple("named_dataset", "y x")(ys, Xs)
        key = hk.PRNGSequence(self.random_seed)
        batch_size = min(self.batch_size, len(ys))
        train_iter = as_batch_iterator(next(key), named, batch_size, True)

        params = self._fn.init(next(key), method="log_prob", **train_iter(0))
        optimizer = optax.adam(self.lr)
        state = optimizer.init(params)
        fn = self._fn

        @jax.jit
        def step(params, state, **batch):
            def loss_fn(p):
                return -jnp.mean(fn.apply(p, None, method="log_prob", **batch))

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, new_state = optimizer.update(grads, state, params)
            return loss, optax.apply_updates(params, updates), new_state

        for _ in range(self.n_iter):
            for j in range(train_iter.num_batches):
                _, params, state = step(params, state, **train_iter(j))

        self._params = params
        return self

    def _density_on_grid(self, X) -> np.ndarray:
        import jax
        from jax import numpy as jnp

        X = self._sanitize_X(X)
        Xs = ((X - self._x_mean) / self._x_std).astype("float32")
        n, G = Xs.shape[0], self.n_grid
        grid = self._grid_s.reshape(1, -1, 1)
        fn, params = self._fn, self._params

        dens = np.empty((n, G), dtype=np.float64)
        for start in range(0, n, self.eval_chunk):
            stop = min(start + self.eval_chunk, n)
            xb = Xs[start:stop]
            m = xb.shape[0]
            Yg = np.tile(grid, (m, 1, 1)).reshape(-1, 1).astype("float32")
            Xg = np.repeat(xb, G, axis=0).astype("float32")
            lp = fn.apply(params, None, method="log_prob", y=jnp.asarray(Yg), x=jnp.asarray(Xg))
            dens[start:stop] = np.asarray(jax.device_get(lp)).reshape(m, G)
        return np.exp(dens)

    def predict_distribution(self, X) -> DistributionPrediction:
        if self._params is None:
            raise RuntimeError("Model must be fitted before calling predict_distribution")
        dens = self._density_on_grid(X)
        return grid_density_to_distribution(self._grid_o, dens)

    def predict(self, X) -> np.ndarray:
        return self.predict_distribution(X).mean
