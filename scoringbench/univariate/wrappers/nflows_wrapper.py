"""Conditional normalizing-flow wrapper (nflows) for ScoringBench.

A genuinely sample-based model: ``fit`` trains a conditional normalizing flow
``p(y | x)`` by maximum likelihood, and ``predict_distribution`` draws
conditional samples that the :class:`SampleBasedWrapper` base turns into the
standard PMF.

The flow models the 1-D standardized target conditioned on the (standardized)
feature vector through a stack of masked piecewise rational-quadratic spline
autoregressive transforms with a standard-normal base density.
"""

from __future__ import annotations

import numpy as np

from .base import DistributionPrediction  # noqa: F401  (re-exported expectation)
from .sample_based import SampleBasedWrapper


class NFlowsWrapper(SampleBasedWrapper):
    """Conditional normalizing flow trained by maximum likelihood.

    Parameters
    ----------
    n_layers : int
        Number of spline autoregressive transform blocks.
    hidden_features : int
        Width of the autoregressive conditioner networks.
    num_bins : int
        Number of spline bins per transform.
    n_epochs : int
        Maximum training epochs.
    batch_size : int
        Mini-batch size for training.
    lr : float
        Adam learning rate.
    tail_bound : float
        Spline support half-width (in standardized target units).
    device : str, optional
        Torch device; defaults to CUDA when available.
    n_samples : int
        Conditional draws per test instance used to build the PMF.
    """

    SAMPLE_CHUNK = 0  # one-shot sampler: draw all N_SAMPLES in a single call

    def __init__(
        self,
        n_layers: int = 4,
        hidden_features: int = 64,
        num_bins: int = 8,
        n_epochs: int = 300,
        batch_size: int = 256,
        lr: float = 1e-3,
        tail_bound: float = 6.0,
        device: str | None = None,
        n_samples: int = 300,
    ):
        self.n_layers = n_layers
        self.hidden_features = hidden_features
        self.num_bins = num_bins
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.tail_bound = tail_bound
        self.device = device
        self.N_SAMPLES = int(n_samples)

        self._flow = None
        self._x_mean = self._x_std = None
        self._y_mean = self._y_std = None

    def _build_flow(self, context_features: int):
        import torch  # noqa: F401
        from nflows.distributions.normal import StandardNormal
        from nflows.flows.base import Flow
        from nflows.transforms.autoregressive import (
            MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
        )
        from nflows.transforms.base import CompositeTransform

        transforms = []
        for _ in range(self.n_layers):
            transforms.append(
                MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    features=1,
                    hidden_features=self.hidden_features,
                    context_features=context_features,
                    num_bins=self.num_bins,
                    tails="linear",
                    tail_bound=self.tail_bound,
                )
            )
        transform = CompositeTransform(transforms)
        return Flow(transform, StandardNormal(shape=[1]))

    @staticmethod
    def _sanitize_X(X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return np.nan_to_num(X, nan=0.0, posinf=1e7, neginf=-1e7)

    def fit(self, X, y) -> "NFlowsWrapper":
        import torch

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dev = torch.device(self.device)

        X = self._sanitize_X(X)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        valid = np.isfinite(y)
        X, y = X[valid], y[valid]
        if len(y) == 0:
            raise ValueError("No valid (finite) training samples after sanitization")

        # Standardize features and target for stable spline training.
        self._x_mean = X.mean(axis=0)
        self._x_std = X.std(axis=0) + 1e-8
        self._y_mean = float(y.mean())
        self._y_std = float(y.std()) + 1e-8

        Xn = (X - self._x_mean) / self._x_std
        yn = (y - self._y_mean) / self._y_std

        self._flow = self._build_flow(context_features=Xn.shape[1]).to(dev)
        optimizer = torch.optim.Adam(self._flow.parameters(), lr=self.lr)

        X_t = torch.as_tensor(Xn, dtype=torch.float32, device=dev)
        y_t = torch.as_tensor(yn, dtype=torch.float32, device=dev).reshape(-1, 1)
        n = X_t.shape[0]
        batch = min(self.batch_size, n)

        self._flow.train()
        for _ in range(self.n_epochs):
            perm = torch.randperm(n, device=dev)
            for start in range(0, n, batch):
                idx = perm[start:start + batch]
                optimizer.zero_grad()
                loss = -self._flow.log_prob(inputs=y_t[idx], context=X_t[idx]).mean()
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._flow.parameters(), 5.0)
                optimizer.step()
        self._flow.eval()
        return self

    def _draw_samples(self, X, n_samples: int) -> np.ndarray:
        import torch

        dev = torch.device(self.device)
        Xn = (self._sanitize_X(X) - self._x_mean) / self._x_std
        X_t = torch.as_tensor(Xn, dtype=torch.float32, device=dev)

        with torch.no_grad():
            # (n_test, n_samples, 1) conditional draws in standardized space.
            samples = self._flow.sample(n_samples, context=X_t)
        samples = samples.squeeze(-1).cpu().numpy().astype(np.float64)
        return samples * self._y_std + self._y_mean
