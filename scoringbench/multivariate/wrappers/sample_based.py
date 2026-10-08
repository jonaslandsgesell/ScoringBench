"""Sample-based multivariate wrapper base class.

``SampleBasedWrapper`` is the multivariate analogue of the univariate class of
the same name, but it never touches a grid or PMF: it accumulates ``(n_test, m,
d)`` draws under a wall-clock budget and wraps them in a
:class:`MultivariateSamplePrediction`.

Subclasses only implement :meth:`_draw_samples(X, n) -> (n_test, n, d)`.
"""

from __future__ import annotations

import time
import warnings

import numpy as np

from .. import config as _config
from ..config import N_DRAWS
from ..convergence import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPSILON,
    DEFAULT_MAX_INSTANCES,
    DEFAULT_MAX_SAMPLES,
    DEFAULT_N_GROUPS,
    assess_sample_prediction,
)
from ..prediction import MultivariateSamplePrediction
from .base import MultivariateWrapper


class SampleBasedWrapper(MultivariateWrapper):
    """Base for models whose predictive law is accessed by sampling.

    Class attributes
    ----------------
    N_SAMPLES : int
        Target number of draws per test instance (defaults to the benchmark-wide
        ``config.N_DRAWS`` so every model emits the same ``m`` — required for a
        fair leaderboard, see ``config.py``).
    SAMPLE_CHUNK : int
        Draws requested per call to :meth:`_draw_samples`; the wall-clock budget
        is checked between chunks.  Set equal to ``N_SAMPLES`` for one-shot
        samplers.
    MAX_SAMPLE_SECONDS : float
        Hard wall-clock cap on sampling per :meth:`predict_ensemble` call.  Once
        exceeded, the prediction is built from whatever draws were collected.
    """

    N_SAMPLES: int = int(N_DRAWS)
    SAMPLE_CHUNK: int = int(N_DRAWS)
    MAX_SAMPLE_SECONDS: float = 120.0

    def _draw_samples(self, X, n_samples: int) -> np.ndarray:
        """Return an ``(n_test, n_samples, d)`` array of conditional draws."""
        raise NotImplementedError

    def _collect_samples(self, X) -> np.ndarray:
        """Accumulate draws in chunks under the wall-clock budget -> (n_test, m, d).

        After the initial ``N_SAMPLES`` draws, if ``config.CONV_ENABLED`` is
        set the multivariate R̂ (Vats & Knudson 2020, det-form) is evaluated on
        a subsample of test instances.  If R̂ has not converged (max over
        instances >= ``CONV_EPSILON``), additional draws are collected in
        ``CONV_BATCH_SIZE`` chunks until convergence, the wall-clock budget, or
        ``CONV_MAX_SAMPLES`` is reached.  No draws are discarded (i.i.d. MC,
        not MCMC).
        """
        target = int(self.N_SAMPLES)
        chunk = int(self.SAMPLE_CHUNK) or target
        collected: list[np.ndarray] = []
        n_have = 0
        start = time.monotonic()
        while n_have < target:
            take = min(chunk, target - n_have)
            s = np.asarray(self._draw_samples(X, take), dtype=np.float64)
            if s.ndim == 2:
                # (n_test, d) single-draw -> promote to (n_test, 1, d)
                s = s[:, None, :]
            if s.ndim != 3:
                raise ValueError(
                    f"_draw_samples must return (n_test, n, d); got shape {s.shape}"
                )
            collected.append(s)
            n_have += s.shape[1]
            if time.monotonic() - start >= self.MAX_SAMPLE_SECONDS:
                break
        if not collected:
            raise RuntimeError("No samples were drawn.")
        samples = np.concatenate(collected, axis=1)

        # --- adaptive top-up: draw more until R̂ < CONV_EPSILON ---------------
        if not getattr(_config, "CONV_ENABLED", False):
            return samples

        while samples.shape[1] < DEFAULT_MAX_SAMPLES:
            if time.monotonic() - start >= self.MAX_SAMPLE_SECONDS:
                break
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
            if diag.rhat_max < DEFAULT_EPSILON:
                break
            extra = np.asarray(self._draw_samples(X, DEFAULT_BATCH_SIZE), dtype=np.float64)
            if extra.ndim == 2:
                extra = extra[:, None, :]
            samples = np.concatenate([samples, extra], axis=1)

        return samples

    def predict_ensemble(self, X) -> MultivariateSamplePrediction:
        samples = self._collect_samples(X)
        return MultivariateSamplePrediction(samples=samples)

    def predict(self, X) -> np.ndarray:
        return self._collect_samples(X).mean(axis=1)
