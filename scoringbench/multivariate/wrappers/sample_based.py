"""Sample-based multivariate wrapper base class.

``SampleBasedWrapper`` is the multivariate analogue of the univariate class of
the same name, but it never touches a grid or PMF: it accumulates ``(n_test, m,
d)`` draws under a wall-clock budget and wraps them in a
:class:`MultivariateSamplePrediction`.

Subclasses only implement :meth:`_draw_samples(X, n) -> (n_test, n, d)`.
"""

from __future__ import annotations

import time

import numpy as np

from ..config import N_DRAWS
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
        """Accumulate draws in chunks under the wall-clock budget -> (n_test, m, d)."""
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
        return np.concatenate(collected, axis=1)

    def predict_ensemble(self, X) -> MultivariateSamplePrediction:
        samples = self._collect_samples(X)
        return MultivariateSamplePrediction(samples=samples)

    def predict(self, X) -> np.ndarray:
        return self._collect_samples(X).mean(axis=1)
