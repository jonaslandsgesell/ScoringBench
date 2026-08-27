"""Shared utilities for sample-based ScoringBench wrappers.

``SampleBasedWrapper``
    Base class for genuinely sample-based models (normalizing flows, BART, …).
    Subclasses only implement ``_draw_samples(X, n)`` returning an
    ``(n_test, n)`` array of conditional draws of the target.  The base class
    accumulates draws in chunks under a wall-clock budget
    (``MAX_SAMPLE_SECONDS``, default 120 s — it may sample for at most two
    minutes, then derives the PMF from whatever was collected) and converts the
    pooled draws into a ``DistributionPrediction``.
"""

from __future__ import annotations

import time

import numpy as np

from .base import DistributionPrediction, ProbabilisticWrapper


# ---------------------------------------------------------------------------
# Samples -> DistributionPrediction
# ---------------------------------------------------------------------------

def samples_to_distribution(
    samples: np.ndarray,
    n_bins: int = 100,
    y_range: tuple[float, float] | None = None,
    *,
    train_range: tuple[float, float],
) -> DistributionPrediction:
    """Sanitize conditional draws and build a ``DistributionPrediction``.

    Thin adapter over :meth:`DistributionPrediction.from_samples`: it only
    replaces non-finite draws with a well-defined fallback, then hands the clean
    draws to the constructor, which reads each row's empirical CDF and bins it
    onto ``n_bins`` uniform native bins.

    Parameters
    ----------
    samples : (n_test, n_draws) array
        Conditional samples of the target, one row per test instance.
    n_bins : int
        Number of equally wide native bins per sample (default 100).
    y_range : tuple[float, float], optional
        Fallback range for sanitizing non-finite samples.
    train_range : (y_lo, y_hi), required
        Shared train-target range the density rules regrid onto.
    """
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim == 1:
        samples = samples[:, None]

    # Replace any non-finite draws with the per-row median; if a whole row is
    # non-finite fall back to the y_range midpoint (or 0 when y_range is None).
    if not np.all(np.isfinite(samples)):
        finite = np.isfinite(samples)
        fallback = 0.0 if y_range is None else 0.5 * (float(y_range[0]) + float(y_range[1]))
        row_median = np.array([
            np.median(samples[i, finite[i]]) if finite[i].any() else fallback
            for i in range(samples.shape[0])
        ])
        samples = np.where(finite, samples, row_median[:, None])

    return DistributionPrediction.from_samples(
        samples, n_bins=n_bins, train_range=train_range
    )


# ---------------------------------------------------------------------------
# Sample-based wrapper base
# ---------------------------------------------------------------------------

class SampleBasedWrapper(ProbabilisticWrapper):
    """Base class for models whose predictive density is accessed via sampling.

    Subclasses implement :meth:`_draw_samples`. The PMF is derived from the
    pooled draws; sampling is capped at ``MAX_SAMPLE_SECONDS`` wall-clock
    seconds per ``predict_distribution`` call.

    Class attributes
    ----------------
    N_SAMPLES : int
        Target number of conditional draws per test instance (default 300).
    SAMPLE_CHUNK : int
        Draws requested per call to ``_draw_samples``; the budget is checked
        between chunks. Set equal to ``N_SAMPLES`` for one-shot samplers.
    MAX_SAMPLE_SECONDS : float
        Hard wall-clock cap on sampling (default 120 s). Once exceeded, the PMF
        is built from whatever draws were collected so far.
    N_BINS : int
        Number of equally *wide* native bins derived from the draws via
        :func:`samples_to_distribution`.
    """

    N_SAMPLES: int = 300
    SAMPLE_CHUNK: int = 100
    MAX_SAMPLE_SECONDS: float = 120.0
    N_BINS: int = 100

    def _draw_samples(self, X, n_samples: int) -> np.ndarray:
        """Return an ``(n_test, n_samples)`` array of conditional target draws."""
        raise NotImplementedError

    def _collect_samples(self, X) -> np.ndarray:
        target = int(self.N_SAMPLES)
        chunk = int(self.SAMPLE_CHUNK) or target
        collected: list[np.ndarray] = []
        n_have = 0
        start = time.monotonic()
        while n_have < target:
            take = min(chunk, target - n_have)
            s = np.asarray(self._draw_samples(X, take), dtype=np.float64)
            if s.ndim == 1:
                s = s[:, None]
            collected.append(s)
            n_have += s.shape[1]
            if time.monotonic() - start >= self.MAX_SAMPLE_SECONDS:
                break
        if not collected:
            raise RuntimeError("No samples were drawn.")
        return np.concatenate(collected, axis=1)

    def predict_distribution(self, X) -> DistributionPrediction:
        samples = self._collect_samples(X)
        return samples_to_distribution(
            samples, n_bins=self.N_BINS, train_range=self._y_train_range
        )

    def predict(self, X) -> np.ndarray:
        return self._collect_samples(X).mean(axis=1)
