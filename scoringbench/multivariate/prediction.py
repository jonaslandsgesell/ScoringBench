"""Multivariate predictive container: :class:`MultivariateSamplePrediction`.

Design invariant (the whole reason this package is separate from
``scoringbench.univariate``)
-----------------------------------------------------------------
A multivariate forecast here is **purely sample based**.  There is no grid, no
PMF, no CDF, no density, no regridding — *the samples are the forecast*.  A model
produces, for every test instance, a matrix of ``n_draws`` predicted target
vectors, each of dimension ``d``:

    samples : (n_test, n_draws, d)

The multivariate scoring rules (energy score, variogram score,
Dawid–Sebastiani) are then estimated in a Monte-Carlo manner directly from these
draws (see :mod:`scoringbench.multivariate.metrics`).  Storing the forecast as a
matrix of draws is exactly what makes the benchmark extensible: any future model
(normalizing flows, diffusion heads, NORI, …) only has to emit draws.

This container is deliberately **not** named ``DistributionPrediction`` — that
name belongs to the univariate, histogram-based world and carries grid/PMF
semantics that do not exist here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class MultivariateSamplePrediction:
    """A Monte-Carlo ensemble of predicted target vectors.

    This is the multivariate analogue of the univariate ``DistributionPrediction``
    container, but it holds *only draws* — no grid, PMF, or density.

    Parameters
    ----------
    samples : (n_test, n_draws, d) float64 array
        ``samples[i]`` is the ``(n_draws, d)`` matrix of predicted target
        vectors for test instance ``i``.  These draws *are* the predictive
        distribution; every scoring rule is a Monte-Carlo functional of them.
    mean : (n_test, d) float64 array, optional
        Point prediction per test instance.  If omitted it is computed as the
        per-instance sample mean ``samples.mean(axis=1)``.

    Notes
    -----
    ``__post_init__`` validates shape and finiteness only — it performs **no**
    regridding, binning, or density estimation.  The samples are stored
    verbatim.
    """

    samples: np.ndarray
    mean: np.ndarray = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        samples = np.asarray(self.samples, dtype=np.float64)
        if samples.ndim != 3:
            raise ValueError(
                f"samples must be 3-D (n_test, n_draws, d); got shape {samples.shape}"
            )
        n_test, n_draws, d = samples.shape
        if n_test < 1 or n_draws < 1 or d < 1:
            raise ValueError(
                f"samples must have positive extent in every axis; got {samples.shape}"
            )
        if not np.all(np.isfinite(samples)):
            raise ValueError("samples contains non-finite values (nan/inf)")

        # frozen dataclass: use object.__setattr__ to store the coerced array.
        object.__setattr__(self, "samples", samples)

        if self.mean is None:
            mean = samples.mean(axis=1)
        else:
            mean = np.asarray(self.mean, dtype=np.float64)
            if mean.shape != (n_test, d):
                raise ValueError(
                    f"mean must have shape (n_test, d)=({n_test}, {d}); got {mean.shape}"
                )
            if not np.all(np.isfinite(mean)):
                raise ValueError("mean contains non-finite values (nan/inf)")
        object.__setattr__(self, "mean", mean)

    # -- convenience read-only views ---------------------------------------

    @property
    def n_test(self) -> int:
        return self.samples.shape[0]

    @property
    def n_draws(self) -> int:
        return self.samples.shape[1]

    @property
    def d(self) -> int:
        return self.samples.shape[2]
