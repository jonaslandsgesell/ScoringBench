"""scoringbench.multivariate — d>1 targets, purely sample-based.

Design invariants for everything here:

- **Purely sample-based.** Every wrapper emits conditional draws; every scoring
  rule is estimated from those draws directly. No grid, no PMF, no CDF, no
  density, no regridding — ever.
- **No cross-talk with univariate model code.** Wrappers/metrics/estimators are
  built directly on the base libraries (``tabpfn``, ``tabicl``). Only the raw
  *dataset loading* plumbing is reused (see ``datasets.py``).
"""

from . import (
    config,
    cv,
    datasets,
    estimators,
    metrics,
    models,
    prediction,
    results,
    runner,
    utils,
    wrappers,
)

__all__ = [
    "config",
    "cv",
    "datasets",
    "estimators",
    "metrics",
    "models",
    "prediction",
    "results",
    "runner",
    "utils",
    "wrappers",
]
