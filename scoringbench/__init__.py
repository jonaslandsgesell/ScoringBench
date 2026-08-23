"""ScoringBench — a benchmark suite for tabular distributional regression models.

This top-level package intentionally exposes *only* the version. The two
self-contained subpackages must be imported explicitly:

    from scoringbench import univariate
    from scoringbench import multivariate

Rationale: eagerly importing both subpackages would drag every optional heavy
dependency of both worlds into any ``import scoringbench``, and would let a
stale ``scoringbench.metrics`` reference keep resolving. There are deliberately
no back-compat shims: ``scoringbench.metrics`` (and the other former top-level
modules) are now an unambiguous ``ImportError`` — use
``scoringbench.univariate.metrics`` instead.
"""
from .version import __version__

__all__ = ["__version__"]
