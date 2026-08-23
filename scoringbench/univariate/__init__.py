"""ScoringBench (univariate) — 1-D targets on a joint-grid representation.

Everything that existed at the old ``scoringbench`` top level now lives here.
The eager re-exports below preserve the previous behaviour so that
``scoringbench.univariate.metrics`` etc. resolve exactly as
``scoringbench.metrics`` used to.
"""
from . import config, datasets, wrappers, models, metrics, cv, runner, results, utils
from ..version import __version__

__all__ = ["config", "datasets", "wrappers", "models", "metrics", "cv", "runner", "results", "utils", "__version__"]
