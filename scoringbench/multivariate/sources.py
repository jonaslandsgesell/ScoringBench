"""Dataset-source registry for the multivariate ScoringBench.

Two sources are provided:

* ``"scoringbench"`` (default) — curated multi-target datasets from OpenML passing
  model-free KCI and residual distance correlation screening. Defined in ``multivariate_datasets.json``.
* ``"synthetic"`` — nonlinear conditional means plus feature-independent,
  jointly sampled R-vine residuals (see :mod:`scoringbench.multivariate.synthetic_targets`).

Design (open-closed)
--------------------
The runner and the front script consume a :class:`Source` and never branch on
the source name: a source exposes exactly what they need — a way to *enumerate*
its dataset configs and a *loader* turning one config into ``(X, Y)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd


@dataclass(frozen=True)
class Source:
    """A named provider of multivariate ``(X, Y)`` datasets.

    Attributes
    ----------
    name:
        Registry key (also used in output-folder names and result rows).
    enumerate_datasets:
        ``(target_dim, sample_size) -> list[dict]`` returning the ordered list
        of dataset-config dicts for this source. Order is stable so
        ``--dataset_index`` (SLURM arrays) maps deterministically.
    load:
        ``(ds_config, target_dim=...) -> (X, Y)``. May raise ``ValueError`` to
        signal that the runner should skip the dataset. ``Y``'s first column is
        the primary target (``target_0``), matching the Source-1 contract.
    """

    name: str
    enumerate_datasets: Callable[[int, int], list[dict]]
    load: Callable[..., tuple[pd.DataFrame, pd.DataFrame]]


# Registry.  ``synthetic`` is imported lazily inside the factory so importing
# this module never hard-requires pyvinecopulib (only the synthetic source does).
def _build_sources() -> dict[str, Source]:
    sources: dict[str, Source] = {}

    from . import synthetic_targets as _syn

    sources["synthetic"] = Source(
        name="synthetic",
        enumerate_datasets=_syn.enumerate_synthetic,
        load=_syn.load_synthetic,
    )

    # Imported here (not at module scope) so importing this registry never
    # hard-requires the ``openml`` client; only enumerating/loading this source
    # touches the network, and an absent manifest simply enumerates nothing.
    from . import native_scoringbench as _nsb

    sources[_nsb.SOURCE_NAME] = Source(
        name=_nsb.SOURCE_NAME,
        enumerate_datasets=_nsb.enumerate_native_scoringbench,
        load=_nsb.load_native_scoringbench,
    )

    # Multi-target OpenML regression datasets passing the KCI dependence screen
    # (scripts/multivariate_screening). Same OpenML loader as ``scoringbench``,
    # but pinned to the frozen screened manifest. Imported lazily for the same
    # reason as above (no hard ``openml`` requirement at registry import).
    from . import multitarget_openml as _mto

    sources[_mto.SOURCE_NAME] = Source(
        name=_mto.SOURCE_NAME,
        enumerate_datasets=_mto.enumerate_multitarget_openml,
        load=_mto.load_multitarget_openml,
    )

    # Univariate ScoringBench datasets promoted to multivariate by promoting the
    # most conditionally-dependent features into targets (scripts/promote_targets).
    # Reuses the univariate loader, so it is imported lazily (no hard dependency
    # at registry import; an absent manifest simply enumerates nothing).
    from . import promoted_scoringbench as _prom

    sources[_prom.SOURCE_NAME] = Source(
        name=_prom.SOURCE_NAME,
        enumerate_datasets=_prom.enumerate_promoted_scoringbench,
        load=_prom.load_promoted_scoringbench,
    )
    return sources


SOURCES: dict[str, Source] = _build_sources()


def get_source(name: str) -> Source:
    """Return the :class:`Source` registered under ``name``.

    Raises ``KeyError`` with the list of valid names on an unknown source.
    """
    try:
        return SOURCES[name]
    except KeyError:
        raise KeyError(
            f"Unknown source {name!r}. Available sources: {sorted(SOURCES)}."
        ) from None
