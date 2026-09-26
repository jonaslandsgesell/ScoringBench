"""Dataset-source registry for the multivariate ScoringBench.

A *source* is a way of producing multivariate ``(X, Y)`` regression problems.
Two are provided:

* ``"synthetic"`` — nonlinear means plus feature-independent residuals from
    randomized simplified R-vines with standard-normal margins (see
  :mod:`scoringbench.multivariate.synthetic_targets`).
* ``"native_scoringbench"`` — **real jointly-measured targets**: curated
  OpenML datasets that were uploaded as multi-target regression and that a
  measured chained-vs-independent screen shows actually need the joint model
  (see :mod:`scoringbench.multivariate.native_scoringbench`). Nothing here is
  constructed — the dependence is whatever the world put in the data. The
  dataset list ships in ``multivariate_datasets.json`` next to that module.

Design (open-closed)
--------------------
The runner and the front script consume a :class:`Source` and never branch on
the source name: a source exposes exactly what they need — a way to *enumerate*
its dataset configs and a *loader* turning one config into ``(X, Y)``. Adding a
new source means adding one :class:`Source` entry to :data:`SOURCES`; no
runner / front-script edits are required. This registry is the *only* new
abstraction introduced for multi-source support — deliberately thin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from . import config as cfg


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
