"""Dataset-source registry for the multivariate ScoringBench.

A *source* is a way of producing multivariate ``(X, Y)`` regression problems.
Two are provided:

* ``"scoringbench"`` — the original **feature-promotion** construction: take a
  standard 1-D regression dataset and promote the (d-1) features carrying the
  strongest conditional (residual) cross-target dependence into targets (see
  :mod:`scoringbench.multivariate.datasets`).
* ``"synthetic"`` — **explicitly-constructed dependent targets**: draw
  copula-coupled residuals with a fixed vine copula so a product-of-marginals
  (independent) model fails *by construction* (see
  :mod:`scoringbench.multivariate.synthetic_targets`).

Design (open-closed)
--------------------
The runner and the front script consume a :class:`Source` and never branch on
the source name: a source exposes exactly what they need — a way to *enumerate*
its dataset configs and a *loader* turning one config into ``(X, Y)``. Adding a
third source means adding one :class:`Source` entry to :data:`SOURCES`; no
runner / front-script edits are required. This registry is the *only* new
abstraction introduced for multi-source support — deliberately thin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from . import config as cfg
from .datasets import (
    get_DATASETS_CONFIG,
    load_multivariate_dataset,
    validate_datasets,
)


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


def _enumerate_scoringbench(target_dim: int, sample_size: int) -> list[dict]:
    """Feature-promotion source: the shared, validated ScoringBench datasets.

    ``target_dim`` / ``sample_size`` are accepted for a uniform source
    signature; the promotion loader consumes ``target_dim`` per-dataset and the
    runner applies ``sample_size`` at fold time, so neither is needed here.
    """
    return validate_datasets(get_DATASETS_CONFIG())


# Registry.  ``synthetic`` is imported lazily inside the factory so importing
# this module never hard-requires pyvinecopulib (only the synthetic source does).
def _build_sources() -> dict[str, Source]:
    sources: dict[str, Source] = {
        "scoringbench": Source(
            name="scoringbench",
            enumerate_datasets=_enumerate_scoringbench,
            load=load_multivariate_dataset,
        ),
    }

    from . import synthetic_targets as _syn

    sources["synthetic"] = Source(
        name="synthetic",
        enumerate_datasets=_syn.enumerate_synthetic,
        load=_syn.load_synthetic,
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
