"""Multi-target OpenML regression datasets passing the KCI screen (source
``multitarget_openml``).

These are exactly the datasets that survive the model-free conditional-dependence
screening in ``scripts/multivariate_screening`` (KCI + residual distance
correlation, Holm-corrected): real OpenML tables with >=2 jointly-measured
numeric (regression) targets whose targets stay dependent given the features.
The curated list ships in :data:`MANIFEST_PATH`
(``multitarget_openml_datasets.json`` next to this module).

Enumeration and loading reuse the generic OpenML multi-target machinery in
:mod:`.native_scoringbench`; this module only pins the screened manifest and the
source name, so a benchmark run (``--source multitarget_openml``) and the screen
read the same frozen suite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from . import native_scoringbench as _nsb

SOURCE_NAME = "multitarget_openml"

#: Curated manifest: the KCI-screened multi-target OpenML regression datasets.
MANIFEST_PATH = Path(__file__).with_name("multitarget_openml_datasets.json")


def enumerate_multitarget_openml(target_dim: int, sample_size: int,
                                 manifest_path: str | Path | None = None) -> list[dict[str, Any]]:
    """Stable-ordered configs for the screened OpenML suite.

    Thin wrapper over :func:`native_scoringbench.enumerate_native_scoringbench`
    that defaults to this source's screened manifest and tags each config with
    ``source = "multitarget_openml"``.
    """
    return _nsb.enumerate_native_scoringbench(
        target_dim, sample_size,
        manifest_path=manifest_path or MANIFEST_PATH,
        source_name=SOURCE_NAME,
    )


def load_multitarget_openml(ds_config: dict[str, Any], target_dim: int | None = None
                            ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load one screened OpenML dataset as ``(X, Y)``.

    Identical to :func:`native_scoringbench.load_native_scoringbench` (the config
    carries ``openml_id`` / ``target_cols``); kept as a named wrapper so the
    source registry stays symmetric.
    """
    return _nsb.load_native_scoringbench(ds_config, target_dim=target_dim)
