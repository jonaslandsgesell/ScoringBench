"""Curated **real** multi-target regression datasets (source
``native_scoringbench``).

Sources 1 and 2 of the multivariate benchmark are both *constructions*: source 1
promotes features of a 1-D dataset into targets, source 2 draws copula-coupled
residuals so independence fails by design. Neither answers the question a
practitioner actually has -- *on real data that was collected as multi-target,
does modelling the joint distribution beat a product of marginals?* This source
supplies that: datasets whose targets were measured jointly in the world, with
the target block declared by whoever uploaded the data.

The dataset list is shipped in :data:`MANIFEST_PATH` (``multivariate_datasets.json``
next to this module). Each entry was curated to have a numeric multi-target block with a
measured joint-vs-independent gap (a chained model beats a product-of-marginals
model), but the benchmark does not re-read that provenance: only the fields it
actually uses are kept in the manifest (``openml_id``, ``name``, ``target_cols``,
``feature_cols``, ``drop_cols``, ``n_rows``). Less info shipped, less to go stale.

Target ordering
---------------
``target_cols`` is ordered so that truncating to the first ``target_dim`` entries
keeps the most strongly conditionally-dependent block. That makes the default
``target_dim = 2`` run a meaningful 2-D problem rather than an arbitrary slice.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from . import config as cfg

SOURCE_NAME = "native_scoringbench"

#: Curated manifest (see module docstring for the bar an entry must clear).
MANIFEST_PATH = Path(__file__).with_name("multivariate_datasets.json")

#: Rows with a missing value in any *used* target are dropped (rf1/rf2/osales
#: carry target NaNs); remaining feature NaNs are median-imputed, matching the
#: harvest pipeline so the benchmark measures what the screen measured.
_FEATURE_NAN_FILL = 0.0


def load_manifest(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Read the curated manifest, or return ``[]`` if it has not been built yet.

    An absent manifest is not an error: it means the curation pipeline has not
    run in this checkout. The source then enumerates nothing and the runner
    simply has no ``native_scoringbench`` datasets to score, rather than
    failing.
    """
    p = Path(path) if path is not None else MANIFEST_PATH
    if not p.exists():
        return []
    payload = json.loads(p.read_text())
    entries = payload["datasets"] if isinstance(payload, dict) else payload
    return list(entries)


def enumerate_native_scoringbench(target_dim: int, sample_size: int
                                  ) -> list[dict[str, Any]]:
    """Stable-ordered configs for every manifest entry with enough targets.

    ``target_dim`` selects the leading ``target_dim`` columns of each entry's
    ``target_cols`` (ordered most-dependent-first). Entries with fewer admissible
    targets than ``target_dim`` are skipped rather than padded, so a run at a
    given ``target_dim`` only contains datasets that genuinely have that many
    jointly-measured continuous targets.

    Ordering is by OpenML id so ``--dataset_index`` (SLURM arrays) maps
    deterministically regardless of manifest write order.
    """
    d = int(target_dim)
    n = int(sample_size)
    configs: list[dict[str, Any]] = []
    for entry in sorted(load_manifest(), key=lambda e: int(e["openml_id"])):
        targets = list(entry.get("target_cols") or [])
        if len(targets) < d:
            continue
        used = targets[:d]
        configs.append({
            "name": f"{entry['name']}_d{d}",
            "id": int(entry["openml_id"]),
            "source": SOURCE_NAME,
            "openml_id": int(entry["openml_id"]),
            "openml_name": entry["name"],
            "target_cols": used,
            "all_target_cols": targets,
            "feature_cols": list(entry["feature_cols"]) if entry.get("feature_cols") else None,
            # Leakage columns the curation pipeline removed (an exact aggregate of
            # the targets, or a row/time identifier). These must be dropped here
            # too: with them present the joint-vs-independent gap does not
            # reproduce, because the aggregate hands the chained arm a target it
            # has not sampled yet.
            "drop_cols": list(entry.get("drop_cols") or []),
            "target_dim": d,
            "n_samples": n,
        })
    return configs


def _fetch(openml_id: int) -> pd.DataFrame:
    """Download (or read from the OpenML cache) one dataset as a DataFrame."""
    import openml

    ds = openml.datasets.get_dataset(int(openml_id), download_data=True,
                                     download_qualities=False)
    df, _, _, _ = ds.get_data(dataset_format="dataframe")
    return df


def load_native_scoringbench(ds_config: dict[str, Any], target_dim: int | None = None
                             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load one curated dataset as ``(X, Y)``.

    ``Y``'s columns are renamed ``target_0 .. target_{d-1}`` in manifest order,
    matching the source-1 contract that ``target_0`` is the primary target.
    Features are every non-target column (or the manifest's ``feature_cols``
    when the curation pipeline pinned a subset), coerced to numeric; columns that
    coerce entirely to NaN are dropped, since a wrapper cannot use them.

    Raises ``ValueError`` -- which the runner treats as "skip this dataset" --
    when the file no longer contains the manifest's target columns or too few
    rows survive target-NaN removal.
    """
    d = int(target_dim) if target_dim is not None else int(ds_config["target_dim"])
    targets = list(ds_config["all_target_cols"])[:d]
    df = _fetch(ds_config["openml_id"])

    missing = [c for c in targets if c not in df.columns]
    if missing:
        raise ValueError(
            f"OpenML {ds_config['openml_id']} ({ds_config['openml_name']}) no longer "
            f"contains manifest target column(s) {missing}; the dataset version on "
            "OpenML has changed and the manifest needs regenerating")

    Y = df[targets].apply(pd.to_numeric, errors="coerce").astype(float)
    keep = Y.notna().all(axis=1)
    if int(keep.sum()) < 2 * cfg.N_FOLDS:
        raise ValueError(
            f"OpenML {ds_config['openml_id']} ({ds_config['openml_name']}): only "
            f"{int(keep.sum())} rows have all {d} targets present; need at least "
            f"{2 * cfg.N_FOLDS} for {cfg.N_FOLDS}-fold CV")

    drop_cols = set(ds_config.get("drop_cols") or [])
    feat_cols = ds_config.get("feature_cols")
    if feat_cols:
        # Never let a declared target leak in as a feature, even if the manifest
        # pinned one: the leading `d` targets are the label block for this run.
        cols = [c for c in feat_cols
                if c in df.columns and c not in targets and c not in drop_cols]
    else:
        # Drop the *whole* declared target block, not just the `d` in use, so a
        # target_dim=2 run cannot see target_3 as a feature (that would make the
        # joint problem trivially separable and destroy the comparison).
        dropped = set(ds_config["all_target_cols"]) | drop_cols
        cols = [c for c in df.columns if c not in dropped]

    X = df.loc[:, cols].apply(
        lambda c: c if pd.api.types.is_numeric_dtype(c) else pd.to_numeric(c, errors="coerce"))
    X = X.loc[:, X.notna().any()]                      # drop all-NaN (non-coercible) columns
    X, Y = X.loc[keep], Y.loc[keep]
    X = X.fillna(X.median(numeric_only=True)).fillna(_FEATURE_NAN_FILL)
    if X.shape[1] == 0:
        raise ValueError(
            f"OpenML {ds_config['openml_id']} ({ds_config['openml_name']}): no usable "
            "numeric feature columns remain after removing the target block")

    Y.columns = [f"target_{i}" for i in range(Y.shape[1])]
    return X.reset_index(drop=True), Y.reset_index(drop=True)
