"""Genuinely-native multi-target regression datasets (source ``dheur``).

Ten datasets from the Dheur et al. (2025) multi-output conformal-regression
benchmark. They ship as flat data files next to this module in
``data/dheur/`` (CSV / ARFF); nothing is downloaded at run time.

The exact ``(X, Y)`` split per dataset mirrors the upstream benchmark's own
loader (``moc/datamodules/real_datamodule.py``): for each dataset a fixed
*target block* is declared (either by name or as the trailing ``k`` columns),
``X`` is everything else, and a couple of datasets are ``dropna``-ed first.

Target ordering / ``target_dim``
--------------------------------
At ``target_dim = d`` the target block is truncated to its leading ``d`` columns
(so ``d = 2`` takes the first two targets), exactly like
:mod:`scoringbench.multivariate.native_scoringbench`. A dataset qualifies for a
given ``d`` only if its target block has at least ``d`` columns; all ten have
``>= 2``.

Preprocessing
-------------
``X`` and ``Y`` are coerced with the *same* numeric handling as
:mod:`native_scoringbench`: ``Y`` is coerced to float and rows missing any used
target are dropped; ``X`` keeps every non-target, non-dropped column coerced to
numeric, drops columns that coerce entirely to NaN (this is how non-numeric /
nominal feature columns are discarded, including ARFF nominals), and fills
remaining feature NaNs with the column median then ``0.0``. This deliberately
differs from the upstream benchmark's heavier preprocessing (cardinality /
pseudo-categorical pruning, one-hot) so the measurement matches the rest of the
ScoringBench multivariate pool.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from . import config as cfg

SOURCE_NAME = "dheur"

#: Flat data files copied from the upstream benchmark's ``data/`` tree.
DATA_DIR = Path(__file__).with_name("data") / "dheur"

#: Feature-NaN fill, matching :mod:`native_scoringbench` so the two native
#: sources preprocess identically.
_FEATURE_NAN_FILL = 0.0


#: Ordered dataset catalogue (stable order -> deterministic ``--dataset_index``).
#:
#: ``targets`` is either an explicit list of target-column *names* or an int
#: ``k`` meaning "the trailing ``k`` columns". ``n_targets`` is that block's
#: size (used by :func:`enumerate_dheur` to test qualification without opening
#: the file). ``read_kwargs`` are passed to ``pandas.read_csv``; ``dropna`` drops
#: incomplete rows *before* the split (upstream does this for ansur2/calcofi).
_DATASETS: list[dict[str, Any]] = [
    {"dataset": "households", "file": "households.csv", "format": "csv",
     "read_kwargs": {"index_col": 0}, "dropna": False,
     "targets": ["inc", "food", "house", "utili"], "n_targets": 4,
     "drop_cols": ["newid", "inc.a"]},
    {"dataset": "air", "file": "air.csv", "format": "csv",
     "dropna": False, "targets": 6, "n_targets": 6, "drop_cols": []},
    {"dataset": "births1", "file": "births1.csv", "format": "csv",
     "dropna": False, "targets": 2, "n_targets": 2, "drop_cols": []},
    {"dataset": "ansur2", "file": "ansur2.csv", "format": "csv",
     "dropna": True, "targets": 2, "n_targets": 2, "drop_cols": []},
    {"dataset": "calcofi", "file": "calcofi.csv", "format": "csv",
     "dropna": True, "targets": 2, "n_targets": 2, "drop_cols": []},
    {"dataset": "bio", "file": "bio.csv", "format": "csv",
     "dropna": False, "targets": ["F7", "F9"], "n_targets": 2, "drop_cols": []},
    {"dataset": "house", "file": "house.csv", "format": "csv",
     "dropna": False, "targets": ["lat", "price"], "n_targets": 2, "drop_cols": []},
    {"dataset": "taxi", "file": "taxi.csv", "format": "csv",
     "dropna": False, "targets": 2, "n_targets": 2, "drop_cols": []},
    {"dataset": "oes97", "file": "oes97.arff", "format": "arff",
     "dropna": False, "targets": 16, "n_targets": 16, "drop_cols": []},
    {"dataset": "oes10", "file": "oes10.arff", "format": "arff",
     "dropna": False, "targets": 16, "n_targets": 16, "drop_cols": []},
]

#: Lookup by plain dataset name (``load_dheur`` resolves its spec from here, so a
#: config dict carried through SLURM need only name the dataset).
_BY_NAME: dict[str, dict[str, Any]] = {d["dataset"]: d for d in _DATASETS}


def enumerate_dheur(target_dim: int, sample_size: int) -> list[dict[str, Any]]:
    """Stable-ordered configs for every dheur dataset with enough targets.

    ``target_dim`` selects the leading ``target_dim`` columns of each dataset's
    target block. Datasets whose block has fewer than ``target_dim`` columns are
    skipped (none are, for ``d <= 2``), so a run at a given ``target_dim`` only
    contains datasets that genuinely have that many jointly-measured targets.

    Order is the fixed catalogue order so ``--dataset_index`` (SLURM arrays) maps
    deterministically.
    """
    d = int(target_dim)
    n = int(sample_size)
    configs: list[dict[str, Any]] = []
    for spec in _DATASETS:
        if int(spec["n_targets"]) < d:
            continue
        # Explicit-name blocks can advertise the used/all target names up front;
        # trailing-``k`` blocks resolve their names at load time (needs the file).
        targets = spec["targets"]
        all_target_cols = list(targets) if isinstance(targets, list) else None
        target_cols = all_target_cols[:d] if all_target_cols is not None else None
        configs.append({
            "name": f"{spec['dataset']}_d{d}",
            "dataset": spec["dataset"],
            "id": spec["dataset"],
            "source": SOURCE_NAME,
            "provenance": "native",
            "file": spec["file"],
            "format": spec["format"],
            "read_kwargs": dict(spec.get("read_kwargs") or {}),
            "dropna": bool(spec.get("dropna", False)),
            "targets": list(targets) if isinstance(targets, list) else int(targets),
            "n_targets": int(spec["n_targets"]),
            "drop_cols": list(spec.get("drop_cols") or []),
            "target_cols": target_cols,
            "all_target_cols": all_target_cols,
            "target_dim": d,
            "n_samples": n,
            "sample_size": n,
        })
    return configs


def _read_frame(spec: dict[str, Any]) -> pd.DataFrame:
    """Read one dheur data file (CSV or ARFF) into a DataFrame.

    ARFF ``bytes``/``object`` columns (nominal labels, string fields) are decoded
    to ``str`` so later numeric coercion can discard them cleanly rather than
    choke on raw ``bytes``.
    """
    path = DATA_DIR / spec["file"]
    if spec["format"] == "arff":
        from scipy.io import arff

        data, _meta = arff.loadarff(path)
        df = pd.DataFrame(data)
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].map(
                    lambda v: v.decode("utf-8", "ignore")
                    if isinstance(v, (bytes, bytearray)) else v)
        return df
    return pd.read_csv(path, **(spec.get("read_kwargs") or {}))


def _target_block(df: pd.DataFrame, spec: dict[str, Any]) -> list[str]:
    """Resolve a dataset's full target block to a concrete list of column names."""
    targets = spec["targets"]
    if isinstance(targets, list):
        return list(targets)
    k = int(targets)
    return list(df.columns[-k:])


def load_dheur(ds_config: dict[str, Any], target_dim: int | None = None
               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load one dheur dataset as ``(X, Y)``.

    ``Y`` is the leading ``target_dim`` columns of the dataset's target block,
    coerced to float and renamed ``target_0 .. target_{d-1}`` (``target_0`` is
    the primary target, matching the Source-1 contract). ``X`` is every other
    column (minus ``drop_cols`` and the *whole* target block, so a ``d = 2`` run
    never sees a later target as a feature), coerced to numeric; columns that
    coerce entirely to NaN are dropped and remaining feature NaNs are filled.

    Raises ``ValueError`` -- which the runner treats as "skip this dataset" --
    when the file lacks a named target column, the block has fewer than
    ``target_dim`` columns, or too few rows survive target-NaN removal.
    """
    d = int(target_dim) if target_dim is not None else int(ds_config["target_dim"])
    spec = _BY_NAME[ds_config["dataset"]]

    df = _read_frame(spec)
    if spec.get("dropna", False):
        df = df.dropna()

    block = _target_block(df, spec)
    if len(block) < d:
        raise ValueError(
            f"dheur dataset {spec['dataset']!r}: target block has {len(block)} "
            f"column(s), fewer than the requested target_dim={d}")
    used = block[:d]

    missing = [c for c in used if c not in df.columns]
    if missing:
        raise ValueError(
            f"dheur dataset {spec['dataset']!r} is missing target column(s) "
            f"{missing}; the shipped data file has changed")

    Y = df[used].apply(pd.to_numeric, errors="coerce").astype(float)
    Y.attrs["categorical_targets"] = [column for column in used
        if not pd.api.types.is_numeric_dtype(df[column].dtype)
        or pd.api.types.is_bool_dtype(df[column].dtype)]
    ds_config["target_cols"] = used
    ds_config["all_target_cols"] = block
    keep = Y.notna().all(axis=1)
    if int(keep.sum()) < 2 * cfg.N_FOLDS:
        raise ValueError(
            f"dheur dataset {spec['dataset']!r}: only {int(keep.sum())} rows have "
            f"all {d} targets present; need at least {2 * cfg.N_FOLDS} for "
            f"{cfg.N_FOLDS}-fold CV")

    # Drop the *whole* declared target block (not just the ``d`` in use) plus any
    # explicit leakage columns, so a target_dim=d run cannot see a later target
    # as a feature.
    dropped = set(block) | set(spec.get("drop_cols") or [])
    cols = [c for c in df.columns if c not in dropped]
    X = df.loc[:, cols].apply(
        lambda c: c if pd.api.types.is_numeric_dtype(c)
        else pd.to_numeric(c, errors="coerce"))
    X = X.loc[:, X.notna().any()]          # drop all-NaN (non-coercible) columns
    X, Y = X.loc[keep], Y.loc[keep]
    if not ds_config.get("screen_protocol"):
        X = X.fillna(X.median(numeric_only=True)).fillna(_FEATURE_NAN_FILL)
    if X.shape[1] == 0:
        raise ValueError(
            f"dheur dataset {spec['dataset']!r}: no usable numeric feature "
            "columns remain after removing the target block")

    Y.columns = [f"target_{i}" for i in range(Y.shape[1])]
    return X.reset_index(drop=True), Y.reset_index(drop=True)
