"""Promoted multivariate datasets (source ``promoted_scoringbench``).

The univariate ScoringBench suite turned multivariate by the feature-promotion
protocol in ``scripts/promote_targets``: for each univariate ``(X, y)`` the most
conditionally-dependent features are promoted into the target block, leaving the
rest as inputs. Unlike ``scoringbench`` / ``multitarget_openml`` (OpenML tables
with *natively* multi-target labels), this source reuses the **univariate loader**
(openml / pmlb / keel) and builds ``Y`` from ``y`` plus the promoted feature
columns -- so it needs no new download path.

The curated list ships in :data:`MANIFEST_PATH` (``promoted_scoringbench_datasets.json`` next
to this module), written by ``scripts/promote_targets/promote.py``. Each entry
carries the univariate dataset config and ``all_target_cols = [y, f_1, ..., f_k]``
ordered most-conditionally-dependent-first, so truncating to the leading
``target_dim`` keeps the strongest block and ``target_0`` is the original target.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from . import config as cfg

SOURCE_NAME = "promoted_scoringbench"

#: Curated manifest of promoted datasets (written by scripts/promote_targets).
MANIFEST_PATH = Path(__file__).with_name("promoted_scoringbench_datasets.json")

#: Reserved column name for the promoted dataset's original univariate target.
TARGET_LABEL = "__y__"
_FEATURE_NAN_FILL = 0.0


def load_manifest(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Read the promoted manifest, or ``[]`` if it has not been built yet."""
    p = Path(path) if path is not None else MANIFEST_PATH
    if not p.exists():
        return []
    payload = json.loads(p.read_text())
    entries = payload["datasets"] if isinstance(payload, dict) else payload
    return list(entries)


def enumerate_promoted_scoringbench(target_dim: int, sample_size: int,
                                    manifest_path: str | Path | None = None,
                                    source_name: str = SOURCE_NAME) -> list[dict[str, Any]]:
    """Stable-ordered configs for every promoted entry admitting ``target_dim``.

    An entry is included at ``d`` only if it was *selected* at ``d`` by the
    promotion screen (``selected_dims``) and has at least ``d`` target columns.
    Ordering is by univariate name so ``--dataset_index`` maps deterministically.
    """
    d = int(target_dim)
    n = int(sample_size)
    configs: list[dict[str, Any]] = []
    for entry in sorted(load_manifest(manifest_path), key=lambda e: str(e["name"])):
        targets = list(entry.get("all_target_cols") or [])
        if len(targets) < d or d not in entry.get("selected_dims", []):
            continue
        configs.append({
            "name": f"{entry['name']}_promoted_d{d}",
            "id": entry["name"],
            "source": source_name,
            "univariate": entry["univariate"],
            "target_cols": targets[:d],
            "all_target_cols": targets,
            "promoted_cols": list(entry.get("promoted_cols") or []),
            "feature_cols": None,
            "target_dim": d,
            "n_samples": n,
        })
    return configs


def load_promoted_scoringbench(ds_config: dict[str, Any], target_dim: int | None = None
                               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load one promoted dataset as ``(X, Y)``.

    ``Y`` = the original univariate target (``target_0``) followed by the leading
    ``target_dim - 1`` promoted feature columns. The *whole* promoted block (every
    column in ``all_target_cols``) is removed from ``X`` so a low-``d`` run cannot
    see a not-yet-promoted target as a feature. Columns are renamed
    ``target_0 .. target_{d-1}`` in promotion order, matching the source contract.
    """
    from scoringbench.univariate.datasets import load_dataset

    d = int(target_dim) if target_dim is not None else int(ds_config["target_dim"])
    X, y = load_dataset(ds_config["univariate"])
    X = pd.DataFrame(X)
    y = pd.to_numeric(pd.Series(y), errors="coerce").astype(float)

    all_targets = list(ds_config["all_target_cols"])
    used = all_targets[:d]                                           # target_0 may be y or a feature
    # Whole-block conditioning: every feature column in the promoted block is
    # removed from X at every d (same contract as native_scoringbench), so a
    # lower-d run can't see a not-yet-used target as a feature.
    block_feats = [c for c in all_targets if c != TARGET_LABEL]
    missing = [c for c in used if c != TARGET_LABEL and c not in X.columns]
    if missing:
        raise ValueError(
            f"promoted dataset {ds_config['id']} no longer has feature column(s) "
            f"{missing}; the univariate dataset changed and the manifest needs rebuilding")

    # Target block in promotion order (original target where the block names it).
    Y = pd.concat(
        [(y.rename(TARGET_LABEL) if col == TARGET_LABEL
          else pd.to_numeric(X[col], errors="coerce").rename(col)) for col in used],
        axis=1,
    ).astype(float)
    keep = Y.notna().all(axis=1)
    if int(keep.sum()) < 2 * cfg.N_FOLDS:
        raise ValueError(
            f"promoted dataset {ds_config['id']}: only {int(keep.sum())} rows have all "
            f"{d} targets present; need at least {2 * cfg.N_FOLDS}")

    # Features: every numeric column except the whole promoted block.
    feat_cols = [c for c in X.columns if c not in block_feats]
    Xf = X.loc[:, feat_cols].apply(
        lambda c: c if pd.api.types.is_numeric_dtype(c) else pd.to_numeric(c, errors="coerce"))
    Xf = Xf.loc[:, Xf.notna().any()]
    Xf, Y = Xf.loc[keep], Y.loc[keep]
    if not ds_config.get("screen_protocol"):
        Xf = Xf.fillna(Xf.median(numeric_only=True)).fillna(_FEATURE_NAN_FILL)
    if Xf.shape[1] == 0:
        raise ValueError(
            f"promoted dataset {ds_config['id']}: no usable feature columns remain")

    Y.columns = [f"target_{i}" for i in range(Y.shape[1])]
    return Xf.reset_index(drop=True), Y.reset_index(drop=True)
