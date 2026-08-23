"""Result persistence and aggregation for the multivariate benchmark.

Output layout is 1:1 compatible with ``autorank_leaderboard.py``: one parquet
row per (dataset, model, fold) under ``output_dir/raw/{model}/{dataset}.parquet``
with a flat set of numeric metric columns (all lower-is-better scoring rules).

Public API
----------
save_fold_parquet(fold_data, output_dir, dataset_name, fold_idx)
build_results_rows(dataset_config, X, Y, cv_results) -> list[dict]
"""

import os
from pathlib import Path

import pandas as pd

from .utils import make_json_serializable


def _detect_parquet_engine():
    try:
        import pyarrow  # noqa: F401
        return "pyarrow"
    except Exception:
        try:
            import fastparquet  # noqa: F401
            return "fastparquet"
        except Exception:
            return None


def _atomic_parquet_write(df: pd.DataFrame, dest: Path, engine: str | None) -> None:
    tmp = dest.with_suffix(".parquet.tmp")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        if engine:
            df.to_parquet(tmp, engine=engine, index=False)
        else:
            df.to_parquet(tmp, index=False)
        os.replace(str(tmp), str(dest))
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Fold-level parquet persistence
# ---------------------------------------------------------------------------

def save_fold_parquet(fold_data: dict, output_dir: Path, dataset_name: str, fold_idx: int) -> None:
    """Write raw fold results as per-(model, dataset) parquet files.

    Files are written to ``output_dir/raw/{model_name}/{dataset_name}.parquet``.
    One row per fold; error records are overwritten on re-run.
    """
    ds_safe = dataset_name.replace(" ", "_")
    fold_idx_val = fold_idx

    parquet_engine = _detect_parquet_engine()
    if not parquet_engine:
        raise RuntimeError("No parquet engine available (install pyarrow or fastparquet)")

    for model_name, metrics in list(fold_data.items()):
        if model_name == "fold":
            continue

        payload = dict(metrics)
        payload["fold"] = fold_idx_val
        payload["dataset"] = ds_safe
        payload["model"] = model_name
        payload = make_json_serializable(payload)
        row_df = pd.DataFrame([payload])

        model_raw_dir = output_dir / "raw" / model_name
        model_raw_dir.mkdir(parents=True, exist_ok=True)
        dest_parquet = model_raw_dir / f"{ds_safe}.parquet"

        if dest_parquet.exists():
            existing = pd.read_parquet(dest_parquet, engine=parquet_engine)
            fold_mask = existing["fold"] == fold_idx_val
            if fold_mask.any():
                existing_row = existing[fold_mask].iloc[0]
                prev_err = existing_row.get("error", None) if "error" in existing.columns else None
                is_prev_error = prev_err is not None and not (
                    isinstance(prev_err, float) and pd.isna(prev_err)
                )
                if not is_prev_error:
                    continue
                existing = existing[~fold_mask]
            combined = pd.concat([existing, row_df], ignore_index=True)
        else:
            combined = row_df

        _atomic_parquet_write(combined, dest_parquet, parquet_engine)


# ---------------------------------------------------------------------------
# Row flattening
# ---------------------------------------------------------------------------

def build_results_rows(
    dataset_config: dict,
    X: pd.DataFrame,
    Y: pd.DataFrame,
    cv_results: list[dict],
) -> list[dict]:
    """Convert a list of fold dicts into flat rows (one per dataset/model/fold)."""
    rows = []
    if not cv_results:
        return rows
    model_names = [k for k in cv_results[0] if k != "fold"]

    n_targets = Y.shape[1] if hasattr(Y, "shape") else None

    for fold_data in cv_results:
        fold_idx = fold_data["fold"]
        for model_name in model_names:
            metrics = fold_data[model_name]
            row = {
                "dataset": dataset_config["name"],
                "dataset_source": dataset_config.get("source", "openml"),
                "dataset_id": dataset_config.get("id", dataset_config.get("loader", "N/A")),
                "model": model_name,
                "fold": fold_idx,
                "n_samples": len(X),
                "n_features": X.shape[1],
                "n_targets": n_targets,
            }
            row.update(metrics)
            rows.append(row)

    return rows
