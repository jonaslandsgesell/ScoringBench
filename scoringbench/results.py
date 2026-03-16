"""Result persistence and aggregation.

Public API
----------
save_fold_json(fold_data, output_dir, dataset_name, fold_idx)
    Persist raw fold results as JSON.

build_results_row(dataset_config, X, cv_results) -> list[dict]
    Flatten CV results into a list of per-fold CSV rows (one per fold).

save_detailed_csv(rows, output_dir)
    Write / append benchmark_results_detailed.csv.

save_aggregated_csv(detailed_df, output_dir)
    Compute per-dataset / per-model statistics and write
    benchmark_results_aggregated.csv.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import make_json_serializable


# Detect parquet engine availability
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
    # Ensure parent exists
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Use pandas to_parquet; will raise if engine missing
        if engine:
            df.to_parquet(tmp, engine=engine, index=False)
        else:
            # This will raise a ValueError if no engine is available
            df.to_parquet(tmp, index=False)
        # Atomic replace
        os.replace(str(tmp), str(dest))
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Fold-level JSON persistence
# ---------------------------------------------------------------------------

def save_fold_json(fold_data: dict, output_dir: Path, dataset_name: str, fold_idx: int) -> None:
    """Write raw fold results to per-model layout:

    <output_dir>/<model_name>/<dataset_name>/fold_<N>/results.json

    `fold_data` is expected to be a dict mapping model_name -> metrics (same
    shape as produced by `run_fold`). We iterate models present in the dict
    and persist each model's metrics independently.
    """
    ds_safe = dataset_name.replace(" ", "_")

    # fold_data may include the key "fold"; remove it when writing per-model
    fold_idx_val = fold_idx

    # Attempt to persist as a per-model parquet file. If parquet engine is
    # unavailable the run will fail (no legacy JSON fallback).
    parquet_engine = _detect_parquet_engine()

    if not parquet_engine:
        raise RuntimeError("No parquet engine available; legacy JSON fallback removed")

    for model_name, metrics in list(fold_data.items()):
        if model_name == "fold":
            continue

        # Prepare a single-row dataframe for this fold
        payload = dict(metrics)
        payload["fold"] = fold_idx_val
        payload["dataset"] = ds_safe
        payload["model"] = model_name
        payload = make_json_serializable(payload)
        row_df = pd.DataFrame([payload])

        dest_parquet = output_dir / f"{model_name}.parquet"

        # Persist/update per-model parquet (no legacy JSON fallback)
        try:
            if dest_parquet.exists():
                existing = pd.read_parquet(dest_parquet, engine=parquet_engine)
                # Drop any existing row for same dataset+fold to allow re-run/upserts
                mask = ~(
                    (existing.get("dataset") == ds_safe) &
                    (existing.get("fold") == fold_idx_val)
                )
                existing_clean = existing[mask]
                combined = pd.concat([existing_clean, row_df], ignore_index=True)
            else:
                combined = row_df

            _atomic_parquet_write(combined, dest_parquet, parquet_engine)
        except Exception:
            # Propagate errors so issues are visible (no JSON fallback)
            raise


# ---------------------------------------------------------------------------
# Row flattening
# ---------------------------------------------------------------------------

def build_results_rows(
    dataset_config: dict,
    X: pd.DataFrame,
    cv_results: list[dict],
) -> list[dict]:
    """Convert a list of fold dicts into flat CSV rows.

    One row per (dataset, model, fold) combination.
    """
    rows = []
    model_names = [k for k in cv_results[0] if k != "fold"]

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
            }
            row.update(metrics)
            rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# CSV saving
# ---------------------------------------------------------------------------

def save_detailed_csv(rows: list[dict], output_dir: Path) -> pd.DataFrame:
    """Append rows to benchmark_results_detailed.csv (or create it)."""
    dest = output_dir / "benchmark_results_detailed.csv"
    new_df = pd.DataFrame(rows)
    if dest.exists():
        existing = pd.read_csv(dest)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(dest, index=False)
    return combined


def save_aggregated_csv(detailed_df: pd.DataFrame, output_dir: Path) -> None:
    """Compute mean ± std per (dataset, model) and write aggregated CSV."""
    if detailed_df.empty:
        return

    numeric_cols = detailed_df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude index-like columns
    numeric_cols = [c for c in numeric_cols if c not in ("fold",)]

    agg = (
        detailed_df.groupby(["dataset", "model"])[numeric_cols]
        .agg(["mean", "std"])
        .round(4)
    )
    agg.to_csv(output_dir / "benchmark_results_aggregated.csv")
