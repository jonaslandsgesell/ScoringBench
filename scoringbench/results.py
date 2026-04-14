"""Result persistence and aggregation.

Public API
----------
save_fold_parquet(fold_data, output_dir, dataset_name, fold_idx)
    Persist raw fold results as JSON.

build_results_row(dataset_config, X, cv_results) -> list[dict]
    Flatten CV results into a list of per-fold rows (one per fold).
"""

import os
from pathlib import Path

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

def save_fold_parquet(fold_data: dict, output_dir: Path, dataset_name: str, fold_idx: int) -> None:
    """Write raw fold results as per-(model, dataset) parquet files.

    Files are written to ``output_dir/raw/{model_name}/{dataset_name}.parquet``.
    One SLURM array job typically owns one dataset, so each file is written by
    exactly one process — no file locking required. Per-dataset files avoid
    concurrency issues when running multiple models or datasets in parallel.
    """
    ds_safe = dataset_name.replace(" ", "_")
    fold_idx_val = fold_idx

    parquet_engine = _detect_parquet_engine()
    if not parquet_engine:
        raise RuntimeError("No parquet engine available (install pyarrow or fastparquet)")

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

        model_raw_dir = output_dir / "raw" / model_name
        model_raw_dir.mkdir(parents=True, exist_ok=True)
        dest_parquet = model_raw_dir / f"{ds_safe}.parquet"

        # Append to the per-(model, dataset) file — safe because a single job
        # owns this file exclusively (one dataset per SLURM array task).
        if dest_parquet.exists():
            existing = pd.read_parquet(dest_parquet, engine=parquet_engine)
            # Idempotency: skip if this fold was already written (e.g. resumed run)
            already_written = (
                (existing["fold"] == fold_idx_val)
            ).any()
            if already_written:
                continue
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
