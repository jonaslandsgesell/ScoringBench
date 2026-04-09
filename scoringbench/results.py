"""Result persistence and aggregation.

Public API
----------
save_fold_parquet(fold_data, output_dir, dataset_name, fold_idx)
    Persist raw fold results as JSON.

build_results_row(dataset_config, X, cv_results) -> list[dict]
    Flatten CV results into a list of per-fold rows (one per fold).
"""

import json
import os
import time
from pathlib import Path

import pandas as pd
from filelock import FileLock, Timeout as FileLockTimeout

from .utils import make_json_serializable

# How long (seconds) to wait for a per-parquet lock before giving up.
# SLURM jobs on the same model can queue briefly; 10 min is generous.
_LOCK_TIMEOUT = 600


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
    """Write raw fold results as per-model parquet files.
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
        lock_path    = dest_parquet.with_suffix(".parquet.lock")

        # Serialise concurrent writes from parallel SLURM array jobs.
        # The lock covers the full read-modify-write so no two processes
        # can interleave their updates to the same parquet file.
        try:
            with FileLock(str(lock_path), timeout=_LOCK_TIMEOUT):
                # Re-read INSIDE the lock: a concurrent job may have appended
                # rows while this job was computing its fold result.
                if dest_parquet.exists():
                    existing = pd.read_parquet(dest_parquet, engine=parquet_engine)
                    # If this (dataset, fold) was already committed (e.g. by a
                    # concurrent job), keep the existing row — never overwrite a
                    # newer result with a stale one.
                    already_written = (
                        (existing.get("dataset") == ds_safe) &
                        (existing.get("fold")    == fold_idx_val)
                    ).any()
                    if already_written:
                        continue
                    combined = pd.concat([existing, row_df], ignore_index=True)
                else:
                    combined = row_df

                _atomic_parquet_write(combined, dest_parquet, parquet_engine)

        except FileLockTimeout:
            raise RuntimeError(
                f"Could not acquire lock for {dest_parquet} within "
                f"{_LOCK_TIMEOUT}s. Another job may be stuck."
            )
        finally:
            # If a stale lock file remains (older than _LOCK_TIMEOUT), remove it.
            # This helps recover from jobs that died without releasing the lock.
            try:
                if lock_path.exists():
                    age = time.time() - lock_path.stat().st_mtime
                    if age > _LOCK_TIMEOUT:
                        try:
                            lock_path.unlink()
                        except Exception:
                            pass
            except Exception:
                pass


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
