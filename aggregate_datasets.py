#!/usr/bin/env python3
"""aggregate_datasets.py — Aggregate per-dataset per-model raw parquet files.

Reads all   output/raw/{model_name}/{dataset_name}.parquet
Writes one  output/{model_name}.parquet  per model.

The new structure organizes files by model, then by dataset, allowing safe
concurrent writes without file conflicts. This script aggregates all datasets
for each model into a single per-model output file.

Usage
-----
    python aggregate_datasets.py                     # default: raw_dir=output/raw, out_dir=output
    python aggregate_datasets.py --raw_dir my/raw    --out_dir my/out
"""

import argparse
import os
from pathlib import Path

import pandas as pd


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


def aggregate(raw_dir: Path, out_dir: Path) -> dict[str, int]:
    """Aggregate per-dataset per-model parquet files into aggregated per-model files.

    Parameters
    ----------
    raw_dir : directory containing {model_name}/{dataset_name}.parquet subdirectories
    out_dir : directory to write {model_name}.parquet aggregated files into

    Returns
    -------
    dict mapping model_name -> number of rows written
    """
    engine = _detect_parquet_engine()
    if not engine:
        raise RuntimeError("No parquet engine found — install pyarrow or fastparquet.")

    if not raw_dir.exists():
        print(f"Raw directory does not exist: {raw_dir}")
        return {}

    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, int] = {}

    # Iterate through each model subdirectory in raw_dir
    for model_entry in sorted(os.listdir(raw_dir)):
        model_dir = raw_dir / model_entry
        
        # Skip if not a directory
        if not model_dir.is_dir():
            continue

        # Find all parquet files in this model directory
        dataset_files = sorted([f for f in os.listdir(model_dir) if f.endswith(".parquet")])
        
        if not dataset_files:
            continue

        model_name = model_entry
        frames = []

        # Read all dataset parquet files for this model
        for ds_fname in dataset_files:
            ds_fpath = model_dir / ds_fname
            try:
                df = pd.read_parquet(ds_fpath, engine=engine)
                if not df.empty:
                    frames.append(df)
            except Exception as exc:
                print(f"  Warning: could not read {ds_fpath}: {exc}")
                continue

        if not frames:
            continue

        # Combine all datasets for this model
        combined = pd.concat(frames, ignore_index=True)

        # Deduplicate: keep only one row per (dataset, fold) —
        # in case a dataset was partially re-run and appended.
        if {"dataset", "fold"}.issubset(combined.columns):
            combined = combined.drop_duplicates(subset=["dataset", "fold"], keep="last")
            combined = combined.sort_values(["dataset", "fold"]).reset_index(drop=True)

        dest = out_dir / f"{model_name}.parquet"
        tmp = dest.with_suffix(".parquet.tmp")
        try:
            combined.to_parquet(tmp, engine=engine, index=False)
            os.replace(str(tmp), str(dest))
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except Exception:
                    pass

        summary[model_name] = len(combined)
        print(f"  {model_name}.parquet  ({len(combined)} rows from {len(dataset_files)} dataset(s))")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Aggregate per-dataset per-model raw parquets into per-model parquets")
    parser.add_argument(
        "--raw_dir", default="output/raw",
        help="Directory containing {model_name}/{dataset_name}.parquet files (default: output/raw)",
    )
    parser.add_argument(
        "--out_dir", default="output",
        help="Directory to write {model_name}.parquet files (default: output)",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    print(f"Aggregating: {raw_dir}  →  {out_dir}")
    summary = aggregate(raw_dir, out_dir)
    if summary:
        total = sum(summary.values())
        print(f"Done. {len(summary)} model file(s), {total} total rows.")
    else:
        print("Nothing written.")


if __name__ == "__main__":
    main()
