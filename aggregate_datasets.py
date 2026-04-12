#!/usr/bin/env python3
"""aggregate_datasets.py — Aggregate per-(model, dataset) raw parquet files.

Reads all   output/raw/{model_name}_{dataset_name}.parquet
Writes one  output/{model_name}.parquet  per model.

The output parquet files are 100% structurally identical to the files that
the old single-file-per-model approach produced, so all downstream scripts
(autorank_leaderboard.py, plot_output.py, etc.) require zero changes.

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
    """Aggregate per-(model, dataset) parquet files into per-model parquet files.

    Parameters
    ----------
    raw_dir : directory containing  {model_name}_{dataset_name}.parquet  files
    out_dir : directory to write    {model_name}.parquet  files into

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

    # Collect all parquet files per model
    model_frames: dict[str, list[pd.DataFrame]] = {}
    for entry in sorted(os.listdir(raw_dir)):
        if not entry.endswith(".parquet"):
            continue
        stem = entry[:-8]  # strip ".parquet"
        # Derive model name: everything before the first underscore-preceded
        # dataset segment.  Because both model_name and dataset_name may
        # contain underscores, we identify the split point by matching known
        # dataset names that are embedded in the filename.
        #
        # Strategy: the filename is  "{model_name}_{ds_safe}.parquet".
        # We read the file and trust the "model" column for the model name,
        # and the "dataset" column for the dataset name.  That way we never
        # need string-split heuristics.
        fpath = raw_dir / entry
        try:
            df = pd.read_parquet(fpath, engine=engine)
        except Exception as exc:
            print(f"  Warning: could not read {fpath}: {exc}")
            continue

        if df.empty:
            continue

        # Determine model name from the "model" column if present, otherwise
        # fall back to stripping the dataset suffix (best-effort).
        if "model" in df.columns and df["model"].nunique() == 1:
            model_name = str(df["model"].iloc[0])
        else:
            # Fallback: use the filename stem directly — the downstream reader
            # sets df['model'] = entry[:-8] anyway.
            model_name = stem

        model_frames.setdefault(model_name, []).append(df)

    if not model_frames:
        print("No raw parquet files found — nothing to aggregate.")
        return {}

    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, int] = {}

    for model_name, frames in model_frames.items():
        combined = pd.concat(frames, ignore_index=True)

        # Deduplicate: keep only one row per (dataset, fold) for each model —
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
        print(f"  {model_name}.parquet  ({len(combined)} rows from "
              f"{len(frames)} dataset file(s))")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Aggregate raw per-dataset parquets into per-model parquets")
    parser.add_argument(
        "--raw_dir", default="output/raw",
        help="Directory containing {model}_{dataset}.parquet files (default: output/raw)",
    )
    parser.add_argument(
        "--out_dir", default="output",
        help="Directory to write {model}.parquet files (default: output)",
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
