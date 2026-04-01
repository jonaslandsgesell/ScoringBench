#!/usr/bin/env python3
"""check_parquet.py — Inspect ScoringBench parquet result files.

Usage
-----
    python scripts/check_parquet.py output/TabPFNWrapper.parquet [...]
    python scripts/check_parquet.py output/*.parquet --min-folds 5

For each file the script prints:
  - Total rows (runs)
  - Unique datasets and folds
  - Runs per dataset × fold table
  - Datasets with fewer than --min-folds folds (if requested)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def _check_single(path: Path, min_folds: int) -> None:
    print(f"\n{'=' * 64}")
    print(f"File : {path}")
    print(f"Size : {path.stat().st_size / 1024:.1f} KB")

    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        print(f"  ERROR reading file: {exc}")
        return

    total_rows = len(df)
    print(f"Total runs (rows)  : {total_rows}")
    print(f"Columns            : {list(df.columns)}")

    if "dataset" not in df.columns:
        print("  (no 'dataset' column — cannot break down by dataset/fold)")
        return

    n_datasets = df["dataset"].nunique()
    print(f"Unique datasets    : {n_datasets}")

    if "model" in df.columns:
        models = sorted(df["model"].unique())
        print(f"Models             : {models}")

    if "fold" in df.columns:
        folds_per_ds = df.groupby("dataset")["fold"].nunique().sort_index()
        min_f = int(folds_per_ds.min())
        max_f = int(folds_per_ds.max())
        n_folds_global = df["fold"].nunique()
        print(f"Unique folds total : {n_folds_global}")
        print(f"Folds/dataset      : min={min_f}  max={max_f}")

        if min_folds > 0:
            short = folds_per_ds[folds_per_ds < min_folds]
            if short.empty:
                print(f"  All datasets have >= {min_folds} fold(s)  [OK]")
            else:
                print(f"  Datasets with < {min_folds} fold(s)  ({len(short)}):")
                for ds, n in short.items():
                    print(f"    {ds}: {n} fold(s)")

        # Pivot: rows = datasets, columns = fold indices, values = run count
        counts = (
            df.groupby(["dataset", "fold"])
            .size()
            .rename("runs")
            .reset_index()
            .pivot(index="dataset", columns="fold", values="runs")
            .fillna(0)
            .astype(int)
        )
        counts.columns.name = "fold"
        print("\n  Runs per dataset × fold:")
        print(counts.to_string())

    else:
        counts = df.groupby("dataset").size().rename("runs")
        print("\n  Runs per dataset (no fold column):")
        print(counts.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect ScoringBench parquet result files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "files", nargs="+", type=Path, metavar="FILE",
        help="One or more parquet files to inspect.",
    )
    parser.add_argument(
        "--min-folds", type=int, default=0, metavar="N",
        help="Flag datasets that have fewer than N folds.",
    )
    args = parser.parse_args()

    missing = [f for f in args.files if not f.exists()]
    if missing:
        for f in missing:
            print(f"ERROR: file not found: {f}", file=sys.stderr)
        sys.exit(1)

    for f in args.files:
        _check_single(f, min_folds=args.min_folds)

    print()


if __name__ == "__main__":
    main()
