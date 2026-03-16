"""Outer benchmark loop — iterates over datasets and orchestrates CV.

Public API
----------
run_benchmark(datasets_config, model_factories, output_dir, n_folds, seed, sample_size)
    Main entry-point called from the CLI script.
    Returns the accumulated detailed results DataFrame.
"""

import json
import traceback
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from . import config as cfg
from .datasets import load_dataset
from .cv import run_cv, run_fold
from .results import (
    build_results_rows,
    save_detailed_csv,
    save_aggregated_csv,
    save_fold_json,
)


def run_benchmark(
    datasets_config: list[dict],
    model_factories: dict[str, Callable],
    output_dir: Path,
    *,
    n_folds: int = cfg.N_FOLDS,
    seed: int = cfg.SEED,
    sample_size: int = cfg.SAMPLE_SIZE,
) -> pd.DataFrame:
    """Iterate over datasets, run CV for each, persist results.

    Parameters
    ----------
    datasets_config  : list of dataset config dicts (see datasets.py)
    model_factories  : {name: callable}  (see models.py)
    output_dir       : root directory for all output files
    n_folds          : number of CV folds
    seed             : global random seed
    sample_size      : global row cap (overridden per dataset by 'sample_size' key)

    Returns
    -------
    DataFrame with one row per (dataset, model, fold).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []

    sep = "=" * 70
    print(sep)
    print(f"ScoringBench  |  {len(datasets_config)} datasets  |  "
          f"{len(model_factories)} models  |  {n_folds}-fold CV")
    print(sep)

    for ds_config in datasets_config:
        name = ds_config["name"]
        print(f"\n{sep}")
        print(f"Dataset: {name}")
        print(sep)

        try:
            X, y = load_dataset(ds_config)
            print(f"Loaded: {len(X)} rows × {X.shape[1]} features")

            # Global sample cap (per-dataset cap is already applied in load_dataset)
            if len(X) > sample_size:
                idx = np.random.choice(len(X), size=sample_size, replace=False)
                X = X.iloc[idx].reset_index(drop=True)
                y = y.iloc[idx].reset_index(drop=True)
                print(f"Capped to {sample_size} rows")

            # For the new per-model layout we determine, for each fold, which
            # models still need to be run. Existing per-model per-fold JSONs are
            # merged into `cv_results` so downstream aggregation sees a full set
            # of models per fold.
            cv_results: list[dict] = []

            # Pre-compute KFold splits once
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            splits = list(kf.split(X))

            for fold_idx in range(n_folds):
                # Start this fold result with any existing per-model parquet rows
                fold_result: dict = {}
                # Discover which models already have this fold by reading
                # per-model parquet files (one parquet per model).
                models_present = []
                ds_safe = name.replace(" ", "_")
                for model_name in model_factories.keys():
                    dest_parquet = output_dir / f"{model_name}.parquet"
                    if dest_parquet.exists():
                        try:
                            existing = pd.read_parquet(dest_parquet)
                            mask = (
                                (existing.get("dataset") == ds_safe) &
                                (existing.get("fold") == fold_idx)
                            )
                            matched = existing[mask]
                            if not matched.empty:
                                # Use the first matching row (should be unique)
                                row = matched.iloc[0].to_dict()
                                # Remove bookkeeping fields to leave metrics only
                                for k in ("dataset", "model", "fold"):
                                    row.pop(k, None)
                                fold_result[model_name] = row
                                models_present.append(model_name)
                        except Exception:
                            # If reading parquet fails, treat as missing and continue
                            pass

                # Determine which models still need running for this fold
                models_to_run = {k: v for k, v in model_factories.items() if k not in models_present}

                # Logging which models are skipped / will be run
                if models_present and not models_to_run:
                    print(f"  Skipping fold {fold_idx + 1}/{n_folds} (all models present): {', '.join(sorted(models_present))}")
                    # Ensure fold has a fold index entry
                    fold_result["fold"] = fold_idx
                    cv_results.append(fold_result)
                    continue
                elif models_present:
                    print(f"  Fold {fold_idx + 1}/{n_folds}: skipping models: {', '.join(sorted(models_present))}; running: {', '.join(sorted(models_to_run.keys()))}")

                train_idx, test_idx = splits[fold_idx]
                print(f"\n  Fold {fold_idx + 1}/{n_folds}", flush=True)
                new_fold_data = run_fold(
                    X.iloc[train_idx], X.iloc[test_idx],
                    y.iloc[train_idx], y.iloc[test_idx],
                    models_to_run, seed,
                )

                # `new_fold_data` contains only the newly-run models; merge with
                # any existing model results discovered earlier.
                for k, v in new_fold_data.items():
                    fold_result[k] = v

                fold_result["fold"] = fold_idx
                # Persist only newly-run per-model results (don't touch existing parquet)
                save_fold_json(new_fold_data, output_dir, name, fold_idx)
                cv_results.append(fold_result)

            cv_results.sort(key=lambda d: d["fold"])

            # Flatten to rows and accumulate
            rows = build_results_rows(ds_config, X, cv_results)
            all_rows.extend(rows)

            # Write / update CSVs after every dataset so partial runs are useful
            detailed_df = save_detailed_csv(rows, output_dir)
            save_aggregated_csv(
                pd.read_csv(output_dir / "benchmark_results_detailed.csv"),
                output_dir,
            )

            print(f"\n✓ {name} done")

        except Exception:
            print(f"\n✗ {name} FAILED")
            traceback.print_exc()

    # Final consolidated save
    final_df = pd.DataFrame(all_rows)
    if not final_df.empty:
        save_aggregated_csv(
            pd.read_csv(output_dir / "benchmark_results_detailed.csv"),
            output_dir,
        )
        print(f"\n{sep}")
        print(f"Benchmark complete. Results in: {output_dir}")
        print(sep)

    return final_df
