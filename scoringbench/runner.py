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
    n_repeats_cv: int = cfg.N_REPEATS_CV,
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
    n_repeats_cv     : number of CV repeats; each repeat draws a fresh
                       subsample of size *sample_size* from the full dataset
                       using seed+repeat — giving diverse folds across repeats
                       while all models in one (repeat, fold) see identical data.
    seed             : global random seed
    sample_size      : max rows fed into KFold per repeat (train+test combined);
                       0 or None means no cap.  Overridden per dataset by
                       the 'sample_size' key in the dataset config.

    Returns
    -------
    DataFrame with one row per (dataset, model, fold).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []

    total_folds = n_repeats_cv * n_folds
    sep = "=" * 70
    print(sep)
    print(f"ScoringBench  |  {len(datasets_config)} datasets  |  "
          f"{len(model_factories)} models  |  "
          f"{n_repeats_cv}×{n_folds}-fold CV ({total_folds} folds total)")
    print(sep)

    for ds_config in datasets_config:
        name = ds_config["name"]
        print(f"\n{sep}")
        print(f"Dataset: {name}")
        print(sep)

        try:
            X, y = load_dataset(ds_config)
            print(f"Loaded: {len(X)} rows × {X.shape[1]} features")

            # Effective sample size: per-dataset override or global
            effective_sample_size = ds_config.get('sample_size', sample_size) or 0

            # For the new per-model layout we determine, for each fold, which
            # models still need to be run. Existing per-model per-fold JSONs are
            # merged into `cv_results` so downstream aggregation sees a full set
            # of models per fold.
            cv_results: list[dict] = []
            ds_safe = name.replace(" ", "_")

            for repeat in range(n_repeats_cv):
                repeat_seed = seed + repeat

                # KFold on the FULL dataset — test indices cover all of X across folds
                kf = KFold(n_splits=n_folds, shuffle=True, random_state=repeat_seed)
                splits = list(kf.split(X))

                # Cap train and test per fold so total ≈ sample_size
                train_cap = (
                    int(effective_sample_size * (n_folds - 1) / n_folds)
                    if effective_sample_size else 0
                )
                test_cap = (effective_sample_size // n_folds) if effective_sample_size else 0

                for fold_idx in range(n_folds):
                    # Global fold key encodes both repeat and fold
                    global_fold = repeat * n_folds + fold_idx

                    # Start this fold result with any existing per-model parquet rows
                    fold_result: dict = {}
                    models_present = []
                    for model_name in model_factories.keys():
                        dest_parquet = output_dir / f"{model_name}.parquet"
                        if dest_parquet.exists():
                            try:
                                existing = pd.read_parquet(dest_parquet)
                                mask = (
                                    (existing.get("dataset") == ds_safe) &
                                    (existing.get("fold") == global_fold)
                                )
                                matched = existing[mask]
                                if not matched.empty:
                                    row = matched.iloc[0].to_dict()
                                    for k in ("dataset", "model", "fold"):
                                        row.pop(k, None)
                                    fold_result[model_name] = row
                                    models_present.append(model_name)
                            except Exception:
                                pass

                    models_to_run = {
                        k: v for k, v in model_factories.items()
                        if k not in models_present
                    }

                    fold_label = f"repeat {repeat + 1}/{n_repeats_cv}, fold {fold_idx + 1}/{n_folds} (global #{global_fold})"
                    if models_present and not models_to_run:
                        print(f"  Skipping {fold_label} (all models present)")
                        fold_result["fold"] = global_fold
                        cv_results.append(fold_result)
                        continue
                    elif models_present:
                        print(f"  {fold_label}: skipping {', '.join(sorted(models_present))}; "
                              f"running {', '.join(sorted(models_to_run.keys()))}")

                    train_idx, test_idx = splits[fold_idx]

                    # Subsample training set per (repeat, fold) for diversity —
                    # each gets a fresh random subset of the full training split.
                    if train_cap and len(train_idx) > train_cap:
                        rng_fold = np.random.default_rng(repeat_seed * 10007 + fold_idx)
                        train_idx = rng_fold.choice(train_idx, size=train_cap, replace=False)

                    # Cap test set to keep evaluation cost bounded
                    if test_cap and len(test_idx) > test_cap:
                        rng_test = np.random.default_rng(repeat_seed * 10007 + fold_idx + 1)
                        test_idx = rng_test.choice(test_idx, size=test_cap, replace=False)

                    print(f"\n  {fold_label}  "
                          f"[{len(train_idx)} train / {len(test_idx)} test]", flush=True)
                    new_fold_data = run_fold(
                        X.iloc[train_idx], X.iloc[test_idx],
                        y.iloc[train_idx], y.iloc[test_idx],
                        models_to_run, seed,
                    )

                    for k, v in new_fold_data.items():
                        fold_result[k] = v

                    fold_result["fold"] = global_fold
                    save_fold_json(new_fold_data, output_dir, name, global_fold)
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
