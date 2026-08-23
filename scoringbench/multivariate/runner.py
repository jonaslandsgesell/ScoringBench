"""Outer benchmark loop for the multivariate ScoringBench.

Iterates over datasets, promotes each to a ``target_dim``-dimensional target
problem, runs K-fold CV, and persists parquet results compatible with
``autorank_leaderboard.py``.

Public API
----------
run_benchmark(datasets_config, model_factories, output_dir, ...) -> DataFrame
"""

import gc
import traceback
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

from . import config as cfg
from .cv import run_fold
from .datasets import load_multivariate_dataset
from .results import build_results_rows, save_fold_parquet


def _empty_cuda_cache() -> None:
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_benchmark(
    datasets_config: list[dict],
    model_factories: dict[str, Callable],
    output_dir: Path,
    *,
    n_folds: int = cfg.N_FOLDS,
    n_repeats_cv: int = cfg.N_REPEATS_CV,
    seed: int = cfg.SEED,
    sample_size: int = cfg.SAMPLE_SIZE,
    target_dim: int = int(cfg.TARGET_DIM),
    load_fn: Callable[..., tuple[pd.DataFrame, pd.DataFrame]] = load_multivariate_dataset,
) -> pd.DataFrame:
    """Iterate over datasets, run CV for each, persist results.

    Parameters mirror the univariate runner; ``target_dim`` is the number of
    target dimensions ``d``.

    ``load_fn`` is the source-specific loader used to turn each ``ds_config``
    into ``(X, Y)``; it must accept ``(ds_config, target_dim=...)`` and may
    raise ``ValueError`` to signal "skip this dataset". It defaults to the
    Source-1 feature-promotion loader so existing callers are unaffected; the
    synthetic source injects its own loader (see ``sources.py``). This is the
    single seam that keeps the runner source-agnostic (open-closed).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []

    total_folds = n_repeats_cv * n_folds
    sep = "=" * 70
    print(sep)
    print(f"ScoringBench-MV  |  {len(datasets_config)} datasets  |  "
          f"{len(model_factories)} models  |  d={target_dim}  |  "
          f"{n_repeats_cv}×{n_folds}-fold CV ({total_folds} folds total)")
    print(sep)

    for ds_config in datasets_config:
        name = ds_config["name"]
        print(f"\n{sep}")
        print(f"Dataset: {name}")
        print(sep)

        try:
            try:
                X, Y = load_fn(ds_config, target_dim=target_dim)
            except ValueError as e:
                print(f"⊘ {name} SKIPPED (loader rejected d={target_dim}): {e}")
                continue
            print(f"Loaded: {len(X)} rows × {X.shape[1]} features → "
                  f"{Y.shape[1]}-dim target")

            effective_sample_size = ds_config.get("sample_size", sample_size) or 0

            cv_results: list[dict] = []
            ds_safe = name.replace(" ", "_")

            for repeat in range(n_repeats_cv):
                repeat_seed = seed + repeat

                kf = KFold(n_splits=n_folds, shuffle=True, random_state=repeat_seed)
                splits = list(kf.split(X))

                train_cap = (
                    int(effective_sample_size * (n_folds - 1) / n_folds)
                    if effective_sample_size else 0
                )
                test_cap = (effective_sample_size // n_folds) if effective_sample_size else 0

                for fold_idx in range(n_folds):
                    global_fold = repeat * n_folds + fold_idx

                    fold_result: dict = {}
                    models_present = []
                    for model_name in model_factories.keys():
                        raw_parquet = output_dir / "raw" / model_name / f"{ds_safe}.parquet"
                        if raw_parquet.exists():
                            try:
                                existing = pd.read_parquet(raw_parquet)
                                matched = existing[existing["fold"] == global_fold]
                                if not matched.empty:
                                    row = matched.iloc[0].to_dict()
                                    for k in ("dataset", "model", "fold"):
                                        row.pop(k, None)
                                    err = row.get("error", None)
                                    if err is not None and not (
                                        isinstance(err, float) and pd.isna(err)
                                    ):
                                        print(f"  Re-running {model_name} for global "
                                              f"fold #{global_fold} (previous error: {err})")
                                        continue
                                    fold_result[model_name] = row
                                    models_present.append(model_name)
                            except Exception:
                                pass

                    models_to_run = {
                        k: v for k, v in model_factories.items()
                        if k not in models_present
                    }

                    fold_label = (f"repeat {repeat + 1}/{n_repeats_cv}, "
                                  f"fold {fold_idx + 1}/{n_folds} (global #{global_fold})")
                    if models_present and not models_to_run:
                        print(f"  Skipping {fold_label} (all models present)")
                        fold_result["fold"] = global_fold
                        cv_results.append(fold_result)
                        continue
                    elif models_present:
                        print(f"  {fold_label}: skipping {', '.join(sorted(models_present))}; "
                              f"running {', '.join(sorted(models_to_run.keys()))}")

                    train_idx, test_idx = splits[fold_idx]

                    if train_cap and len(train_idx) > train_cap:
                        rng_fold = np.random.default_rng(repeat_seed * 10007 + fold_idx)
                        train_idx = rng_fold.choice(train_idx, size=train_cap, replace=False)

                    if test_cap and len(test_idx) > test_cap:
                        rng_test = np.random.default_rng(repeat_seed * 10007 + fold_idx + 1)
                        test_idx = rng_test.choice(test_idx, size=test_cap, replace=False)

                    print(f"\n  {fold_label}  "
                          f"[{len(train_idx)} train / {len(test_idx)} test]", flush=True)
                    new_fold_data = run_fold(
                        X.iloc[train_idx], X.iloc[test_idx],
                        Y.iloc[train_idx], Y.iloc[test_idx],
                        models_to_run, seed,
                    )

                    for k, v in new_fold_data.items():
                        fold_result[k] = v

                    fold_result["fold"] = global_fold
                    save_fold_parquet(new_fold_data, output_dir, name, global_fold)
                    cv_results.append(fold_result)

                    gc.collect()
                    _empty_cuda_cache()

            cv_results.sort(key=lambda d: d["fold"])

            rows = build_results_rows(ds_config, X, Y, cv_results)
            all_rows.extend(rows)

            print(f"\n✓ {name} done")

        except Exception:
            print(f"\n✗ {name} FAILED")
            traceback.print_exc()

        finally:
            gc.collect()
            _empty_cuda_cache()

    final_df = pd.DataFrame(all_rows)
    if not final_df.empty:
        print(f"\n{sep}")
        print(f"Benchmark complete. Results in: {output_dir}")
        print(sep)

    return final_df
