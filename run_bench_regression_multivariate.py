#module load Python/3.11.1-GCCcore-10.3.0
"""
run_bench_regression_multivariate.py — ScoringBench multivariate front script.

Builds d-dimensional targets from standard 1-D regression datasets by promoting
the (d-1) features most correlated with the target into targets (Source 1), then
benchmarks purely sample-based multivariate models with the energy score,
variogram score and Dawid-Sebastiani score. Output is 1:1 compatible with
``autorank_leaderboard.py``.

Edit ``scoringbench.multivariate.models.MODELS`` to add / swap models.

Usage
-----
    python run_bench_regression_multivariate.py               # 5-fold CV, all datasets
    python run_bench_regression_multivariate.py --lite        # 2-fold CV (fast smoke test)
    python run_bench_regression_multivariate.py --target_dim 3 --sample_size 3000
    python run_bench_regression_multivariate.py --output_dir my_results/
    python run_bench_regression_multivariate.py --dataset_index 0   # single dataset (SLURM arrays)

The default output directory is ``./output_multivariate_d{d}_n{sample_size}/`` so
different d / sample-size sweeps never overwrite each other.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import argparse

from scoringbench.multivariate import config as cfg
from scoringbench.multivariate.models import MODELS
from scoringbench.multivariate.runner import run_benchmark
from scoringbench.multivariate.sources import SOURCES, get_source
from scoringbench.multivariate.utils import set_seed


def parse_args():
    p = argparse.ArgumentParser(description="ScoringBench multivariate regression benchmark")
    p.add_argument(
        "--lite", action="store_true",
        help="2-fold CV — quick smoke test over all datasets",
    )
    p.add_argument(
        "--source", default="scoringbench", choices=sorted(SOURCES),
        help="Dataset source: 'scoringbench' (feature promotion from real "
             "regression datasets) or 'synthetic' (explicit copula-coupled "
             "dependent targets). Default: scoringbench.",
    )
    p.add_argument(
        "--output_dir", default=None,
        help="Directory for results "
             "(default: ./output_multivariate_{source}_d{d}_n{sample_size}/)",
    )
    p.add_argument("--seed",          type=int, default=cfg.SEED)
    p.add_argument("--sample_size",   type=int, default=cfg.SAMPLE_SIZE)
    p.add_argument("--target_dim",    type=int, default=cfg.TARGET_DIM,
                   help="Target dimension d (promotes d-1 correlated features to targets)")
    p.add_argument("--n_repeats_cv",  type=int, default=cfg.N_REPEATS_CV,
                   help="Number of repeated CV rounds (each uses a fresh resample)")
    p.add_argument(
        "--dataset_index", type=int, default=None,
        help="0-based index into DATASETS_CONFIG. If set, only that one dataset "
             "is benchmarked (for SLURM array jobs). If omitted, all datasets run.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    source = get_source(args.source)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Encode source, d and sample size so sweeps never collide.
        output_dir = project_root / (
            f"output_multivariate_{args.source}_d{args.target_dim}_n{args.sample_size}"
        )

    n_folds = 2 if args.lite else cfg.N_FOLDS

    if output_dir.exists():
        print(f"Resuming into existing output directory: {output_dir}")
        print("Completed (dataset, fold) pairs will be skipped.")

    print(f"Enumerating datasets for source '{args.source}'...")
    validated_datasets = source.enumerate_datasets(args.target_dim, args.sample_size)

    if args.dataset_index is not None:
        if args.dataset_index < 0 or args.dataset_index >= len(validated_datasets):
            print(f"Error: --dataset_index {args.dataset_index} is out of range "
                  f"(0..{len(validated_datasets) - 1} for {len(validated_datasets)} datasets).")
            sys.exit(1)
        datasets_to_run = [validated_datasets[args.dataset_index]]
        print(f"Running single dataset #{args.dataset_index}: {datasets_to_run[0]['name']}")
    else:
        datasets_to_run = validated_datasets

    run_benchmark(
        datasets_config=datasets_to_run,
        model_factories=MODELS,
        output_dir=output_dir,
        n_folds=n_folds,
        n_repeats_cv=args.n_repeats_cv,
        seed=args.seed,
        sample_size=args.sample_size,
        target_dim=args.target_dim,
        load_fn=source.load,
    )
