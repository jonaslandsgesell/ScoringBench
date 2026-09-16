#module load Python/3.11.1-GCCcore-10.3.0
#srun --partition=gpu --gres=gpu:RTX-AN_EPOCHS0:1 --pty bash
"""
run_bench_regression.py — ScoringBench front script.

The benchmark line-up lives in ``scoringbench/univariate/models.py`` — edit the
``MODELS`` dict there to add models.  Everything else is automatic.

Usage
-----
    python run_bench_regression.py           # 5-fold CV, all datasets
    python run_bench_regression.py --lite    # 2-fold CV (fast smoke test)
    python run_bench_regression.py --output_dir my_results/
    python run_bench_regression.py --seed 0 --sample_size 1000
"""

import sys
from pathlib import Path

#os.environ['HF_TOKEN'] = 'hf_hash'  # set your Hugging Face token here for tabpfn, or have it in your environment variables for automatic pick-up by the wrapper

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import argparse

from scoringbench.univariate import config as cfg
from scoringbench.univariate.datasets import get_DATASETS_CONFIG, validate_datasets
from scoringbench.univariate.runner import run_benchmark
from scoringbench.univariate.utils import set_seed

# 👉 To add / remove / swap models, edit the MODELS dict in:
#        scoringbench/univariate/models.py
from scoringbench.univariate.models import MODELS


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="ScoringBench regression benchmark")
    p.add_argument(
        "--lite", action="store_true",
        help="2-fold CV — quick smoke test over all datasets",
    )
    p.add_argument(
        "--output_dir", default=None,
        help="Directory for results (default: ./output/ with per-model subfolders)",
    )
    p.add_argument("--seed",          type=int, default=cfg.SEED)
    p.add_argument("--sample_size",   type=int, default=cfg.SAMPLE_SIZE)
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

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Use a single shared output directory that contains per-model subfolders
        output_dir = project_root / "output"
    n_folds = 2 if args.lite else cfg.N_FOLDS
    n_folds = 2 if args.lite else cfg.N_FOLDS

    if output_dir.exists():
        print(f"Resuming into existing output directory: {output_dir}")
        print("Completed (dataset, fold) pairs will be skipped.")

    # === LAZY LOAD & VALIDATE DATASETS ONLY WHEN BENCHMARK RUNS ===
    print("Loading and validating datasets...")
    all_datasets = get_DATASETS_CONFIG()
    validated_datasets = validate_datasets(all_datasets)
    
    if args.dataset_index is not None:
        if args.dataset_index < 0 or args.dataset_index >= len(validated_datasets):
            print(f"Error: --dataset_index {args.dataset_index} is out of range "
                  f"(0..{len(validated_datasets) - 1} for {len(validated_datasets)} datasets).")
            sys.exit(1)
        datasets_to_run = [validated_datasets[args.dataset_index]]
        print(f"Running single dataset #{args.dataset_index}: "
              f"{datasets_to_run[0]['name']}")
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
    )
