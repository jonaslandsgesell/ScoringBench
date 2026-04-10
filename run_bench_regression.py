#module load Python/3.11.1-GCCcore-10.3.0
#srun --partition=gpu --gres=gpu:RTX-AN_EPOCHS0:1 --pty bash
"""
run_bench_regression.py — ScoringBench front script.

Edit MODELS to add / swap models.  Everything else is automatic.

Usage
-----
    python run_bench_regression.py           # 5-fold CV, all datasets
    python run_bench_regression.py --lite    # 2-fold CV (fast smoke test)
    python run_bench_regression.py --output_dir my_results/
    python run_bench_regression.py --seed 0 --sample_size 1000
"""

import os
import sys
import hashlib
from pathlib import Path

#os.environ['HF_TOKEN'] = 'hf_hash'  # set your Hugging Face token here for tabpfn, or have it in your environment variables for automatic pick-up by the wrapper

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import argparse

from scoringbench import config as cfg
from scoringbench.datasets import get_DATASETS_CONFIG, validate_datasets
from scoringbench.runner import run_benchmark
from scoringbench.utils import set_seed
from scoringbench.wrappers import TabPFNWrapper, FinetuneTabPFNWrapper, TabICLWrapper, XGBVectorWrapper, XGBQuantileVectorWrapper, PytabkitRealMLPWrapper, PytabkitRealMLPHPOWrapper


# ---------------------------------------------------------------------------
# TabPFN Version & Model Paths
# ---------------------------------------------------------------------------

TABPFN_VERSION = "realv2_5"

MODEL_PATH_MAP = {
    "realv2_5": "tabpfn-v2.5-regressor-v2.5_real.ckpt",
    "v2_6": "tabpfn-v2.6-regressor-v2.6_default.ckpt",
}


# ---------------------------------------------------------------------------
# Models — edit here to add / replace / wrap models
# Each value is a zero-arg factory that returns a fresh, unfitted wrapper.
# ---------------------------------------------------------------------------

N_EPOCHS=80


def _create_finetune_model_tabpfn(beta_name, tabpfn_version):
    """
    Factory for creating finetune models with specified beta loss.
    
    Args:
        beta_name: Name of the beta loss (e.g., "crps", "crls", "wCRPS_left", "beta_0.5", etc.)
    
    Returns:
        A lambda that creates a FinetuneTabPFNWrapper instance.
    """
    return lambda: FinetuneTabPFNWrapper(
        device="cuda",
        epochs=N_EPOCHS,
        learning_rate=1e-5,
        weight_decay=0.1,
        crps_loss_weight=1.0,
        mse_loss_weight=0.0,
        ce_loss_weight=0.0,
        n_finetune_ctx_plus_query_samples=20_000,
        n_estimators_finetune=1,
        n_estimators_validation=8,
        n_estimators_final_inference=8,
        early_stopping=True,
        early_stopping_patience=20,
        finetune_ctx_query_split_ratio=0.4,
        extra_regressor_kwargs={"average_before_softmax": True},
        beta=beta_name,
        model_path=MODEL_PATH_MAP[tabpfn_version],
    )


# Beta loss configurations for finetune models
# To add more betas, simply append to this list, and the model will be automatically added to MODELS
FINETUNE_BETAS = [
    "crls",
    "crps",
    "wCRPS_left",
    "wCRPS_center",
    "wCRPS_right",
    "ce",
    "beta_0.1",
    "beta_0.3",
    "beta_0.5",
    "beta_0.7",
    "beta_0.9",
    "beta_1.1",
    "beta_1.3",
    "beta_1.5",
    "beta_1.7",
    "beta_1.8",
    "beta_1.9",
    "is_90",
    "cde",
]

dict_finetuned_models = {
    f"finetune_tabpfn_{TABPFN_VERSION}_{beta}": _create_finetune_model_tabpfn(beta, TABPFN_VERSION)
    for beta in FINETUNE_BETAS
}

MODELS = {
    f"tabpfn_realv2_5": lambda: TabPFNWrapper(model_path=MODEL_PATH_MAP["realv2_5"]),
    f"tabpfn_v2_6": lambda: TabPFNWrapper(model_path=MODEL_PATH_MAP["v2_6"]),
    **dict_finetuned_models,
    "tabiclv2": lambda: TabICLWrapper(),
    "xgb_vector": lambda: XGBVectorWrapper(n_bins=50, num_boost_round=100),
    "xgb_vector_quantile": lambda: XGBQuantileVectorWrapper(n_bins=50, num_boost_round=100),
    "pytabkit_realmlp_td": lambda: PytabkitRealMLPWrapper(
        train_metric_name='multi_pinball(0.01,0.03,0.05,0.07,0.09,0.11,0.13,0.15,0.17,0.19,0.21,0.23,0.25,0.27,0.29,0.31,0.33,0.35,0.37,0.39,0.41,0.43,0.45,0.47,0.49,0.51,0.53,0.55,0.57,0.59,0.61,0.63,0.65,0.67,0.69,0.71,0.73,0.75,0.77,0.79,0.81,0.83,0.85,0.87,0.89,0.91,0.93,0.95,0.97,0.99)',
        val_metric_name='multi_pinball(0.01,0.03,0.05,0.07,0.09,0.11,0.13,0.15,0.17,0.19,0.21,0.23,0.25,0.27,0.29,0.31,0.33,0.35,0.37,0.39,0.41,0.43,0.45,0.47,0.49,0.51,0.53,0.55,0.57,0.59,0.61,0.63,0.65,0.67,0.69,0.71,0.73,0.75,0.77,0.79,0.81,0.83,0.85,0.87,0.89,0.91,0.93,0.95,0.97,0.99)',
        n_quantiles=50,
    ),
    # "pytabkit_realmlp_hpo_cv_8_new": lambda: PytabkitRealMLPHPOWrapper(
    #     train_metric_name='multi_pinball(0.01,0.03,0.05,0.07,0.09,0.11,0.13,0.15,0.17,0.19,0.21,0.23,0.25,0.27,0.29,0.31,0.33,0.35,0.37,0.39,0.41,0.43,0.45,0.47,0.49,0.51,0.53,0.55,0.57,0.59,0.61,0.63,0.65,0.67,0.69,0.71,0.73,0.75,0.77,0.79,0.81,0.83,0.85,0.87,0.89,0.91,0.93,0.95,0.97,0.99)',
    #     val_metric_name='multi_pinball(0.01,0.03,0.05,0.07,0.09,0.11,0.13,0.15,0.17,0.19,0.21,0.23,0.25,0.27,0.29,0.31,0.33,0.35,0.37,0.39,0.41,0.43,0.45,0.47,0.49,0.51,0.53,0.55,0.57,0.59,0.61,0.63,0.65,0.67,0.69,0.71,0.73,0.75,0.77,0.79,0.81,0.83,0.85,0.87,0.89,0.91,0.93,0.95,0.97,0.99)',
    #     n_quantiles=50,
    #     n_cv=8,
    #     hpo_space_name='tabarena-new',
    #     use_caruana_ensembling=True,
    # ),
}



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
