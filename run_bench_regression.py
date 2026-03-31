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
from scoringbench.datasets import DATASETS_CONFIG
from scoringbench.runner import run_benchmark
from scoringbench.utils import set_seed
from scoringbench.wrappers import TabPFNWrapper, FinetuneTabPFNWrapper, TabICLWrapper, XGBVectorWrapper, XGBQuantileVectorWrapper


# ---------------------------------------------------------------------------
# Models — edit here to add / replace / wrap models
# Each value is a zero-arg factory that returns a fresh, unfitted wrapper.
# ---------------------------------------------------------------------------

#_f728b95a_seed42_folds5_size3000 is with different datasets (hand picked)
# models_1ab52f4f_seed42_folds5_size3000 is run with handpicked datasets including mae early stopping tabpfn

# 4d40c9... is with 3 benchmark suites (297, 299, 269) and scoring rules
# models_3d841de1_seed42_folds5_size3000is run with 3 benchmark suites (297, 299, 269) including mae early stopping tabpfn
N_EPOCHS=80
MODELS = {
    "tabpfn": lambda: TabPFNWrapper(),
    "tabpfnv2_6": lambda: TabPFNWrapper(model_path="tabpfn-v2.6-regressor-v2.6_default.ckpt"),
    "finetune_realtabpfnv2_5_crls": lambda: FinetuneTabPFNWrapper(
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
        beta="crls",
    ),
    "finetune_realtabpfnv2_5_crps": lambda: FinetuneTabPFNWrapper(
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
        beta="crps",
    ),
    # "finetune_tabpfn_wcrps_left": lambda: FinetuneTabPFNWrapper(
    #     device="cuda",
    #     epochs=N_EPOCHS,
    #     learning_rate=1e-5,
    #     weight_decay=0.1,
    #     crps_loss_weight=1.0,
    #     mse_loss_weight=0.0,
    #     ce_loss_weight=0.0,
    #     n_finetune_ctx_plus_query_samples=20_000,
    #     n_estimators_finetune=1,
    #     n_estimators_validation=8,
    #     n_estimators_final_inference=8,
    #     early_stopping=True,
    #     early_stopping_patience=20,
    #     finetune_ctx_query_split_ratio=0.4,
    #     extra_regressor_kwargs={"average_before_softmax": True},
    #     beta="wCRPS_left",
    # ),
    "finetune_realtabpfnv2_5_wcrps_center": lambda: FinetuneTabPFNWrapper(
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
        beta="wCRPS_center",
    ),
    # "finetune_tabpfn_wcrps_right": lambda: FinetuneTabPFNWrapper(
    #     device="cuda",
    #     epochs=N_EPOCHS,
    #     learning_rate=1e-5,
    #     weight_decay=0.1,
    #     crps_loss_weight=1.0,
    #     mse_loss_weight=0.0,
    #     ce_loss_weight=0.0,
    #     n_finetune_ctx_plus_query_samples=20_000,
    #     n_estimators_finetune=1,
    #     n_estimators_validation=8,
    #     n_estimators_final_inference=8,
    #     early_stopping=True,
    #     early_stopping_patience=20,
    #     finetune_ctx_query_split_ratio=0.4,
    #     extra_regressor_kwargs={"average_before_softmax": True},
    #     beta="wCRPS_right",
    # ),
    # "finetune_tabpfn_brier": lambda: FinetuneTabPFNWrapper(
    #     device="cuda",
    #     epochs=N_EPOCHS,
    #     learning_rate=1e-5,
    #     weight_decay=0.1,
    #     crps_loss_weight=1.0,
    #     mse_loss_weight=0.0,
    #     ce_loss_weight=0.0,
    #     n_finetune_ctx_plus_query_samples=20_000,
    #     n_estimators_finetune=1,
    #     n_estimators_validation=8,
    #     n_estimators_final_inference=8,
    #     early_stopping=True,
    #     early_stopping_patience=20,
    #     finetune_ctx_query_split_ratio=0.4,
    #     extra_regressor_kwargs={"average_before_softmax": True},
    #     beta="brier",
    # ),
    "finetune_realtabpfnv2_5_ce": lambda: FinetuneTabPFNWrapper(
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
        beta="ce",
    ),
    "finetune_realtabpfnv2_5_beta_0.5": lambda: FinetuneTabPFNWrapper(
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
        beta="beta_0.5",
    ),
    "finetune_realtabpfnv2_5_beta_1.5": lambda: FinetuneTabPFNWrapper(
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
        beta="beta_1.5",
    ),
    "finetune_realtabpfnv2_5_beta_1.8": lambda: FinetuneTabPFNWrapper(
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
        beta="beta_1.8",
    ),
    "finetune_realtabpfnv2_5_beta_1.8_mae": lambda: FinetuneTabPFNWrapper(
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
        beta="beta_1.8",
        early_stopping_metric="mae",
    ),
    "finetune_realtabpfnv2_5_is_90": lambda: FinetuneTabPFNWrapper(
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
        beta="is_90",
    ),
    "tabicl": lambda: TabICLWrapper(),
    "xgb_vector": lambda: XGBVectorWrapper(n_bins=50, num_boost_round=100),
    "xgb_vector_quantile": lambda: XGBQuantileVectorWrapper(n_bins=50, num_boost_round=100),
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
    p.add_argument("--seed",        type=int, default=cfg.SEED)
    p.add_argument("--sample_size", type=int, default=cfg.SAMPLE_SIZE)
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

    run_benchmark(
        datasets_config=DATASETS_CONFIG,
        model_factories=MODELS,
        output_dir=output_dir,
        n_folds=n_folds,
        seed=args.seed,
        sample_size=args.sample_size,
    )
