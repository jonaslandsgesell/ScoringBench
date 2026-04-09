#!/bin/bash
# Run all datasets in parallel via a SLURM array job.
# Each task handles exactly one dataset (0-indexed).
#
# Usage — parallel (one job per dataset, 0..103):
#   sbatch --array=0-103 run_benchmark.sbatch
#
# Usage — single dataset (e.g. dataset 7 only):
#   sbatch --array=7 run_benchmark.sbatch
#
# Usage — sequential (all datasets in one job, original behaviour):
#   sbatch run_benchmark.sbatch
#
#SBATCH --partition=gpu
#SBATCH --gres=gpu:RTX-A6000:1
#SBATCH --mem=64G
#SBATCH --time=120:00:00
#SBATCH --job-name=scoringbench

if [ -n "${SLURM_ARRAY_TASK_ID}" ]; then
    # Array mode: run only the dataset at index SLURM_ARRAY_TASK_ID
    python run_bench_regression.py --dataset_index "${SLURM_ARRAY_TASK_ID}"
else
    # Sequential mode: run all datasets in one job (original behaviour)
    python run_bench_regression.py
fi