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
# GPUs --gres=gpu:RTX-A6000:1  --gres=gpu:RTX-PRO-6000:1 --gres=gpu:H200:1 (smaller to larger)
#SBATCH --partition=gpu
#SBATCH --gres=gpu:RTX-A6000:1

#SBATCH --mem=128G
#SBATCH --time=120:00:00
#SBATCH --job-name=scoringbench
# Log naming:
#   * Array mode:  slurm_logs/scoringbench-<arrayJobId>_<taskId>.{out,err}
#                  (matches the "22606392_34" shown by squeue).
#   * Single mode: %A resolves to the job id and %a to 4294967294, so a plain
#                  non-array run lands in scoringbench-<jobId>_4294967294.*.
#                  Prefer `sbatch --array=<n> …` for readable names.
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

# Hardcoded sample size
SAMPLE_SIZE="3000" #"3000" is original 

if [ -n "${SLURM_ARRAY_TASK_ID}" ]; then
    # Array mode: run only the dataset at index SLURM_ARRAY_TASK_ID
    python run_bench_regression.py --dataset_index "${SLURM_ARRAY_TASK_ID}" \
        --sample_size "${SAMPLE_SIZE}" \
        --output_dir "./output_${SAMPLE_SIZE}"
else
    # Sequential mode: run all datasets in one job (original behaviour)
    python run_bench_regression.py \
        --sample_size "${SAMPLE_SIZE}" \
        --output_dir "./output_${SAMPLE_SIZE}"
fi