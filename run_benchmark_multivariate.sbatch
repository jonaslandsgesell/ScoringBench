#!/bin/bash
# Run all datasets in parallel via a SLURM array job (MULTIVARIATE benchmark).
# Each task handles exactly one dataset (0-indexed).
#
# This wraps run_bench_regression_multivariate.py, which promotes the (d-1)
# features most correlated with the target into additional targets to build a
# d-dimensional multivariate regression task.
#
# Usage — parallel (one job per dataset, 0..103):
#   sbatch --array=0-103 run_benchmark_multivariate.sbatch
#
# Usage — single dataset (e.g. dataset 7 only):
#   sbatch --array=7 run_benchmark_multivariate.sbatch
#
# Usage — sequential (all datasets in one job):
#   sbatch run_benchmark_multivariate.sbatch
#
# GPUs --gres=gpu:RTX-A6000:1  --gres=gpu:RTX-PRO-6000:1 --gres=gpu:H200:1 (smaller to larger)
##SBATCH --partition=gpu
##SBATCH --gres=gpu:RTX-A6000:1

#SBATCH --mem=128G
#SBATCH --time=120:00:00
#SBATCH --job-name=scoringbench_mv
# Log naming:
#   * Array mode:  slurm_logs/scoringbench_mv-<arrayJobId>_<taskId>.{out,err}
#                  (matches the "22606392_34" shown by squeue).
#   * Single mode: %A resolves to the job id and %a to 4294967294, so a plain
#                  non-array run lands in scoringbench_mv-<jobId>_4294967294.*.
#                  Prefer `sbatch --array=<n> …` for readable names.
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

# Hardcoded run configuration
SAMPLE_SIZE="1000"        # "3000" is original
TARGET_DIM="3"            # target dimension d (promotes d-1 correlated features)
SOURCE="synthetic" #"scoringbench"     # dataset source: "scoringbench" (real) or "synthetic"

# Output dir mirrors the runner's default naming so sweeps never collide.
OUTPUT_DIR="./output_multivariate_${SOURCE}_d${TARGET_DIM}_n${SAMPLE_SIZE}"

if [ -n "${SLURM_ARRAY_TASK_ID}" ]; then
    # Array mode: run only the dataset at index SLURM_ARRAY_TASK_ID
    python run_bench_regression_multivariate.py --dataset_index "${SLURM_ARRAY_TASK_ID}" \
        --source "${SOURCE}" \
        --sample_size "${SAMPLE_SIZE}" \
        --target_dim "${TARGET_DIM}" \
        --output_dir "${OUTPUT_DIR}"
else
    # Sequential mode: run all datasets in one job
    python run_bench_regression_multivariate.py \
        --source "${SOURCE}" \
        --sample_size "${SAMPLE_SIZE}" \
        --target_dim "${TARGET_DIM}" \
        --output_dir "${OUTPUT_DIR}"
fi
