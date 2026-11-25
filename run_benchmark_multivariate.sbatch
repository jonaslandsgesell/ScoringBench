#!/bin/bash
# Run the MULTIVARIATE benchmark as a SLURM array job, one dataset per task.
# Set SOURCE and TARGET_DIM below; the array range is 0..N-1 with N from the table.
#
# Usage — parallel (one task per dataset), e.g. promoted_scoringbench at d=2:
#   sbatch --array=0-44 run_benchmark_multivariate.sbatch
#
# Usage — single dataset (e.g. dataset 7 only):
#   sbatch --array=7 run_benchmark_multivariate.sbatch
#
# Usage — sequential (all datasets in one job):
#   sbatch run_benchmark_multivariate.sbatch
#
# GPUs --gres=gpu:RTX-A6000:1  --gres=gpu:RTX-PRO-6000:1 --gres=gpu:H200:1 (smaller to larger)
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

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

# Run configuration
SAMPLE_SIZE="3000"                # "3000" is original
TARGET_DIM="2"                    # target dimension d: "2" | "3" | "5" | "10"
SOURCE="promoted_scoringbench"    # dataset source, datasets at d=2/3/5/10:
                                  #   "promoted_scoringbench"  45/36/22/6  univariate ScoringBench with promoted targets
                                  #   "scoringbench"           51/44/28/7  curated multi-target OpenML (multivariate_datasets.json)
                                  #   "multitarget_openml"     14/6/4/2    KCI-screened multi-target OpenML
                                  #   "synthetic"              100         copula-coupled residuals

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
