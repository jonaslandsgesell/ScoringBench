#!/bin/bash
# Promote univariate ScoringBench datasets to multivariate, one per array task.
# Each task screens one dataset (feature promotion + KCI) and writes a part file
# under results/parts/; merge afterwards into promoted_scoringbench_datasets.json.
#
# There are 100 univariate datasets (datasets.json), so:
# Submit from the repository root (paths below are relative to it):
#   sbatch --array=0-99 scripts/promote_targets/run_promote_targets.sbatch
# then merge:
#   python scripts/promote_targets/promote.py --merge \
#       --output_dir scripts/promote_targets/results
#
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --job-name=promote_targets
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

OUTPUT_DIR="scripts/promote_targets/results"
# Cap BLAS/OpenMP threads at the allocation; otherwise numpy/scikit-learn size
# their thread pools to every core on the shared node.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
mkdir -p slurm_logs

# Extra arguments after the script name are passed through, e.g.
#   sbatch --array=0-99 scripts/promote_targets/run_promote_targets.sbatch --refine 0 \
#       --output_dir scripts/promote_targets/results_norefine
python scripts/promote_targets/promote.py \
    --dataset_index "${SLURM_ARRAY_TASK_ID}" \
    --output_dir "${OUTPUT_DIR}" "$@"
