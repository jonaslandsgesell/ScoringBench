#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:RTX-A6000:1
#SBATCH --mem=64G
#SBATCH --time=120:00:00

python run_bench_regression.py

#run with "sbatch run_benchmark.sbatch" (from within scoringBench folder)