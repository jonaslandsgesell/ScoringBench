## Why this benchmark?
Proper scoring rules have long been used to rigorously evaluate probabilistic forecasts, but their application has been largely confined to classification tasks. ScoringBench brings proper scoring rule based evaluation to **probabilistic regression** — an inherently continuous setting where models must predict full predictive distributions over real-valued targets.

This matters because modern tabular foundation models (e.g., TabPFN, TabICL) natively output full probability distributions, not just point estimates. This means that practically useful quantities such as **prediction intervals, quantile estimates, and uncertainty bounds are readily extracted from those base models** — but existing benchmarks have no way to measure how well those distributional outputs are calibrated or sharp.

ScoringBench was created to:
- Bring proper scoring rules (CRPS, CRLS, Interval Score, Beta-Energy Scores) to regression benchmarking, not just classification.
- Enable fair comparison of probabilistic regression models on the full predictive distribution.
- Highlight the value of distributional outputs for real-world decision making, where prediction intervals are often more actionable than point estimates.
- Support research and development of models that output full predictive distributions, not just point estimates.

For more details on the motivation and methodology, see the accompanying publications by the authors https://arxiv.org/abs/2603.29928 and https://arxiv.org/abs/2603.08206.

# ScoringBench

ScoringBench is a compact benchmarking suite for probabilistic regression on tabular data. It evaluates full predictive distributions using proper scoring rules (CRPS, CRLS, Interval Score, Beta-Energy, etc.). The codebase is lightweight and intended to be easy to run and extend.

## Quick overview — important scripts

- `run_bench_regression.py`: run the benchmark (all datasets, models, CV folds). Use `--lite` for a fast smoke test and `--output_dir` to change the output path.
- `autorank_leaderboard.py`: compute statistical rankings with critical-difference diagrams; generates JSON data and LaTeX tables in `output/figures/leaderboard/`.
- `plot_output.py`: generate summary and per-dataset tables/plots from benchmark outputs. Defaults are reasonable; use `--relative`, `--median`, or `--output` to customize.

## Related tools

- [autorank](https://sherbold.github.io/autorank/) — statistical ranking and critical-difference diagrams

## Benchmark output (summary)

Each run writes a self-contained directory under `output/` with aggregated CSVs and per-fold results. Newer runs save per-model Parquet files (`<run>/model_name.parquet`) containing fold-level rows.

Typical files:

- `benchmark_results_aggregated.csv` — mean ± std per model/dataset
- `benchmark_results_detailed.csv` — fold-level rows
- `output/` — per-model parquet

## Workflow

1. git clone --recurse-submodules https://github.com/jonaslandsgesell/ScoringBench.git
2. Add your custom wrapper with a unique name (see `scoringbench/wrappers/` and inherit `ProbabilisticWrapper`).
3. python run_bench_regression.py
4. python autorank_leaderboard.py
5. Commit your model parquet file (documenting each run) and the updated leaderboard data in `output/figures/leaderboard/`. Since the output repository is separate from the main repository, push to both. This serves as a public ledger and allows traceability.
6. Create a pull request to the ScoringBench repository for review; contributions that meet standards will be merged.
7. Upon merge, https://scoringbench.bolt.host/ will automatically display the updated leaderboard; the data is also available in the repository.

## Tests

Run the test suite with:

```
python -m pytest tests
```
