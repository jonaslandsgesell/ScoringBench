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

Each run writes per-dataset per-model raw Parquet files to `output/raw/{model_name}/{dataset_name}.parquet`. This structure avoids concurrency issues when running multiple datasets in parallel (SLURM array jobs).

Typical directory structure:

- `output/raw/{model_name}/{dataset_name}.parquet` — raw results organized by model and dataset
- `output/{model_name}.parquet` — aggregated per-model parquet files (after running autorank_leaderboard.py)


## Workflow

1. git clone --recurse-submodules https://github.com/jonaslandsgesell/ScoringBench.git
2. Add your custom wrapper with a unique name (see `scoringbench/wrappers/` and inherit `ProbabilisticWrapper`).
3. python run_bench_regression.py
4. python autorank_leaderboard.py
5. Commit aggregated per-model Parquet files (`output/*.parquet`) and the generated JSON ranking files in `output/figures/leaderboard/` to git LFS. Since the output repository is separate from the main repository, push to both. This serves as a public ledger and allows traceability.
6. Create a pull request to the ScoringBench repository for review; contributions that meet standards will be merged.
7. Upon merge, https://scoringbench.bolt.host/ will automatically display the updated leaderboard; the data is also available in the repository.

## Tests

Run the test suite with:

```
python -m pytest tests
```

## Examples & Diagnostics

### Configuration Comparison Diagnostic

Diagnostic script to evaluate how hyperparameters affect distributional metrics:

```python
import numpy as np
import pandas as pd
import time
from scoringbench.wrappers.tabpfn import TabPFNWrapper
from scoringbench.metrics import compute_metrics

CONFIGS = [
    {"name": "v2.5: param=0.9", "model_path": "tabpfn-v2.5-regressor-v2.5_real.ckpt", "hyperparameter": 0.9},
    {"name": "v2.5: param=1.0", "model_path": "tabpfn-v2.5-regressor-v2.5_real.ckpt", "hyperparameter": 1.0},
    {"name": "v2.6: param=0.9", "model_path": "tabpfn-v2.6-regressor-v2.6_default.ckpt", "hyperparameter": 0.9},
    {"name": "v2.6: param=1.0", "model_path": "tabpfn-v2.6-regressor-v2.6_default.ckpt", "hyperparameter": 1.0},
]

def evaluate_config(X_train, y_train, X_test, y_test, config_dict):
    model = TabPFNWrapper(n_estimators=8, random_state=42, **{k: v for k, v in config_dict.items() if k != "name"})
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    
    y_test_np = np.asarray(y_test, dtype=float)
    dist = model.predict_distribution(X_test)
    metrics = compute_metrics(dist, y_test_np)
    metrics["train_time"] = train_time
    return metrics

# Generate data & evaluate
rng = np.random.default_rng(42)
n_train, n_test, n_features = 100, 200, 2
X = rng.normal(0, 1, (n_train + n_test, n_features))
y = X @ rng.normal(0, 1, n_features) + rng.normal(0, 1, n_train + n_test)

results = [{"config_name": cfg["name"], **evaluate_config(X[:n_train], y[:n_train], X[n_train:], y[n_train:], cfg)} 
           for cfg in CONFIGS]
df = pd.DataFrame(results)
print(df)
```

**Metrics evaluated:** CRPS, log-score, CRLS, sharpness, dispersion, interval scores, beta-energy scores, quantile weighted WCRPS (left, center, right), and others.

### CI/CD Assertions

Add regression tests to your pipeline using ScoringBench metrics:

```python
import numpy as np
from scoringbench.wrappers.tabpfn import TabPFNWrapper
from scoringbench.metrics import compute_metrics

model = TabPFNWrapper(n_estimators=8, random_state=42, model_path="tabpfn-v2.6-regressor-v2.6_default.ckpt")
model.fit(X_train, y_train)

y_test_np = np.asarray(y_test, dtype=float)
dist = model.predict_distribution(X_test)
metrics = compute_metrics(dist, y_test_np)

# Assert on distributional metrics
assert metrics["crps"] < 0.5, f"CRPS {metrics['crps']} exceeds threshold"
assert metrics["log_score"] < 1.0, f"log_score {metrics['log_score']} exceeds threshold"
assert not np.any(np.isnan(dist.mean())), "Predictions contain NaN"
assert not np.any(np.isinf(dist.mean())), "Predictions contain Inf"
```

### Parallel HPC Execution (SLURM)

Run the full benchmark in parallel across datasets:

```bash
# All datasets (0–103) in parallel:
sbatch --array=0-103 run_benchmark.sbatch

# Single dataset:
sbatch --array=42 run_benchmark.sbatch

# Sequential mode:
sbatch run_benchmark.sbatch
```
