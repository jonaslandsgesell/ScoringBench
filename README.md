## Why this benchmark?
Proper scoring rules have long been used to rigorously evaluate probabilistic forecasts, but their application has been largely confined to classification tasks. ScoringBench is a Benchmark for **probabilistic regression** — an inherently continuous setting where models must predict full predictive distributions over real-valued targets.

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

- `run_bench_regression.py`: run the univariate benchmark (all datasets, models, CV folds). Use `--lite` for a fast smoke test and `--output_dir` to change the output path.
- `run_bench_regression_multivariate.py`: run the multivariate (d-dimensional target) benchmark. Defaults to writing `output_multivariate_d{d}_n{sample_size}/`. See the [Multivariate benchmark](#multivariate-benchmark-d-dimensional-targets) section.
- `autorank_leaderboard.py`: compute statistical rankings with critical-difference diagrams; generates JSON data and LaTeX tables in `<output_dir>/figures/leaderboard/`. Use `--output_dir` to choose the input/output folder (default `output_3000`). Works for both univariate and multivariate outputs.

## Related tools

- [autorank](https://sherbold.github.io/autorank/) — statistical ranking and critical-difference diagrams

## Benchmark output (summary)

Each run writes per-dataset per-model raw Parquet files to `output/raw/{model_name}/{dataset_name}.parquet`. This structure avoids concurrency issues when running multiple datasets in parallel (SLURM array jobs).

Typical directory structure:

- `output/raw/{model_name}/{dataset_name}.parquet` — raw results organized by model and dataset
- `output/{model_name}.parquet` — aggregated per-model parquet files (after running autorank_leaderboard.py)


## Workflow

1. git clone --recurse-submodules https://github.com/jonaslandsgesell/ScoringBench.git
2. Add your custom wrapper with a unique name (see `scoringbench/univariate/wrappers/` and inherit `ProbabilisticWrapper`).
3. python run_bench_regression.py
4. python autorank_leaderboard.py
5. Commit aggregated per-model Parquet files (`output/*.parquet`) and the generated JSON ranking files in `output/figures/leaderboard/` to git LFS. Since the output repository is separate from the main repository, push to both. This serves as a public ledger and allows traceability.
6. Create a pull request to the ScoringBench repository for review; contributions that meet standards will be merged.
7. Upon merge, https://scoringbench.com will automatically display the updated leaderboard; the data is also available in the repository.

## Multivariate benchmark (d-dimensional targets)

The multivariate benchmark evaluates **purely sample-based** models on
`d`-dimensional targets using proper scoring rules that are estimated directly
from draws (energy score, variogram score, Dawid-Sebastiani).

Everything lives under `scoringbench/multivariate/`; edit
`scoringbench/multivariate/models.py` (`MODELS`) to add / swap models.

### Dataset sources (`--source`)

The multivariate targets can come from two interchangeable **sources**, selected
with `--source`. Both return the same `(X, Y)` contract (`Y` has columns
`target_0 .. target_{d-1}`, `target_0` first), so every downstream step
(scoring, aggregation, leaderboard) is identical regardless of source.

| `--source`     | How the `d` targets are built |
| -------------- | ----------------------------- |
| `scoringbench` | **(default)** Promote real data: take a standard 1-D regression dataset and promote the `d-1` features most correlated with the original target into extra targets (the original target becomes `target_0`). |
| `synthetic`    | Nonlinear conditional means plus feature-independent, jointly sampled R-vine residuals (see below). |

```bash
# Real feature-promotion datasets (default)
python run_bench_regression_multivariate.py --source scoringbench

# Synthetic random-vine datasets
python run_bench_regression_multivariate.py --source synthetic
```

The output directory name is **prefixed with the source** so runs from different
sources never collide:

```
output_multivariate_{source}_d{d}_n{sample_size}/
# e.g. output_multivariate_scoringbench_d3_n3000/
#      output_multivariate_synthetic_d3_n1000/
```

`--dataset_index` selects a single dataset out of the list that source
enumerates (useful for SLURM array jobs).

### Synthetic source: randomized R-vines

The generator follows the model-randomization design in
[Nagler, Schellhase and Czado (2017), Section 4.1](https://arxiv.org/html/1701.00845v2#S4.SS1),
using Vatter's suggested direct uniform R-vine structure sampling and
normal-quantile transformation. It uses
[`pyvinecopulib`](https://github.com/vinecopulib/pyvinecopulib) for vine
construction and sampling.

For each dataset, draw one vine over the `target_dim` residual variables,
independently of the standard-normal features `X`. Apply `norm.ppf` to the
vine samples, clipping uniforms to `[1e-12, 1 - 1e-12]` to keep values finite.
The targets, named `target_0 .. target_{d-1}`, are always generated as

$$Y_k = \sqrt{0.6}\,f_k(X) + \sqrt{0.4}\,\varepsilon_k.$$

Each randomized mean uses 12 symmetrized ridge terms and 6 product terms on
up to 3 randomly selected features. It satisfies `f_k(X) = f_k(-X)`, so its
population linear projection is constant under the normal feature distribution.
A fixed, independent 2048-row calibration sample approximately standardizes
each function; its definition never depends on the evaluation batch. Nonlinear
means are unconditional, with no enable/disable option.

The residuals have standard-normal margins and a **target-only** copula,
independent of `X`. Consequently the same nontrivial copula remains in `Y | X`.
This preserves what a joint model can learn that an independent model cannot;
strong dependence in a single vine over both `X` and `Y` would not ensure this.

The three scenarios use strong dependence with three family regimes:

| Factor | Choices |
| ------ | ------- |
| Dependence strength | `strong`: absolute tau drawn from `Beta(5, 5)` |
| `tail` families | Student-t with 4 degrees of freedom, Clayton, Gumbel, equally likely per edge |
| `no_tail` families | Gaussian or Frank, equally likely per edge |
| `mixed` families | Choose tail/no-tail with probability 1/2, then uniformly within that group |

Every edge receives an independently drawn strength and sign. Clayton and
Gumbel use uniformly sampled rotations of 0, 90, 180 or 270 degrees. The
absolute tau is `min(beta_draw, SYNTHETIC_TAU_UPPER) * SYNTHETIC_DECAY**level`,
where tree levels start at **one**. Defaults are `SYNTHETIC_TAU_UPPER=1.0` and
`SYNTHETIC_DECAY=0.8`, giving a first-tree expected absolute tau of 0.40.
There is no positive tau floor; higher trees can approach independence.
Parameters are bounded by the numerical limits of the selected library family.
The randomization follows the paper's strong-dependence scenarios; the nonlinear
regression construction above is an additional benchmark design choice.

The `SYNTHETIC_*` settings in `scoringbench/multivariate/config.py` control the
suite. The default **100** datasets per `(d, n)` shape are balanced across the
three scenarios (33 or 34 replicates each). Each replicate draws a new structure
and edge parameters; the model stays fixed for all observations in that dataset.
Independent seeded streams separate structure, families, strengths, rotations,
features, mean functions and residual observations. Changing the observation count or expanding the suite does
not change an existing replicate's generating model.

These are randomized parametric simplified-vine scenarios, not guarantees that
one fitted estimator always wins. Tests check nonlinear held-out predictability,
the residual copula's invariance to feature count, and better joint scores for
the true copula than for independently shuffled forecasts with identical
marginals, alongside determinism and artifact round trips. Point scores alone
cannot reward correct dependence when the marginal means are identical.

#### Reproducibility: frozen artifacts, generated explicitly (no on-the-fly fallback)

To stay reproducible across `numpy` / `pyvinecopulib` versions (whose RNG
streams are not guaranteed stable), the generated arrays are **frozen as
committed parquet artifacts**, with a `manifest.json` recording the generating
parameters, the nonlinear mean specification, the complete realized residual
vine and its signed edge taus, each
artifact's `sha256`, and the library versions used.

Artifacts are **scoped by shape** into a per-`(d, n)` subfolder so that
different `target_dim` / `sample_size` sweeps never collide (each subfolder
carries its own manifest):

```
datasets/synthetic/
  d3_n1000/
    syn_tail_strong_d3_r0_{hash}.parquet
    ... (SYNTHETIC_N_DATASETS datasets across three scenarios)
    manifest.json
  d2_n3000/
    ...
```

**Loading never regenerates on the fly.** It reads those exact bytes (and
verifies the manifest, configuration, `sha256`, shape and columns). If the artifact for the
requested `(d, n)` shape is **missing**, the loader raises `FileNotFoundError`
telling you to generate it explicitly — this guarantees results always come from
the committed, version-pinned bytes rather than a silent, possibly drifted
regeneration.

So before running `--source synthetic` with a given `--target-dim d` /
`--sample-size n`, generate that shape once:

```bash
# writes datasets/synthetic/d{d}_n{n}/*.parquet + manifest.json
PYTHONPATH=. python scripts/generate_synthetic.py --target-dim 3 --sample-size 1000
python run_bench_regression_multivariate.py --source synthetic --target_dim 3 --sample_size 1000
```

Run the generator without shape arguments to use the benchmark's configured
defaults. Use `--force` only to explicitly replace an existing frozen set.

Dataset hashes include the sampling design and nonlinear mean settings, so
changed generators do not reuse cached dataset identities. When changing the
generator, use a fresh benchmark `--output_dir` as well: existing aggregates
and raw results belong to the previous datasets and must not be mixed in.

> The parquet artifacts are binary and can grow with `d` / `n`; if you track
> them in git, add them to **git-LFS** (e.g. `git lfs track
> "datasets/synthetic/**/*.parquet"`) and keep the `manifest.json` files off LFS
> so their diffs stay readable.

### Running the multivariate benchmark

```bash
# 5-fold CV, all datasets, defaults (d=3, sample_size=3000)
python run_bench_regression_multivariate.py

# Fast smoke test (2-fold CV)
python run_bench_regression_multivariate.py --lite

# Choose the target dimension d and the per-dataset sample size
python run_bench_regression_multivariate.py --target_dim 3 --sample_size 3000

# Run a single dataset (e.g. for SLURM array jobs)
python run_bench_regression_multivariate.py --dataset_index 0

# Pick the dataset source (real feature-promotion vs synthetic random vines)
python run_bench_regression_multivariate.py --source synthetic

# Explicit output directory (overrides the default naming)
python run_bench_regression_multivariate.py --output_dir my_results/
```

By default the results are written to a folder whose name **encodes the source,
the target dimension `d`, and the sample size**, so different source / `d` /
sample-size sweeps never overwrite each other:

```
output_multivariate_{source}_d{d}_n{sample_size}/   # e.g. output_multivariate_scoringbench_d3_n3000/
```

Raw per-dataset per-model Parquet files use the same layout as the univariate
benchmark: `output_multivariate_{source}_d{d}_n{sample_size}/raw/{model_name}/{dataset_name}.parquet`.

### Analyzing the multivariate results

The multivariate output is 1:1 compatible with `aggregate_datasets.py` and
`autorank_leaderboard.py` — just point them at the multivariate output folder:

```bash
# Aggregate raw per-dataset files into per-model files (optional; the
# leaderboard script also aggregates automatically)
python aggregate_datasets.py \
    --raw_dir output_multivariate_d3_n3000/raw \
    --out_dir output_multivariate_d3_n3000

# Statistical rankings + critical-difference diagrams over all multivariate
# scoring rules (energy_score_beta_*, variogram_score_p_*, dawid_sebastiani, ...)
python autorank_leaderboard.py --output_dir output_multivariate_d3_n3000
```

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
from scoringbench.univariate.wrappers.tabpfn import TabPFNWrapper
from scoringbench.univariate.metrics import compute_metrics

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
from scoringbench.univariate.wrappers.tabpfn import TabPFNWrapper
from scoringbench.univariate.metrics import compute_metrics

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
