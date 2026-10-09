# Multivariate benchmark (d-dimensional targets)

The multivariate benchmark evaluates **purely sample-based** models on
`d`-dimensional targets using proper scoring rules that are estimated directly
from draws (energy score, variogram score, Dawid-Sebastiani).

Everything lives under `scoringbench/multivariate/`; edit
`scoringbench/multivariate/models.py` (`MODELS`) to add / swap models.

## Dataset sources (`--source`)

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

## Synthetic source: randomized R-vines

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

### Reproducibility: frozen artifacts, generated explicitly (no on-the-fly fallback)

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

## Running the multivariate benchmark

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

## Analyzing the multivariate results

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
