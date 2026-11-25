# Feature-promotion: univariate → multivariate ScoringBench

This folder holds the protocol that **promotes** the univariate ScoringBench
suite (`datasets.json`, 100 single-target regression datasets) into a
*multivariate* one by turning the most **conditionally dependent** features into
additional targets. The goal is a suite whose joint targets are at least as
strongly conditionally dependent — `residual_dcor ≥ EFFECT_MIN` under the same
KCI screen — as the native `multitarget_openml` suite, so a joint model can beat
a product of marginals.

No benchmark-model scores enter selection; it is metadata + a model-free screen,
reusing `scripts/multivariate_screening` (`ci_screen`, `screening.screen_dataset`).

## Why conditional (not marginal) dependence

A joint model only helps when `Q(Y₁…Y_d | X) ≠ ∏ⱼ Q(Yⱼ | X)`. Features that only
correlate with the target *marginally* can be conditionally independent of it
once the other features are given. So candidates are ranked by **partial
distance correlation**: residualize the anchor target and the candidate on the
conditioning features (out-of-bag random forest, ridge for wide X), then
distance-correlate the two residuals.

## Protocol (per dataset)

1. Load `(X, y)` via the univariate loader (openml / pmlb / keel).
2. **Continuous** numeric columns are the only admissible targets (lattice /
   discrete columns fail the multivariate continuous-target gate). `target_0` is
   the original target `y` when it is continuous; otherwise the discrete `y` is
   dropped and the block anchors on the continuous feature most dependent on `y`.
3. On the fixed curation split (`ci_screen.curation_split`; the benchmark never
   sees it), rank candidates by partial distance correlation with the anchor given
   every other feature, and take the top ones, up to `MAX_PROMOTE = 9`
   (→ `d = 10` with the anchor), always leaving `MIN_CONDITIONERS = 2` features
   as inputs.
4. **Refine** (`REFINE = 2` passes): re-rank every candidate against the
   conditioning set the screen will actually use (all numeric features minus the
   current promoted block). Stop early when the block no longer changes. Without
   this, a feature whose link to the anchor runs through another promoted feature
   is ranked too low.
5. Freeze `all_target_cols = [anchor, f₁, …, f_k]` and screen it at
   `d = 2, 3, 5, 10` with the multivariate screen (quality gates + KCI + residual
   distance-correlation effect size, `PROMOTE_EFFECT_MIN = 0.15`). The whole
   promoted block is removed from the inputs at every `d`, as in
   `native_scoringbench`, so a low-`d` run can't see a not-yet-used target as a
   feature. Selected at `d` ⇔ targets jointly KCI-dependent given X with
   `residual_dcor ≥ 0.15`.

Result (2026-11-26, 100 univariate datasets): 45 selected at `d = 2`
(median residual distance correlation 0.607), 38 at `d = 3`, 23 at `d = 5`,
7 at `d = 10`. The suite (selected at `d = 2`) has 45, 36, 22 and 6 datasets at
`d = 2, 3, 5, 10`.

## Run

```bash
python scripts/promote_targets/promote.py                 # all 100 univariate datasets
python scripts/promote_targets/promote.py --max_datasets 15   # quick yield estimate
# options: --refine N (default 2)  --seed  --dataset_index  --merge
# cluster (from the repo root): sbatch --array=0-99 scripts/promote_targets/run_promote_targets.sbatch,
#   then promote.py --merge
# -> scripts/promote_targets/results/promoted_scoringbench_datasets.json   (the suite)
#    scripts/promote_targets/results/report.md                (decisions + effect-size vs multitarget_openml)
```

## Adopt as a benchmark source

`promoted_scoringbench_datasets.json` is read by the `promoted_scoringbench` source
(`scoringbench/multivariate/promoted_scoringbench.py`). To benchmark it, copy the
manifest next to that module and run the benchmark:

```bash
cp scripts/promote_targets/results/promoted_scoringbench_datasets.json \
   scoringbench/multivariate/promoted_scoringbench_datasets.json
python run_bench_regression_multivariate.py --source promoted_scoringbench --target_dim 2
# cluster: set SOURCE="promoted_scoringbench" (and TARGET_DIM) in run_benchmark_multivariate.sbatch, then
#   sbatch --array=0-<N-1> run_benchmark_multivariate.sbatch
```

Each entry carries the univariate config and `all_target_cols` (ordered
most-conditionally-dependent-first); the source truncates to the leading
`target_dim`, and `selected_dims` controls which datasets appear at each `d`.
