# Wrapper patterns for ScoringBench

Every model is a single wrapper module at `scoringbench/univariate/wrappers/<model_key>.py`
that produces a `DistributionPrediction`. This reference stays deliberately
generic — **read the real files** for current signatures and details rather than
trusting any snippet here:

```
scoringbench/univariate/wrappers/
  base.py              # DistributionPrediction, ProbabilisticWrapper (the contract)
  sample_based.py      # SampleBasedWrapper + the shared converters
  <model_key>.py       # <-- your new wrapper lives here
```

---

## The three ways to provide a wrapper

A `DistributionPrediction` (see `base.py`) holds a per-bin PMF (`probas`, sums to
1 per row) over a grid (`bin_edges`/`bin_midpoints`). The grid can be **shared**
(1-D, same for every observation) or **per-observation** (2-D, one grid per row).
That, plus sampling, gives exactly three ways to build a wrapper. Pick the one
that matches how your model emits a distribution, then open the listed reference
file(s) and adapt.

### 1. Global PMF grid — shared 1-D grid for all observations

The model scores every observation on **one fixed `y`-grid**: either class
probabilities over discretized bins, or a density `p(y|x)` evaluated on a shared
grid. Build with `grid_density_to_distribution(...)` (in `sample_based.py`), or
assemble `DistributionPrediction` directly with 1-D `bin_edges`/`bin_midpoints`.
References: `xgb_vector.py` (per-bin probabilities), `cde_wrapper.py`,
`flexcode_wrapper.py` (analytic density on a shared grid).

### 2. Per-observation PMF grid — 2-D grid, one per observation

The model emits **per-observation quantiles**, either analytic (`ppf` of a
parametric distribution) or predicted directly. Convert with
`quantiles_to_distribution(...)` (in `quantile_based.py`); it re-samples the CDF
implied by the quantiles on a regular per-sample grid. References:
`ngboost_wrapper.py` (closed-form `ppf`), `catboost_wrapper.py`,
`crepes_wrapper.py` (predicted quantiles).

### 3. Sample-based wrapper — draws conditional samples of `y`

The model can only **draw conditional samples** of the target. Subclass
`SampleBasedWrapper` and implement only `_draw_samples(self, X, n_samples)`
returning an `(n_test, n_samples)` array; the base class collects draws under a
wall-clock budget and builds the PMF for you (`predict`/`predict_distribution`
are provided). Behaviour is tuned via class attributes — read `sample_based.py`
for the current attribute names and defaults. References: `nflows_wrapper.py`,
`bart_wrapper.py`.

---

## The wrapper contract

Read `base.py` and `sample_based.py` for the authoritative signatures, then
follow all of these:

1. **Subclass the right base.** `ProbabilisticWrapper` for ways 1–2,
   `SampleBasedWrapper` for way 3.
2. **Implement the required methods.** Ways 1–2: `fit(X, y)` returns `self`,
   `predict(X)` returns the mean as a 1-D array, `predict_distribution(X)`
   returns a `DistributionPrediction`. Way 3: implement `fit` and
   `_draw_samples` only.
3. **Keep the third-party import lazy** (inside `fit`/`_build_model`, never at
   module top level) so `wrappers/__init__.py` can bind the class to `None` when
   the library is absent.
4. **Sanitize.** Cast `X` to float and replace non-finite values; drop non-finite
   `y` rows at fit; pass a `y_range` to the converters to clamp bad
   quantiles/samples.
5. **Document and parameterize.** Class docstring with one-line description, paper
   title/authors, code URL, license. Expose hyperparameters as constructor args
   with defaults stored as attributes. Do not auto-tune unless asked.

Potentially use the shared converters (`quantiles_to_distribution` in
`quantile_based.py`; `samples_to_distribution` and `grid_density_to_distribution`
in `sample_based.py`) over hand-building `DistributionPrediction`, so the on-disk
representation matches every other model.

---

## Registration edits

Read the current contents of each file before editing and match its style.

- **`scoringbench/univariate/wrappers/__init__.py`** — add `("<ClassName>", "<model_key>")`
  to the lazy-import list and `"<ClassName>"` to `__all__`.
- **`run_bench_regression.py`** — import `<ClassName>` and add one `MODELS` entry
  keyed by the unique, version-stamped `model_name`. The value is a zero-arg
  factory (`lambda: <ClassName>(...)`) so each CV fold gets a fresh instance. For
  many configs, build a `dict_*` comprehension like the existing ones and use a
  late-binding default arg (`lambda p=preset: ...`) to avoid the closure bug.
- **`requirements_models.txt`** — one line for the optional dependency.

---

## Naming rules (enforce before writing code)

- The `MODELS` key is the on-disk results folder (`output/raw/<key>/`), the
  aggregated `output/<key>.parquet`, and the leaderboard label.
- It MUST be unique — reusing a key overwrites another model's ledger. Verify with
  `grep -oE '"[a-z0-9_.]+":\s*lambda' run_bench_regression.py` and `ls output/raw/`.
- It MUST encode the version/config (library/checkpoint version, distribution
  family, loss). Bumping a version means a NEW key — never re-point an old one.
- lowercase + digits + underscores only; `v2.5` → `v2_5`.
