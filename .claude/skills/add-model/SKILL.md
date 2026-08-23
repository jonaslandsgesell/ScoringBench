---
name: add-model
description: Add a new probabilistic regression model to the ScoringBench benchmark. Use this skill whenever the user wants to integrate a new tabular probabilistic-regression model into ScoringBench — even if they just say "add X model", "integrate X", "support X", or "wrap X for the benchmark". Creates the ProbabilisticWrapper subclass, registers it for lazy import, wires it into the MODELS registry in run_bench_regression.py under a UNIQUE, version-stamped model name, and documents how to run, rank, and submit results. Reads existing similar wrappers for inspiration and optionally fetches a documentation URL to understand the new model's API.
argument-hint: <ModelName> [<pip-package>] [<doc-url>]
user-invocable: true
---

# Add a model to ScoringBench

This skill integrates a new probabilistic-regression model into ScoringBench so it
can be benchmarked with proper scoring rules (CRPS, CRLS, Interval Score,
Beta-Energy, …) and appear on the leaderboard.

Every model is exposed through a **wrapper** in
`scoringbench/univariate/wrappers/<file>.py` that subclasses `ProbabilisticWrapper` and emits a
`DistributionPrediction` (a discretized predictive distribution). The wrapper is
registered for lazy import in `scoringbench/univariate/wrappers/__init__.py`, then wired into the
`MODELS` dict in `run_bench_regression.py` under a **unique, version-stamped model
name** (the name becomes the on-disk results key and the leaderboard label).

You typically touch **four files** and **add one wrapper module**:

1. `scoringbench/univariate/wrappers/<model_key>.py` — new wrapper module
2. `scoringbench/univariate/wrappers/__init__.py` — lazy-import registration + `__all__`
3. `run_bench_regression.py` — import the class and add a `MODELS` entry
4. `requirements_models.txt` — declare the optional pip dependency
5. `tests/wrapper/test_wrapper_integration.py` — add the model to the shared
   integration suite

Read `references/wrapper_patterns.md` for the three ways to build a wrapper and
pointers to the relevant source files.

---

## Step 0: Gather inputs

Parse `$ARGUMENTS` for the model name. Collect the following (ask only for what is
missing or unclear):

| Input | Example | Notes |
|---|---|---|
| `ModelName` | `"NGBoost v0.5"` | Human-readable name of the method |
| `model_name` (registry key) | `"ngboost_normal_v0_5"` | **UNIQUE, version-stamped** key used in `MODELS` and on disk. See "Naming rules" below. |
| `model_key` (file/key) | `ngboost` | snake_case module name under `scoringbench/univariate/wrappers/` |
| `ClassName` | `NGBoostWrapper` | CamelCase wrapper class, suffixed `Wrapper` |
| `pip_package` | `"ngboost>=0.5.0"` | Pip install spec for `requirements_models.txt` |
| `doc_url` | `"https://..."` | Documentation / GitHub / paper URL (optional) |
| `wrapper_kind` | `per-observation grid` | Which of the **three ways** the model emits a distribution (see `references/wrapper_patterns.md`): (1) global PMF grid — one shared 1-D `y`-grid for all observations; (2) per-observation PMF grid — 2-D grid built from per-sample quantiles; (3) sample-based — draws conditional samples. Drives which reference wrapper you adapt. |
| `supports_gpu` | `true` | Whether the library uses GPU |

### Naming rules (critical — enforce these)

The `model_name` registry key is the **single source of identity**: it is the
key in `MODELS`, the folder under `output/raw/<model_name>/`, the aggregated
`output/<model_name>.parquet`, and the leaderboard label. Therefore:

- **It MUST be unique.** Reusing an existing key silently overwrites another
  model's results and corrupts the leaderboard ledger. Before choosing a name,
  list the existing keys and confirm there is no collision:
  - `grep -oE '"[a-z0-9_.]+":\s*lambda' run_bench_regression.py` (and check the
    `dict_*` comprehensions for templated keys), and
  - `ls output/raw/` to see names that already have committed results.
- **It MUST encode the version** of the model/checkpoint/library (e.g.
  `tabpfn_v2_6`, `tabpfn_realv2_5`, `tabiclv2`, `xgblss_Gaussian`). Bumping a
  model version means a **new key** (e.g. `ngboost_normal_v0_5` →
  `ngboost_normal_v0_6`) — never re-point an existing key at a new version, or
  historical comparisons break.
- Use lowercase, digits and underscores only; replace dots in versions with
  underscores (`v2.5` → `v2_5`). Encode the key distinguishing config in the
  name (distribution family, loss, checkpoint) so two configs of the same
  library don't collide.

State the chosen `model_name` back to the user and explicitly confirm it is new
and version-stamped before writing any code.

---

## Step 1: Understand the model API

If `doc_url` was provided, fetch it to understand:

- Import path (e.g. `from ngboost import NGBRegressor`).
- Constructor parameters and sensible defaults.
- `.fit(X, y, ...)` signature.
- How the model exposes its **predictive distribution**: a parametric object
  with `.ppf`/`.cdf`/`.dist`, a quantile/`predict(..., quantiles=…)` call, a
  `.sample(...)` method, or per-bin class probabilities. This decides which of
  the three ways (and reference wrapper) you adapt.

If no URL is given, infer the API from the user's description and the closest
existing wrapper.

---

## Step 2: Pick the closest existing wrapper as a reference

`references/wrapper_patterns.md` describes the **three ways** to provide a wrapper
and lists the reference file(s) for each. In short:

1. **Global PMF grid** — one shared 1-D `y`-grid for all observations
   (`grid_density_to_distribution`, or a direct `DistributionPrediction`). See
   `xgb_vector.py`, `cde_wrapper.py`, `flexcode_wrapper.py`.
2. **Per-observation PMF grid** — 2-D grid from per-sample quantiles
   (`quantiles_to_distribution`, in `quantile_based.py`). See `ngboost_wrapper.py`,
   `catboost_wrapper.py`, `crepes_wrapper.py`.
3. **Sample-based** — subclass `SampleBasedWrapper`, implement `_draw_samples`.
   See `nflows_wrapper.py`, `bart_wrapper.py`.

Read `references/wrapper_patterns.md` and the reference wrapper closest to your
model now. Use them as a structural guide — adapt rather than copy.

---

## Step 3: Create the wrapper module

Create `scoringbench/univariate/wrappers/<model_key>.py` following the way you picked in
Step 2. The full wrapper contract lives in `references/wrapper_patterns.md`; the
essentials are: subclass the right base, keep the third-party import lazy,
sanitize inputs/outputs, prefer the shared converters over hand-rolling
`DistributionPrediction`, add a docstring (description, paper, code URL, license),
and expose hyperparameters as constructor args. Read `base.py` and
`sample_based.py` for the current signatures before you write code.

---

## Step 4: Register the wrapper for lazy import

Edit [scoringbench/univariate/wrappers/__init__.py](scoringbench/univariate/wrappers/__init__.py):

1. Add a `("<ClassName>", "<model_key>")` tuple to the `_OPTIONAL` list. The loop
   imports the class if its backing library is installed, else binds the name to
   `None` so the package still imports without every competitor's stack present.
2. Add `"<ClassName>"` to `__all__`.

If your wrapper module exposes a presets/registry constant (like
`CDE_PRESETS`), also import it where `run_bench_regression.py` consumes it.

---

## Step 5: Wire the model into the MODELS registry

Edit [run_bench_regression.py](run_bench_regression.py) (and
[run_benchmark.sbatch](run_benchmark.sbatch) if using a cluster):

1. Add `<ClassName>` to the `from scoringbench.univariate.wrappers import (...)` block.
2. Add one `MODELS` entry keyed by the **unique, version-stamped `model_name`**
   from Step 0, whose value is a **zero-arg factory** (`lambda: <ClassName>(...)`)
   so each CV fold gets a fresh instance. For several configs, build a `dict_*`
   comprehension (see `dict_finetuned_models`, `dict_cde_models`) with a
   late-binding default arg to avoid the closure bug. `references/wrapper_patterns.md`
   has the details.
3. Re-confirm the key does not already exist in `MODELS` nor under `output/raw/`.

---

## Step 6: Declare the optional dependency

Add the `pip_package` spec to
[requirements_models.txt](requirements_models.txt) (alongside the existing
optional model deps). Keep it as one line; add a trailing comment if it pulls in
a heavy/conflict-prone transitive tree (see the `surjectors` entry).

---

## Step 7: Register the wrapper in the integration tests

Add the model to the shared wrapper integration suite under
[tests/wrapper](tests/wrapper) so it is exercised on the standard 1-D regression
alongside every other wrapper. Read
[tests/wrapper/test_wrapper_integration.py](tests/wrapper/test_wrapper_integration.py)
(and `test_sample_based_wrappers.py` for way-3 models) and follow its
"Adding a new model" instructions: write a zero-arg `_make_<name>()` factory and
append a `pytest.param` entry to `MODEL_FACTORIES`. Models whose optional package
is missing are skipped automatically. Run `python -m pytest tests/wrapper` and
confirm the new parametrization passes (or skips cleanly when uninstalled).

---

## Step 8: Smoke-test the wrapper

Validate the wrapper in isolation first: on a small synthetic `(X, y)`, call
`fit`, then `predict_distribution`, then `scoringbench.univariate.metrics.compute_metrics`,
and assert the metrics (e.g. `crps`) and `dist.mean()` are finite. Then run a
fast end-to-end smoke test through the harness and confirm finite-metric parquet
files appear under the smoke output dir:

```bash
python run_bench_regression.py --lite --output_dir output_smoke/
python -m pytest tests
```

If the model is GPU-only and no GPU is available, note it for the user rather than
forcing a CPU run.

---

## Step 9: Run, rank, and submit (the contribution workflow)

This is the official ScoringBench contribution workflow from
[README.md](README.md). Walk the user through it — **do not push or open PRs on
their behalf without confirmation**:

1. **Clone with submodules** (if not already cloned):
   ```bash
   git clone --recurse-submodules https://github.com/jonaslandsgesell/ScoringBench.git
   ```
2. **Add your custom wrapper with a unique name** — see `scoringbench/univariate/wrappers/`
   and inherit `ProbabilisticWrapper` (Steps 3–7 above). The `model_name` key in
   `MODELS` must be unique and version-stamped (Step 0).
3. **Run the benchmark** to produce raw per-(model, dataset) results:
   ```bash
   python run_bench_regression.py
   ```
   Outputs land in `output/raw/<model_name>/<dataset_name>.parquet`. On a cluster
   you can fan out across datasets:
   ```bash
   sbatch --array=0-103 run_benchmark.sbatch
   ```
4. **Aggregate + rank** to build the leaderboard:
   ```bash
   python autorank_leaderboard.py
   ```
   This writes aggregated `output/<model_name>.parquet` and the JSON ranking
   files + LaTeX tables in `output/figures/leaderboard/`.
5. **Commit the ledger to git LFS.** Commit the aggregated per-model Parquet files
   (`output/*.parquet`) and the generated JSON ranking files in
   `output/figures/leaderboard/`. Since the output repository is separate from the
   main repository, **push to both**. This serves as a public ledger and allows
   traceability.
6. **Open a pull request** to the ScoringBench repository for review;
   contributions that meet standards will be merged.
7. **Leaderboard updates automatically.** Upon merge, https://scoringbench.com
   will display the updated leaderboard; the data is also available in the
   repository.

Reproducibility checklist to include in the PR description:
- the exact `model_name` (unique + version-stamped) and the library version,
- the wrapper file and the `MODELS` entry,
- the command(s) used and the seed/sample-size settings,
- confirmation that no existing `model_name`/results were overwritten.

---

## Step 10: Report

Summarize what was created/edited:
- the new wrapper module and which of the three ways it uses,
- the registration edits in `__init__.py` and `run_bench_regression.py`,
- the integration-test entry added under `tests/wrapper`,
- the chosen `model_name` (and why it is unique + version-stamped),
- the dependency added to `requirements_models.txt`,
- any TODOs left for the user (e.g. implement `predict_distribution` if the API
  was unclear, run on GPU, tune hyperparameters, then run the Step 9 workflow).
