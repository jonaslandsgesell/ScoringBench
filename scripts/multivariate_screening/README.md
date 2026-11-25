# Multivariate dataset screening

This folder holds the code that decides which OpenML datasets enter the multivariate ScoringBench suite. No benchmark-model scores are involved. Every decision comes from metadata filters and a model-free conditional-dependence test.

The benchmark itself does not import this code. It reads the frozen list in `scoringbench/multivariate/multivariate_datasets.json`.

All outputs and the OpenML cache live under `scripts/multivariate_screening/results/` (git-ignored).

## Full run (one line)

Screen all of OpenML end to end, reusing the cached API responses, and write the selected suite:

```bash
python scripts/multivariate_screening/build_pool.py && python scripts/multivariate_screening/screen_pool.py
# -> scripts/multivariate_screening/results/screened_kci_datasets.json   (KCI-selected suite, manifest schema)
```

The stages below explain each step; for a quick end-to-end check first, add `--max_pool 100` to `build_pool.py`.

## Pipeline

Run all commands from the repository root.

### 1. Describe the pool: `build_pool.py`

```bash
python scripts/multivariate_screening/build_pool.py
# -> scripts/multivariate_screening/results/openml_pool.parquet
```

This scans **every active OpenML dataset** (~7,000) and, by default, writes a row only for those the uploader declared as multi-target (`d_y > 1`) — about 2% of OpenML. Whether a dataset is multivariate comes only from OpenML's own metadata: the uploader's `default_target_attribute`, which is a comma-separated list for multi-target uploads. We never choose target columns ourselves, so the starting population can't be criticised as hand-picked.

OpenML's dataset-list API cannot filter on the number of targets (its only metadata filters are `number_instances`, `number_features`, `number_classes`, `number_missing_values`), so `d_y` is read from each dataset's description and the `d_y > 1` filter is applied client-side while the pool is built. The full scan still visits every description (cached after the first run), but the written pool is small. Pass `--include_singletarget` to keep the unfiltered population instead; the remaining selection filters stay downstream in `screen_pool.py`.

| column | meaning |
| --- | --- |
| `openml_id`, `name`, `version`, `upload_date` | OpenML identity |
| `n_rows` | number of instances |
| `d_y` | number of declared default targets |
| `d_x` | remaining columns, excluding declared row-id/ignore columns |
| `d_y_numeric`, `d_x_numeric` | numeric targets / numeric features (only filled when `d_y > 1`) |
| `n_missing_values` | missing cells in the whole table |
| `target_cols` | declared targets, in upload order |
| `numeric_target_cols` | numeric subset of `target_cols` (only when `d_y > 1`) |
| `excluded_cols` | OpenML's declared row-id/ignore columns |

API responses are cached in `scripts/multivariate_screening/results/openml_pool_cache/`, which git ignores, so a rerun only fetches new datasets. The first run makes about 7,000 small requests and takes a while. Options: `--output`, `--cache_dir`, `--workers` (default 8), `--include_singletarget`, and `--max_pool N` (stop once N datasets are in the pool — for a quick end-to-end smoke test before a full run).

### 2. Filter and screen the pool: `screen_pool.py`

```bash
python scripts/multivariate_screening/screen_pool.py                 # filter + KCI screen
python scripts/multivariate_screening/screen_pool.py --filter_only   # just write the candidates
# -> scripts/multivariate_screening/results/kci_continuous_audited_v3/pool/
```

The script applies only mechanical filters, in this order:

1. `declared_multitarget`: `d_y >= 2` (a no-op on a default pool, which is already `d_y > 1`; it still runs so an `--include_singletarget` pool is handled)
2. `numeric_targets`: at least 2 numeric declared targets. This removes multi-label classification uploads. Non-numeric declared targets are dropped from both Y and X.
3. `has_features`: at least 1 numeric feature
4. `min_rows`: `n_rows >= --min_rows` (default 100; the smallest dataset in the current suite, slump, has 103)
5. `latest_version`: keep the highest version of each dataset name

Each step's count and the datasets it removed are written to `filter_log.json`. The survivors are written to `candidates.json`, which has the same schema as `multivariate_datasets.json`. Then every candidate is screened at `d = 2` by default (`--target_dims`), using the **first `d` declared numeric targets in upload order**. `d = 2` is the suite entry gate — a dataset is admitted when its two most-dependent targets are jointly dependent given X — and the manifest keeps the full numeric target list, so the benchmark can still run each dataset at higher `d`. Screening higher `d` as well (`--target_dims 2 3 5 10`) is much slower (one KCI test per dataset per `d`) and does not change admission; pass it only when you want the per-`d` decisions recorded.

### 3. Re-check an existing manifest: `screen_manifest.py`

```bash
python scripts/multivariate_screening/screen_manifest.py                       # the shipped suite
python scripts/multivariate_screening/screen_manifest.py --manifest my.json    # any manifest
# -> scripts/multivariate_screening/results/kci_continuous_audited_v3/manifest/
```

This runs the same screen as step 2 on a manifest you supply. Options: `--source`, `--target_dims`, `--dataset_index` (screen a single entry), `--quality_only`.

## What the screen does

For each candidate at each `d` (`decisions.screen_candidate`):

1. **Curation split.** A fixed random 20% of rows, capped at 1,000, is used for screening only. The rest is the benchmark split. Missing features are imputed on the curation split only.
2. **Quality gates** (`screening.py`). Rejects non-continuous targets, targets leaked into the features (copies, affine maps, sums), redundant (collinear) targets, and the hard-coded `KNOWN_NONCONTINUOUS` / `DUPLICATE_FAMILIES` IDs in `decisions.py`.
3. **Dependence test** (`ci_screen.py`). A KCI test of residual dependence between the targets given X, at `ALPHA = 0.05`, with a residual distance-correlation effect-size gate (`EFFECT_MIN = 0.20`) and a target scale-ratio gate (`SCALE_RATIO_MAX = 500`).
4. **Family correction** (`decisions.finalize_suite`). Holm correction across all candidates within each `d`, and exact-duplicate detection through data fingerprints.

## Outputs

| file | content |
| --- | --- |
| `filter_log.json` | pool filters, counts, removed datasets (pool run only) |
| `candidates.json` | candidate manifest after filtering (pool run only) |
| `d{d}/screening/<dataset>_d{d}.json` | full decision per candidate and dimension |
| `suite_draft.json` | all decisions, Holm-adjusted, with package versions and a `suite_id` hash |
| `report.md` | summary table per `d`, plus one decision row per candidate |
| `../screened_kci_datasets.json` | the KCI-**selected** subset, in `multivariate_datasets.json` schema (written at the results root by `screen_pool.py`) |

To adopt a new suite, copy `results/screened_kci_datasets.json` to `scoringbench/multivariate/multivariate_datasets.json`. This changes the `--dataset_index` mapping, so restart any running sweeps.

## Tests

```bash
pytest tests/multivariate/test_screening.py tests/multivariate/test_ci_screen.py tests/multivariate/test_kci_smoke.py
```

`pytest.ini` puts `scripts/` on the path, so tests import `multivariate_screening.*`.
