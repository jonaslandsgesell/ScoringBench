# Migration: `scoringbench` → univariate / multivariate subpackages

This repo is being restructured into two **self-contained, symmetric**
subpackages under `scoringbench/` ("Alternative A, pure form"), with **no import
path between them in either direction**.

## New tree

```
scoringbench/
├── __init__.py          # version only — no eager subpackage import
├── version.py           # stays at top level
├── univariate/          # everything that existed before (pure move)
│   ├── __init__.py       # old eager re-exports live here now
│   ├── _integration.py
│   ├── config.py
│   ├── cv.py
│   ├── datasets.py
│   ├── latex_tables.py
│   ├── metrics.py
│   ├── models.py
│   ├── results.py
│   ├── runner.py
│   ├── utils.py
│   └── wrappers/
├── multivariate/        # scaffolding only so far (await instructions)
│   └── __init__.py
└── common/              # intentionally EMPTY for now
    ├── __init__.py
    └── README.md

tests/
├── conftest.py          # GPU-cleanup autouse fixture (applies to all subtrees)
└── univariate/          # all pre-existing tests moved here
    ├── analytical/
    └── wrapper/
```

## Decisions

### No back-compat shims — `scoringbench.metrics` is now an `ImportError`

We deliberately did **not** leave `scoringbench/metrics.py` (or the other former
top-level modules) forwarding to `scoringbench.univariate.metrics`. A shim would
silently defeat the "no cross-talk" property this restructure exists to create.
`scoringbench.metrics` now raises `ModuleNotFoundError`; use
`scoringbench.univariate.metrics` instead. All call sites were mechanically
rewritten.

Top-level `scoringbench/__init__.py` exposes **only** `__version__` — no eager
subpackage import — so `import scoringbench` never drags in the optional heavy
dependencies of both worlds.

### `common/` is intentionally empty

`scoringbench/common/` is a placeholder. Nothing is hoisted into it during this
work. It will be populated only later, after the split is in place and the true
shared surface has been observed in practice. **Duplication between `univariate/`
and `multivariate/` is ACCEPTED and EXPECTED at this stage.** See
`scoringbench/common/README.md`.

### Deferred: output-layout restructuring

The `output/` submodule and `output_1000..8000/` committed results, plus
`aggregate_datasets.py` / `autorank_leaderboard.py`, are **out of scope** and
untouched. The future multivariate runner will write to a flat top-level
`output_multivariate/` (already added to `.gitignore`) using the existing
`raw/{model}/{dataset}.parquet` convention; a `# TODO(output-layout):` will mark
the write site as provisional when that runner is built.

### Deferred: `N_DRAWS` fairness note (multivariate)

Every sample-based estimator is biased in the number of draws `m`, and the bias
differs per rule — so a model that can afford more draws could win on estimator
bias alone. When the multivariate package is built, `config.py` will pin one
benchmark-wide `N_DRAWS` and every persisted output row will record it.

## Status

- **Phase 1 (pure move)** — done and committed
  (`refactor: move scoringbench to scoringbench.univariate (pure move)`).
- **Phase 2 (empty `common/` + `multivariate/` scaffold)** — done.
- **Phase 3 (build `multivariate/`)** — deferred; only the package folder is
  scaffolded. Awaiting instructions.
