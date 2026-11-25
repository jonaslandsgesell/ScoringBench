"""Model-free dependency screening for the multivariate ScoringBench.

Pipeline (run from the repository root):

1. ``python scripts/multivariate_screening/build_pool.py`` -- describe every
   active OpenML dataset (``d_x``, ``d_y`` from the uploader's declared default
   targets) into ``results/openml_pool.parquet``. No selection happens here.
2. ``python scripts/multivariate_screening/screen_pool.py`` -- apply explicit,
   logged filters to that pool and run the KCI screen on what remains.
3. ``python scripts/multivariate_screening/screen_manifest.py`` -- re-check an
   existing manifest (by default the shipped ``multivariate_datasets.json``).

Modules: :mod:`.ci_screen` (KCI + residual distance correlation tests),
:mod:`.screening` (curation split, data-quality gates), :mod:`.decisions`
(per-candidate decision, Holm correction, persistence).
"""
