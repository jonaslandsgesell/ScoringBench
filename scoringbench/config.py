"""Global benchmark configuration.

All hyperparameters live here so you only ever touch one file to
change the training regime.  Models and the CV loop import from here.
"""

# ---------------------------------------------------------------------------
# Cross-validation / sampling
# ---------------------------------------------------------------------------
SEED: int = 42
N_FOLDS: int = 5
SAMPLE_SIZE: int = 3_000   # cap applied when a dataset has no per-dataset limit

