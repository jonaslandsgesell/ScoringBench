"""Global benchmark configuration.

All hyperparameters live here so you only ever touch one file to
change the training regime.  Models and the CV loop import from here.
"""

# ---------------------------------------------------------------------------
# Cross-validation / sampling
# ---------------------------------------------------------------------------
SEED: int = 42
N_FOLDS: int = 5
N_REPEATS_CV: int = 1      # repeated CV: each repeat draws a fresh subsample of the dataset
SAMPLE_SIZE: int = 3_000   # max rows fed into KFold (train+test) per repeat; 0/None = no cap

