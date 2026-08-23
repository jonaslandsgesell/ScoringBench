"""Shared utility helpers for the multivariate benchmark.

Mirrors ``scoringbench.univariate.utils`` but does **not** import from the
univariate package (the two subpackages are kept independent by design).
"""
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility.

    CUDA seeding is best-effort: on some nodes the installed PyTorch build is
    incompatible with the GPU (e.g. a Quadro P5000 / sm_61), in which case any
    CUDA call raises. Scoring in this package runs on CPU regardless, so we
    never want seeding to abort the run.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_json_serializable(obj):
    """Recursively convert numpy types to plain Python for JSON serialisation."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj
