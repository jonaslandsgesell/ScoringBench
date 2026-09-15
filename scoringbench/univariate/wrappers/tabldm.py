"""Xiaomi-TabLDM wrapper for ScoringBench.

Xiaomi-TabLDM is a tabular *large data* foundation model for classification and
regression via in-context learning (ICL): a single pretrained Transformer makes
predictions in one forward pass given the labelled training rows as context, with
no task-specific weight updates. Its regression head predicts a set of quantiles
per test row, which the library wraps into a proper ``QuantileDistribution``
(monotone spline + exponential/GPD tail extrapolation) exposing ``icdf``/``cdf``/
``sample``/``mean``. Requesting ``output_type="quantiles"`` with explicit
``alphas`` therefore returns the full *conditional* predictive distribution — not
just an averaged point estimate — which we discretize into a
``DistributionPrediction`` exactly like the other multi-quantile heads (TabICL,
CatBoost, NGBoost, pytabkit).

Paper:  "Xiaomi-TabLDM: A Tabular Foundation Model" (Wang et al., 2026),
        arXiv:2609.03880.
Code:   https://github.com/xiaomi-research/xiaomi-tabldm
Model:  https://huggingface.co/occams/Xiaomi-TabLDM (checkpoint auto-downloaded)
License: Apache-2.0.

Installation (either works — the wrapper prefers a local checkout if present):

    # Option A — pip (recommended; inference-only, checkpoint from HF Hub):
    pip install "Xiaomi-TabLDM @ git+https://github.com/xiaomi-research/xiaomi-tabldm.git"

    # Option B — local checkout under additional_models/ (for offline / editing):
    git clone https://github.com/xiaomi-research/xiaomi-tabldm.git \
        additional_models/xiaomi-tabldm
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .base import DistributionPrediction, ProbabilisticWrapper

# Prefer a local checkout of `xiaomi-tabldm` when present in the workspace so the
# benchmark can run offline or against an edited source tree, mirroring the
# TabICL wrapper's local-first import strategy. This file is
# <repo>/scoringbench/univariate/wrappers/tabldm.py, so parents[3] is the repo
# root. The upstream package importable name is ``tabldm`` regardless of whether
# it was pip-installed or cloned.
_repo_root = Path(__file__).resolve().parents[3]
for _candidate in (
    _repo_root / "additional_models" / "xiaomi-tabldm",
    _repo_root / "additional_models" / "xiaomi-tabldm" / "src",
):
    if (_candidate / "tabldm").is_dir() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))


class TabLDMWrapper(ProbabilisticWrapper):
    """Wraps ``tabldm.TabLDMRegressor`` (Xiaomi-TabLDM v0.1).

    ``predict`` returns the distribution mean (``output_type="mean"``).
    ``predict_distribution`` requests a dense grid of quantiles
    (``output_type="quantiles"``, ``alphas=...``) and converts the per-sample
    quantile matrix into a piecewise-uniform ``DistributionPrediction`` via the
    shared multi-quantile constructor, so every quantile model is discretized
    identically.

    Notes
    -----
    * Distributional output requires the plain in-context path
      (``enhance_candidates=False``, the library default). The enhanced
      NNLS-ensembling path collapses to ``output_type="mean"`` only, so leave
      ``enhance_candidates`` at its default when distributional metrics matter.
    * The checkpoint is downloaded from the Hugging Face Hub on first use unless a
      local ``model_path`` is given.
    """

    def __init__(self, *, n_quantiles: int = 200, **kwargs):
        from tabldm import TabLDMRegressor

        # Dense, evenly spaced interior quantile levels in (0, 1); the library
        # extrapolates the outer tails analytically, so we stay away from the
        # exact 0/1 endpoints.
        self._ALPHAS = np.linspace(0.005, 0.995, int(n_quantiles)).tolist()
        self._model = TabLDMRegressor(**kwargs)

    def fit(self, X, y) -> "TabLDMWrapper":
        self._set_train_range(y)
        self._model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        X_arr = np.asarray(X.values if hasattr(X, "values") else X)
        return np.asarray(self._model.predict(X_arr, output_type="mean")).reshape(-1)

    def predict_distribution(self, X) -> DistributionPrediction:
        X_arr = np.asarray(X.values if hasattr(X, "values") else X)
        raw_q = self._model.predict(X_arr, output_type="quantiles", alphas=self._ALPHAS)

        # Robustly coerce to a (n_samples, n_alphas) float array. The library may
        # return the array directly or a {"quantiles": array} dict.
        if isinstance(raw_q, dict):
            q_arr = raw_q.get("quantiles", next(iter(raw_q.values())))
        else:
            q_arr = raw_q

        if isinstance(q_arr, list):
            q = np.vstack([np.asarray(r, dtype=float).ravel() for r in q_arr])
        else:
            q = np.asarray(q_arr, dtype=float)

        if q.ndim == 1:
            q = q[np.newaxis, :]
        # Orient to (n_samples, n_alphas) if it came back transposed.
        if q.shape[1] != len(self._ALPHAS) and q.shape[0] == len(self._ALPHAS):
            q = q.T

        # Enforce monotonicity defensively (the library already fixes crossing).
        q = np.sort(q, axis=1)

        alphas = np.asarray(self._ALPHAS, dtype=float)
        return DistributionPrediction.from_multi_quantile(
            q, alphas, train_range=self._y_train_range
        )
