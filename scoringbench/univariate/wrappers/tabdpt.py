"""TabDPT (Layer6) univariate regression wrapper for ScoringBench.

TabDPT v1.3 exposes a *probabilistic* regression head via
``TabDPTRegressor.predict(X, output_type="full")``, which returns a
``FullPrediction`` dict with:

    - ``logits``  : torch.Tensor of shape ``(n_test, num_bars)`` — unnormalised
                    log-probabilities over the bins.
    - ``borders`` : torch.Tensor of shape ``(num_bars + 1,)`` — strictly
                    increasing bin edges in *raw target space*.

This is exactly a piecewise-uniform histogram (a "bar distribution", the same
family as TabPFN's native output), so it maps directly onto the grid-native
path of :class:`DistributionPrediction` — no quantile inversion or sampling
round-trip is required, and the model's own resolution is preserved.

See the released API:
https://github.com/layer6ai-labs/TabDPT-inference/pull/75
"""

from __future__ import annotations

import numpy as np

from .base import DistributionPrediction, ProbabilisticWrapper


class TabDPTWrapper(ProbabilisticWrapper):
    """Wraps ``tabdpt.TabDPTRegressor`` with a DistributionPrediction interface.

    The predictive distribution is read directly from TabDPT's native bar
    distribution (``logits`` + ``borders``) via ``output_type="full"``, so the
    histogram is grid-native (its own resolution is authoritative).

    Parameters
    ----------
    device : str | None
        Torch device (``"cuda"`` / ``"cpu"``).  ``None`` auto-selects CUDA when
        available.
    n_ensembles : int
        Number of feature-permutation ensembles used at inference time
        (forwarded to ``predict``).  TabDPT's default is 8.
    context_size : int | None
        Optional retrieval/context size override forwarded to ``predict``.
    seed : int | None
        Seed for the inference-time feature permutation ensembling.
    **kwargs
        Extra keyword arguments forwarded to ``TabDPTRegressor(...)``
        (e.g. ``normalizer``, ``use_flash``, ``compile``).
    """

    def __init__(self, device=None, n_ensembles: int = 8, context_size=None,
                 seed: int | None = None, **kwargs):
        import torch
        from tabdpt import TabDPTRegressor

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        self._n_ensembles = int(n_ensembles)
        self._context_size = context_size
        self._seed = seed
        # TabDPT compiles its transformer by default; on CPU this is slow and can
        # fail, so only enable compilation when explicitly requested.
        kwargs.setdefault("compile", device != "cpu")
        self._model = TabDPTRegressor(device=self._device, **kwargs)

    def fit(self, X, y) -> "TabDPTWrapper":
        self._set_train_range(y)
        X = np.asarray(X.values if hasattr(X, "values") else X, dtype=np.float64)
        y = np.asarray(y.values if hasattr(y, "values") else y, dtype=np.float64).reshape(-1)
        self._model.fit(X, y)
        return self

    def _predict_kwargs(self) -> dict:
        kw = {"n_ensembles": self._n_ensembles}
        if self._context_size is not None:
            kw["context_size"] = self._context_size
        if self._seed is not None:
            kw["seed"] = self._seed
        return kw

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X.values if hasattr(X, "values") else X, dtype=np.float64)
        return np.asarray(self._model.predict(X, **self._predict_kwargs())).reshape(-1)

    def predict_distribution(self, X) -> DistributionPrediction:
        import torch

        X = np.asarray(X.values if hasattr(X, "values") else X, dtype=np.float64)
        with torch.no_grad():
            full = self._model.predict(X, output_type="full", **self._predict_kwargs())

        logits = full["logits"]
        borders = full["borders"]
        if not isinstance(logits, torch.Tensor):
            logits = torch.as_tensor(logits)
        if not isinstance(borders, torch.Tensor):
            borders = torch.as_tensor(borders)

        bin_edges = borders.detach().cpu().numpy().astype(np.float64)      # (n_bins+1,)
        bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2.0             # (n_bins,)

        probas = torch.softmax(logits.float(), dim=-1).detach().cpu().numpy()  # (n, n_bins)
        mean = (probas * bin_midpoints[None, :]).sum(axis=-1)                  # (n,)

        return DistributionPrediction(
            probas=probas,
            bin_edges=bin_edges,
            bin_midpoints=bin_midpoints,
            mean=mean,
            train_range=self._y_train_range,
            is_grid_native=True,
        )
