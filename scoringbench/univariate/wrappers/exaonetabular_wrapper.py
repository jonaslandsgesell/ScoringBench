"""EXAONE-Tabular regressor wrapper for ScoringBench.

The released EXAONE-Tabular regression head predicts a **999-quantile
distribution** per row; the public ``EXAONETabularRegressor.predict`` collapses
that bank to a single point estimate (a trimmed mean, by default).  For
ScoringBench we want the *distribution*, so ``predict_distribution`` reproduces
the same fitted, weighted ensemble forward pass but keeps the raw quantile bank
instead of collapsing it, then feeds those quantiles through the shared
``quantiles_to_distribution`` mapping (identical to the TabICL / CatBoost /
XGB-quantile / NGBoost wrappers, so every quantile model is discretized the
same way).

Ensemble combination mirrors the library exactly:

* ``fit`` (in the underlying regressor) holds out a slice of the support and
  solves a non-negative least-squares (NNLS) problem for per-member weights,
  blended 75/25 with the uniform ``1/E`` prior.  Those weights live in
  ``regressor._fitted_state["member_weights"]`` (``None`` when the support is
  too small — under ``nnls_min_validation_rows``, ~10k rows — in which case the
  members are averaged uniformly).
* We apply that same convex weight vector to each member's *quantile column*.
  A convex combination of quantile functions is itself a valid quantile
  function, so this is coherent; sorting per row afterwards guards against any
  residual tau-crossing.

``predict`` delegates straight to the library point estimate (trimmed mean by
default; pass a ``manifest`` whose ``RegressionConfig`` sets
``point_estimate="median"`` to read the median quantile instead).

Notes
-----
* ``X`` must be numeric (encode categoricals to codes yourself; the built-in
  preprocessor mean-imputes ``NaN`` but does not encode raw strings).
* Targets must be finite; they are standardized against the fitted support and
  mapped back, so no manual scaling of ``y`` is required.
* Tables wider than 1024 columns are narrowed by univariate ``f_regression``.
"""

from __future__ import annotations

import numpy as np

from .base import DistributionPrediction, ProbabilisticWrapper
from .quantile_based import quantiles_to_distribution


class EXAONETabularWrapper(ProbabilisticWrapper):
    """Wraps ``EXAONETabularRegressor`` and exposes its 999-quantile head.

    Parameters
    ----------
    device : str
        Torch device passed to ``from_pretrained`` (e.g. ``"cuda:0"`` / ``"cpu"``).
    weights, revision, cache_dir, filename, manifest, ensemble_count,
    compute_dtype, seed, max_vram_bytes :
        Forwarded verbatim to ``EXAONETabularRegressor.from_pretrained``.  Pass a
        ``manifest`` whose ``RegressionConfig`` sets ``point_estimate="median"``
        or ``member_weighting="uniform"`` to change the reduction / weighting.
    """

    def __init__(self, *, device: str = "cuda:0", **from_pretrained_kwargs):
        from exaonetabular import EXAONETabularRegressor

        self._device = device
        self._model = EXAONETabularRegressor.from_pretrained(
            device=device, **from_pretrained_kwargs
        )

    # ------------------------------------------------------------------
    def fit(self, X, y) -> "EXAONETabularWrapper":
        X_arr = np.asarray(X.values if hasattr(X, "values") else X, dtype=np.float64)
        y_arr = np.asarray(y.values if hasattr(y, "values") else y, dtype=np.float64).reshape(-1)
        self._model.fit(X_arr, y_arr)
        return self

    def predict(self, X) -> np.ndarray:
        X_arr = np.asarray(X.values if hasattr(X, "values") else X, dtype=np.float64)
        return np.asarray(self._model.predict(X_arr), dtype=np.float64)

    # ------------------------------------------------------------------
    def predict_distribution(self, X) -> DistributionPrediction:
        import torch

        model = self._model
        state = model._state()  # raises if not fitted, same contract as predict
        X_arr = np.asarray(X.values if hasattr(X, "values") else X, dtype=np.float64)
        if X_arr.ndim != 2 or X_arr.shape[1] != state["n_features"]:
            raise ValueError("query features have an invalid shape")

        query = state["preprocessor"].transform(X_arr).values
        n_rows = query.shape[0]

        # Quantile levels the head predicts: linspace(1/(N+1), N/(N+1), N).
        levels = model.model.quantile_levels.detach().cpu().numpy().astype(np.float64)

        if n_rows == 0:
            empty = np.empty((0, levels.size), dtype=np.float64)
            return quantiles_to_distribution(empty, levels)

        # Reproduce the fitted, pooled ensemble forward pass, keeping the *raw*
        # quantile bank (members, rows, n_quantiles) instead of collapsing it to
        # a point like the library's `predict` does.
        support_x = torch.as_tensor(state["support_x"], dtype=torch.float32, device=model.device)
        support_y = torch.as_tensor(state["support_y"], dtype=torch.float32, device=model.device)
        query_x = torch.as_tensor(query, dtype=torch.float32, device=model.device)

        banks = torch.cat(
            [
                self._member_quantiles(support_x, support_y, query_x, n_svd=n_svd, seed=seed)
                for n_svd, seed in state["passes"]
            ],
            dim=0,
        )  # (members, rows, n_quantiles)

        # De-normalize onto the original target scale, exactly as `predict` does
        # for its point predictions.
        banks = banks.float() * state["scale"] + state["center"]

        # Combine members: fitted NNLS weights (convex) if present, else uniform.
        weights = state["member_weights"]
        if weights is None:
            combined = banks.mean(dim=0)  # (rows, n_quantiles)
        else:
            blend = torch.as_tensor(weights, dtype=banks.dtype, device=banks.device)
            combined = (blend.view(-1, 1, 1) * banks).sum(dim=0)

        q = combined.detach().cpu().numpy().astype(np.float64)
        # A convex mix of monotone quantile functions is monotone, but numerical
        # noise / tau-crossing can flip neighbours; sort defensively per row.
        q = np.sort(q, axis=1)

        return quantiles_to_distribution(q, levels)

    # ------------------------------------------------------------------
    def _member_quantiles(self, support_x, support_y, query_x, *, n_svd: int, seed: int):
        """One ensemble forward returning (members, rows, n_quantiles) quantiles.

        Mirrors ``EXAONETabularRegressor._member_points`` but skips the
        ``_collapse_members`` reduction, so the full per-quantile bank survives.
        The plan is rebuilt identically (same ``n_svd`` / ``seed`` carried in the
        fitted state) so member m denotes the same preprocessing rule the fitted
        weights were solved against.
        """
        import torch

        # Import the plan builders from the library's ensemble module so member
        # indexing matches the fit exactly.
        from exaonetabular.ensemble import EnsemblePlan, build_ensemble_inputs

        model = self._model
        plan = EnsemblePlan(
            members=model.manifest.runtime.ensemble_count,
            seed=seed,
            task="regression",
            n_svd=n_svd,
        )
        batch_xs, batch_y, batch_xq, _fitted_plan = build_ensemble_inputs(
            support_x, support_y, query_x, plan
        )
        model.model.eval()
        with torch.inference_mode():
            output = model._forward_chunked(batch_xs, batch_y, batch_xq)
        # output: (members, rows, n_quantiles) — sort quantiles per row/member
        # so a crossed bank cannot invert the CDF the mapping reads off it.
        return torch.sort(output.float(), dim=-1).values
