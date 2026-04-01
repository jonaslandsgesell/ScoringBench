"""Wrapper for PyTabKit RealMLP_TD_Regressor."""

from __future__ import annotations

import logging
import numpy as np
import re

from .base import DistributionPrediction, ProbabilisticWrapper


# Reduce noisy info logs from Lightning during benchmarking
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
logging.getLogger("lightning_fabric").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

class PytabkitRealMLPWrapper(ProbabilisticWrapper):
    """Wraps `pytabkit.RealMLP_TD_Regressor` for ScoringBench.

    This wrapper attempts to obtain predictive quantiles from the underlying
    model. It constructs per-sample bin edges from quantile particles and
    returns a DistributionPrediction compatible with the rest of the codebase.
    """

    def __init__(self, train_metric_name: str = "multi_pinball(0.25,0.5,0.75)",
                 val_metric_name: str = "multi_pinball(0.25,0.5,0.75)",
                 n_quantiles: int = 50,
                 quantile_min: float = 0.001,
                 quantile_max: float = 0.999,
                 **kwargs):
        self.n_quantiles = n_quantiles
        # Explicit 50 quantiles (linear spacing 0.01..0.99) to avoid
        # subtle differences between pytabkit versions and to make the
        # quantiles explicit for reproducibility.
        # NOTE: this literal matches 50 values from 0.01 to 0.99 step 0.02.
        if n_quantiles == 50:
            self._alphas = np.array([
                0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19,
                0.21, 0.23, 0.25, 0.27, 0.29, 0.31, 0.33, 0.35, 0.37, 0.39,
                0.41, 0.43, 0.45, 0.47, 0.49, 0.51, 0.53, 0.55, 0.57, 0.59,
                0.61, 0.63, 0.65, 0.67, 0.69, 0.71, 0.73, 0.75, 0.77, 0.79,
                0.81, 0.83, 0.85, 0.87, 0.89, 0.91, 0.93, 0.95, 0.97, 0.99,
            ])
        else:
            # Fallback to linear spacing for other requested sizes
            self._alphas = np.linspace(quantile_min, quantile_max, n_quantiles)
        self._model = None
        self._init_kwargs = dict(
            train_metric_name=train_metric_name,
            val_metric_name=val_metric_name,
            **kwargs,
        )

    def _build_model(self):
        # Import here so users without pytabkit can import the repo.
        try:
            from pytabkit import RealMLP_TD_Regressor
        except Exception as exc:  # pragma: no cover - import/runtime issues
            raise ImportError(
                "Failed to import pytabkit. Install pytabkit[models] to use this wrapper"
            ) from exc

        # Quiet down pytabkit's own logging to avoid noisy output.
        logging.getLogger('pytabkit').setLevel(logging.WARNING)

        # Instantiate the model. Let any runtime errors surface to the caller
        # instead of swallowing them.
        self._model = RealMLP_TD_Regressor(**self._init_kwargs)
        logger.debug("Built pytabkit RealMLP_TD_Regressor (init kwargs: %s)", self._init_kwargs)

    def fit(self, X, y) -> "PytabkitRealMLPWrapper":
        if self._model is None:
            self._build_model()
        # scikit-learn style fit — let exceptions propagate.
        self._model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        if self._model is None:
            self._build_model()
        # Default point prediction
        raw = np.asarray(self._model.predict(X))
        if raw.ndim == 2:
            # Model returned quantile predictions (n_samples, n_quantiles).
            # Derive a consistent point estimate via the distribution mean.
            return self.predict_distribution(X).mean
        return raw

    def predict_distribution(self, X) -> DistributionPrediction:
        if self._model is None:
            self._build_model()
        # Try a few common pytabkit quantile APIs. We avoid broad exception
        # swallowing so callers see failures during fit/predict.
        q = None
        if hasattr(self._model, "predict_quantiles"):
            # Preferred API: predict_quantiles(X, quantiles=...)
            q = self._model.predict_quantiles(X, quantiles=self._alphas)

        if q is None and hasattr(self._model, "predict"):
            # Some versions accept a `quantiles` kwarg; if that raises a
            # TypeError (signature mismatch) we fall back to plain `predict(X)`.
            try:
                q = self._model.predict(X, quantiles=self._alphas)
            except TypeError:
                q = self._model.predict(X)

        if q is None:
            # Let the absence of a quantile API be explicit to the caller.
            raise NotImplementedError(
                "Installed pytabkit does not expose a quantile prediction API "
                "compatible with this wrapper."
            )

        # Some pytabkit variants return (values, alphas)
        q_alphas = None
        if isinstance(q, (list, tuple)) and len(q) == 2:
            q_vals, q_alphas = q
            q = np.asarray(q_vals)
            q_alphas = np.asarray(q_alphas)
        else:
            q = np.asarray(q)

        # Ensure shape (n_samples, n_quantiles) with robust handling for
        # different return layouts (some versions return (n_q, n_samples)).
        n_samples = X.shape[0]
        if q.ndim == 1:
            # Single vector of quantiles: assume it corresponds to one sample
            # or a prototype for all samples. Tile to samples when sizes match.
            if q.size == self.n_quantiles:
                q = np.tile(q[None, :], (n_samples, 1))
            else:
                raise ValueError(
                    f"Unexpected 1D quantiles length {q.size}, expected {self.n_quantiles}"
                )
        elif q.ndim == 2:
            # Common shapes: (n_samples, n_quantiles) or (n_quantiles, n_samples)
            if q.shape[0] == n_samples and q.shape[1] == self.n_quantiles:
                pass
            elif q.shape[0] == self.n_quantiles and q.shape[1] == n_samples:
                q = q.T
            elif q.shape[0] == n_samples and q.shape[1] == 1:
                # single-quantile per sample: replicate across quantiles
                q = np.tile(q, (1, self.n_quantiles))
            else:
                # If the model returned a different number of quantiles
                # attempt to infer the returned alpha locations and
                # interpolate to the requested `self._alphas`.
                orig_q_count = q.shape[1]
                # Try to get alphas from explicit return or from metric strings
                if q_alphas is None:
                    # parse numbers from the metric string if present
                    def _parse_alphas_from_metric(s: str | None):
                        if not s:
                            return None
                        m = re.search(r"\(([^)]+)\)", s)
                        if not m:
                            return None
                        parts = [p.strip() for p in m.group(1).split(',') if p.strip()]
                        try:
                            return np.array([float(p) for p in parts])
                        except Exception:
                            return None

                    q_alphas = _parse_alphas_from_metric(self._init_kwargs.get('train_metric_name'))
                    if q_alphas is None:
                        q_alphas = _parse_alphas_from_metric(self._init_kwargs.get('val_metric_name'))

                if q_alphas is not None and q_alphas.size == orig_q_count:
                    # interpolate per-sample to requested alphas
                    # Interpolate returned quantiles to the requested alphas
                    # so downstream code always sees `self.n_quantiles`.
                    # ensure monotonic alphas
                    order = np.argsort(q_alphas)
                    q_alphas_sorted = q_alphas[order]
                    q_sorted = q[:, order]
                    # interpolate each sample
                    q = np.vstack([
                        np.interp(self._alphas, q_alphas_sorted, q_sorted[i, :])
                        for i in range(n_samples)
                    ])
                else:
                    # Last resort: try to reshape if elements match
                    try:
                        q = q.reshape(n_samples, self.n_quantiles)
                    except Exception:
                        raise ValueError(
                            f"Unexpected quantiles shape {q.shape}; cannot reshape to (n_samples={n_samples}, n_quantiles={self.n_quantiles})"
                        )

        # 1) Finite protection and monotonicity
        q = np.nan_to_num(q, nan=np.nanmin(q), posinf=np.nanmax(q), neginf=np.nanmin(q))
        q = np.sort(q, axis=1)

        # 2) Per-sample bin edges: extend slightly beyond endpoints
        left_w = np.maximum(q[:, 1] - q[:, 0], 1e-7)
        right_w = np.maximum(q[:, -1] - q[:, -2], 1e-7)
        bin_edges = np.concatenate(
            [(q[:, 0] - left_w)[:, None], q, (q[:, -1] + right_w)[:, None]],
            axis=1,
        )

        # 3) Masses per bin from alphas: [alpha0, diff(alpha), ..., 1-alpha_last]
        masses = np.concatenate([[self._alphas[0]], np.diff(self._alphas), [1.0 - self._alphas[-1]]])
        probas = np.broadcast_to(masses[None, :], (q.shape[0], len(masses))).copy()

        bin_midpoints = (bin_edges[:, :-1] + bin_edges[:, 1:]) / 2
        mean = np.sum(probas * bin_midpoints, axis=-1)

        return DistributionPrediction(
            probas=probas,
            bin_edges=bin_edges,
            bin_midpoints=bin_midpoints,
            mean=mean,
        )
