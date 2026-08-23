"""Conditional Density Estimation (CDE) wrappers for ScoringBench.

Thin adapters around the estimators in the ``cde`` package
(https://github.com/freelunchtheorem/Conditional_Density_Estimation, now a
PyTorch backend). Every estimator exposes a closed-form conditional density
``pdf(X, Y)``, so we derive the standard ``DistributionPrediction`` PMF
*analytically* by evaluating the density on a fixed, shared ``y``-grid and
normalizing — no sampling required. This keeps the external interface identical
to the other wrappers while exploiting each estimator's exact density.

Supported estimators (``estimator`` argument):

    "MDN"   — MixtureDensityNetwork (neural Gaussian mixture)
    "KMN"   — KernelMixtureNetwork (neural kernel mixture)
    "NF"    — NormalizingFlowEstimator (neural normalizing flow)
    "CKDE"  — ConditionalKernelDensityEstimation (kernel, statsmodels)
    "NKDE"  — NeighborKernelDensityEstimation (kernel, neighborhood weighting)
    "LSCDE" — LSConditionalDensityEstimation (least-squares CDE)
"""

from __future__ import annotations

import numpy as np

from .base import DistributionPrediction, ProbabilisticWrapper
from .sample_based import grid_density_to_distribution

_ESTIMATORS = {
    "MDN": "MixtureDensityNetwork",
    "KMN": "KernelMixtureNetwork",
    "NF": "NormalizingFlowEstimator",
    "CKDE": "ConditionalKernelDensityEstimation",
    "NKDE": "NeighborKernelDensityEstimation",
    "LSCDE": "LSConditionalDensityEstimation",
}

# Canonical default hyperparameters per estimator. Single source of truth shared
# by the benchmark's MODELS registration and the integration tests, so the
# per-estimator settings are defined in exactly one place. Only estimators that
# pass tests/wrapper/test_wrapper_integration.py are kept here; KMN / CKDE /
# LSCDE were dropped because their conditional density is badly mis-dispersed on
# the calibration DGP (interval-width / coverage checks fail). NKDE was dropped
# for being far too slow at benchmark scale (cross-validated bandwidth costs
# ~250 s/fold vs. ~15-25 s for MDN / NF).
CDE_PRESETS = {
    "MDN": dict(n_training_epochs=300, n_centers=10, hidden_sizes=(32, 32)),
    "NF": dict(n_training_epochs=300, n_flows=6, hidden_sizes=(32, 32)),
}


class CDEWrapper(ProbabilisticWrapper):
    """Conditional density estimator with an analytic grid-derived PMF.

    Parameters
    ----------
    estimator : str
        Which ``cde`` estimator to use. One of ``"MDN"``, ``"KMN"``, ``"NF"``,
        ``"CKDE"``, ``"NKDE"``, ``"LSCDE"``.
    n_grid : int
        Number of points on the shared ``y``-grid used to discretize the
        analytic conditional density into the ScoringBench PMF.
    grid_pad : float
        Fraction of the training ``y`` range to extend the grid beyond
        ``[y_min, y_max]`` on each side (captures tail mass).
    eval_chunk : int
        Number of test rows whose (row x grid) density block is evaluated per
        ``pdf`` call, bounding peak memory.
    random_seed : int
        Seed forwarded to the estimator.
    est_params : dict, optional
        Extra keyword arguments forwarded to the underlying estimator
        constructor (e.g. ``n_centers``, ``n_training_epochs``, ``bandwidth``).
        When ``None`` (default), the estimator's entry in ``CDE_PRESETS`` is
        used.
    """

    def __init__(
        self,
        estimator: str = "MDN",
        n_grid: int = 200,
        grid_pad: float = 0.15,
        eval_chunk: int = 256,
        random_seed: int = 0,
        est_params: dict | None = None,
    ):
        if estimator not in _ESTIMATORS:
            raise ValueError(
                f"Unsupported CDE estimator {estimator!r}; "
                f"choose from {sorted(_ESTIMATORS)}"
            )
        self.estimator = estimator
        self.n_grid = int(n_grid)
        self.grid_pad = float(grid_pad)
        self.eval_chunk = int(eval_chunk)
        self.random_seed = random_seed
        self.est_params = dict(CDE_PRESETS.get(estimator, {})) if est_params is None else est_params

        self._model = None
        self._ndim_x: int = 1
        self._y_range: tuple[float, float] = (0.0, 1.0)

    def _build_model(self, ndim_x: int):
        try:
            import cde.density_estimator as de
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "Failed to import cde. Install cde (and progressbar2) to use "
                "this wrapper."
            ) from exc

        cls = getattr(de, _ESTIMATORS[self.estimator])
        self._model = cls(
            name=f"sb_{self.estimator}",
            ndim_x=ndim_x,
            ndim_y=1,
            random_seed=self.random_seed,
            **self.est_params,
        )

    @staticmethod
    def _sanitize_X(X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return np.nan_to_num(X, nan=0.0, posinf=1e7, neginf=-1e7)

    def fit(self, X, y) -> "CDEWrapper":
        X = self._sanitize_X(X)
        y = np.asarray(y, dtype=np.float32).reshape(-1)

        valid = np.isfinite(y)
        X, y = X[valid], y[valid]
        if len(y) == 0:
            raise ValueError("No valid (finite) training samples after sanitization")

        self._ndim_x = X.shape[1]
        self._y_range = (float(y.min()), float(y.max()))

        self._build_model(self._ndim_x)
        self._model.fit(X, y.reshape(-1, 1))
        return self

    def _grid(self) -> np.ndarray:
        lo, hi = self._y_range
        span = hi - lo
        if not np.isfinite(span) or span <= 0:
            span = max(abs(hi), 1.0)
        pad = self.grid_pad * span
        return np.linspace(lo - pad, hi + pad, self.n_grid).astype(np.float32)

    def _density_on_grid(self, X: np.ndarray, grid: np.ndarray) -> np.ndarray:
        """Evaluate pdf(X, grid) -> (n, G), chunked over rows."""
        n = X.shape[0]
        G = grid.shape[0]
        out = np.empty((n, G), dtype=np.float64)
        for start in range(0, n, self.eval_chunk):
            stop = min(start + self.eval_chunk, n)
            Xc = X[start:stop]
            m = Xc.shape[0]
            X_rep = np.repeat(Xc, G, axis=0)
            Y_rep = np.tile(grid, m).reshape(-1, 1)
            p = np.asarray(self._model.pdf(X_rep, Y_rep), dtype=np.float64)
            out[start:stop] = p.reshape(m, G)
        return out

    def predict_distribution(self, X) -> DistributionPrediction:
        X = self._sanitize_X(X)
        grid = self._grid()
        dens = self._density_on_grid(X, grid)  # (n, G)
        return grid_density_to_distribution(grid, dens)

    def predict(self, X) -> np.ndarray:
        return self.predict_distribution(X).mean
