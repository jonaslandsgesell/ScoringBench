"""FlexCode wrapper for ScoringBench.

FlexCode (https://github.com/lee-group-cmu/FlexCode) performs conditional
density estimation by regressing the coefficients of an orthonormal basis
expansion of ``p(y | x)`` with an arbitrary pluggable regressor (XGBoost,
random forest, ...). ``predict`` returns the conditional density evaluated on a
shared ``y``-grid, which maps directly onto the ScoringBench shared-grid PMF via
:func:`grid_density_to_distribution`.
"""

from __future__ import annotations

import numpy as np

from .base import DistributionPrediction, ProbabilisticWrapper
from .sample_based import grid_density_to_distribution

_REGRESSORS = {
    "xgboost": "XGBoost",
    "randomforest": "RandomForest",
    "nn": "NN",
    "lasso": "Lasso",
}

# Canonical default hyperparameters per regressor backend. Single source of
# truth shared by the benchmark's MODELS registration and the integration tests.
# max_basis=63: fewer basis functions leave the conditional density too wide
# (interval-width check fails); 63 is well inside the calibration tolerances.
FLEXCODE_PRESETS = {
    "xgboost": dict(max_basis=63, basis_system="cosine", regression_params={"max_depth": 3}),
    "randomforest": dict(max_basis=63, basis_system="cosine", regression_params={"n_estimators": 100}),
}


class FlexCodeWrapper(ProbabilisticWrapper):
    """FlexCode basis-expansion CDE with a grid-derived PMF.

    Parameters
    ----------
    regressor : str
        Backend regressor for the basis coefficients. One of ``"xgboost"``,
        ``"randomforest"``, ``"nn"``, ``"lasso"``.
    max_basis : int, optional
        Maximum number of basis functions in the expansion. Defaults to the
        regressor's :data:`FLEXCODE_PRESETS` entry.
    basis_system : str, optional
        Orthonormal basis family (e.g. ``"cosine"``). Defaults to the
        regressor's :data:`FLEXCODE_PRESETS` entry.
    n_grid : int
        Number of points on the shared ``y``-grid used to discretize the
        predicted conditional density into the ScoringBench PMF.
    grid_pad : float
        Fraction of the training ``y`` range to extend the density grid beyond
        ``[y_min, y_max]`` on each side.
    regression_params : dict, optional
        Extra keyword arguments forwarded to the backend regressor. Defaults to
        the regressor's :data:`FLEXCODE_PRESETS` entry.
    """

    def __init__(
        self,
        regressor: str = "xgboost",
        max_basis: int | None = None,
        basis_system: str | None = None,
        n_grid: int = 200,
        grid_pad: float = 0.15,
        regression_params: dict | None = None,
    ):
        if regressor not in _REGRESSORS:
            raise ValueError(
                f"Unsupported FlexCode regressor {regressor!r}; "
                f"choose from {sorted(_REGRESSORS)}"
            )
        preset = FLEXCODE_PRESETS.get(regressor, {})
        self.regressor = regressor
        self.max_basis = int(max_basis if max_basis is not None else preset.get("max_basis", 31))
        self.basis_system = basis_system if basis_system is not None else preset.get("basis_system", "cosine")
        self.n_grid = int(n_grid)
        self.grid_pad = float(grid_pad)
        if regression_params is not None:
            self.regression_params = regression_params
        else:
            self.regression_params = dict(preset.get("regression_params", {}))

        self._model = None
        self._y_range: tuple[float, float] = (0.0, 1.0)

    @staticmethod
    def _sanitize_X(X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return np.nan_to_num(X, nan=0.0, posinf=1e7, neginf=-1e7)

    def fit(self, X, y) -> "FlexCodeWrapper":
        try:
            import flexcode
            from flexcode import regression_models as rm
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "Failed to import flexcode. Install flexcode to use this wrapper."
            ) from exc

        X = self._sanitize_X(X)
        y = np.asarray(y, dtype=np.float64).reshape(-1)

        valid = np.isfinite(y)
        X, y = X[valid], y[valid]
        if len(y) == 0:
            raise ValueError("No valid (finite) training samples after sanitization")

        lo, hi = float(y.min()), float(y.max())
        span = hi - lo
        if not np.isfinite(span) or span <= 0:
            span = max(abs(hi), 1.0)
        pad = self.grid_pad * span
        self._y_range = (lo - pad, hi + pad)

        model_cls = getattr(rm, _REGRESSORS[self.regressor])
        self._model = flexcode.FlexCodeModel(
            model_cls,
            max_basis=self.max_basis,
            basis_system=self.basis_system,
            z_min=self._y_range[0],
            z_max=self._y_range[1],
            regression_params=self.regression_params,
        )
        self._model.fit(X, y.reshape(-1, 1))
        return self

    def predict_distribution(self, X) -> DistributionPrediction:
        X = self._sanitize_X(X)
        cdes, z_grid = self._model.predict(X, n_grid=self.n_grid)
        z_grid = np.asarray(z_grid, dtype=np.float64).reshape(-1)
        return grid_density_to_distribution(z_grid, np.asarray(cdes, dtype=np.float64))

    def predict(self, X) -> np.ndarray:
        return self.predict_distribution(X).mean
