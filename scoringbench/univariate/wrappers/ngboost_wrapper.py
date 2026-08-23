"""NGBoost wrapper for ScoringBench.

NGBoost (https://github.com/stanfordmlgroup/ngboost) fits a gradient-boosted
*parametric* conditional distribution (e.g. a per-sample Normal). The predictive
density is available in closed form, so — unlike the genuinely sample-based
wrappers — we read the analytic distribution directly and convert it to the
standard ``DistributionPrediction`` PMF via its inverse CDF (``ppf``) on a fixed
quantile grid, mirroring the CatBoost / XGB-quantile wrappers.
"""

from __future__ import annotations

import numpy as np

from .base import DistributionPrediction, ProbabilisticWrapper
from .quantile_based import quantiles_to_distribution

_DISTNS = {
    "normal": "Normal",
    "lognormal": "LogNormal",
    "exponential": "Exponential",
}


class NGBoostWrapper(ProbabilisticWrapper):
    """NGBoost natural-gradient boosting with a parametric predictive density.

    Parameters
    ----------
    dist : str
        Parametric family for the conditional distribution. One of
        ``"normal"``, ``"lognormal"``, ``"exponential"``.
    n_estimators : int
        Number of boosting stages.
    learning_rate : float
        Boosting learning rate.
    n_quantiles : int
        Number of probability levels used to discretize the analytic predictive
        distribution into the ScoringBench PMF.
    ngb_params : dict, optional
        Extra keyword arguments forwarded to ``NGBRegressor``.
    """

    def __init__(
        self,
        dist: str = "normal",
        n_estimators: int = 500,
        learning_rate: float = 0.01,
        n_quantiles: int = 99,
        ngb_params: dict | None = None,
    ):
        self.dist = dist
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.n_quantiles = n_quantiles
        self.ngb_params = ngb_params or {}

        self._alphas = np.array(
            [k / (n_quantiles + 1) for k in range(1, n_quantiles + 1)]
        )
        self._model = None
        self._y_range: tuple[float, float] = (0.0, 1.0)

    def _build_model(self):
        try:
            from ngboost import NGBRegressor
            from ngboost.distns import Exponential, LogNormal, Normal
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "Failed to import ngboost. Install ngboost to use this wrapper."
            ) from exc

        registry = {"Normal": Normal, "LogNormal": LogNormal, "Exponential": Exponential}
        name = _DISTNS.get(self.dist.lower())
        if name is None:
            raise ValueError(f"Unsupported NGBoost dist: {self.dist!r}")

        self._model = NGBRegressor(
            Dist=registry[name],
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            verbose=False,
            **self.ngb_params,
        )

    @staticmethod
    def _sanitize_X(X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return np.nan_to_num(X, nan=0.0, posinf=1e7, neginf=-1e7)

    def fit(self, X, y) -> "NGBoostWrapper":
        if self._model is None:
            self._build_model()
        X = self._sanitize_X(X)
        y = np.asarray(y, dtype=np.float64).reshape(-1)

        valid = np.isfinite(y)
        X, y = X[valid], y[valid]
        if len(y) == 0:
            raise ValueError("No valid (finite) training samples after sanitization")

        self._y_range = (float(y.min()), float(y.max()))
        self._model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return np.asarray(self._model.predict(self._sanitize_X(X)), dtype=np.float64).reshape(-1)

    def predict_distribution(self, X) -> DistributionPrediction:
        dist = self._model.pred_dist(self._sanitize_X(X))
        frozen = dist.dist  # frozen scipy distribution with vector parameters

        # ppf broadcasts the (K, 1) grid against the (n,) parameters -> (K, n).
        q = frozen.ppf(self._alphas[:, None]).T  # (n, K)

        mean = None
        if hasattr(dist, "mean"):
            try:
                mean = np.asarray(dist.mean(), dtype=np.float64).reshape(-1)
            except Exception:
                mean = None

        return quantiles_to_distribution(
            q, self._alphas, mean=mean, y_range=self._y_range
        )
