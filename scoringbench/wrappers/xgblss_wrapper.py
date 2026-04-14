"""XGBoost LSS (Location-Scale-Shape) wrapper for ScoringBench."""

from __future__ import annotations

import numpy as np
import xgboost as xgb

from .base import DistributionPrediction, ProbabilisticWrapper


class XGBLSSWrapper(ProbabilisticWrapper):
    """XGBoost LSS regressor for ScoringBench.

    Trains XGBoost LSS to predict the full distribution parameters (location,
    scale, shape) and generates quantile predictions. Uses a grid of equally-spaced
    quantiles (default 100 quantiles) to construct the distribution.

    Args:
        n_quantiles: Number of equally-spaced quantile levels. Defaults to 100.
        num_boost_round: Number of boosting iterations. Defaults to 100.
        distribution: The distribution family to use ('Gaussian', 'StudentT', etc.).
            Defaults to 'Gaussian'.
        xgblss_params: Extra keyword arguments forwarded to XGBoostLSS training params.
    """

    def __init__(
        self,
        n_quantiles: int = 100,
        num_boost_round: int = 100,
        distribution: str = "Gaussian",
        xgblss_params: dict | None = None,
    ):
        self.n_quantiles = n_quantiles
        self.num_boost_round = num_boost_round
        self.distribution = distribution
        self.xgblss_params = xgblss_params or {}

        # Generate quantile percentiles using linspace
        self._alphas = np.linspace(1 / (n_quantiles + 1), n_quantiles / (n_quantiles + 1), n_quantiles)
        
        self._model = None
        self._y_range: tuple[float, float] = (0.0, 1.0)

    def _get_distribution(self):
        """Load and return the specified distribution class."""
        try:
            if self.distribution == "Gaussian":
                from xgboostlss.distributions.Gaussian import Gaussian
                return Gaussian()
            elif self.distribution == "StudentT":
                from xgboostlss.distributions.StudentT import StudentT
                return StudentT()
            else:
                # Try generic import fallback
                module_name = f"xgboostlss.distributions.{self.distribution}"
                dist_module = __import__(module_name, fromlist=[self.distribution])
                dist_class = getattr(dist_module, self.distribution)
                return dist_class()
        except ImportError as exc:
            raise ImportError(
                f"Failed to import distribution '{self.distribution}' from xgboostlss. "
                f"Install xgboostlss to use this wrapper."
            ) from exc

    def fit(self, X, y) -> "XGBLSSWrapper":
        try:
            from xgboostlss.model import XGBoostLSS
        except ImportError as exc:
            raise ImportError(
                "Failed to import xgboostlss. Install xgboostlss to use this wrapper."
            ) from exc

        y = np.asarray(y, dtype=float)
        self._y_range = (float(y.min()), float(y.max()))

        # Get distribution
        dist = self._get_distribution()

        # Create model
        self._model = XGBoostLSS(dist)

        # Prepare data
        dtrain = xgb.DMatrix(X, label=y)

        # Set up params
        params = {
            "eta": 0.1,
            "max_depth": 3,
            **self.xgblss_params,
        }

        # Train the model
        self._model.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=self.num_boost_round,
            verbose_eval=False,
        )
        return self

    def predict(self, X) -> np.ndarray:
        """Point prediction: median of quantile predictions."""
        q = self._get_quantiles(X)
        return np.median(q, axis=1)

    def _get_quantiles(self, X) -> np.ndarray:
        """Get quantile predictions from the LSS model.
        
        Returns:
            Array of shape (n_samples, n_quantiles) with quantile predictions.
        """
        # Prepare data
        dtest = xgb.DMatrix(X)
        
        # Get quantile predictions using all alphas
        # predict() returns a pandas DataFrame with quantile columns
        q_df = self._model.predict(
            dtest,
            pred_type="quantiles",
            quantiles=self._alphas.tolist(),
        )
        
        # Convert to numpy array (n_samples, n_quantiles)
        q = np.asarray(q_df, dtype=float)
        
        return q

    def predict_distribution(self, X) -> DistributionPrediction:
        """Predict the full distribution using quantiles."""
        
        # Get quantile predictions
        q = self._get_quantiles(X)

        # Finite-value protection
        if not np.all(np.isfinite(q)):
            q = np.nan_to_num(
                q,
                nan=self._y_range[0],
                posinf=self._y_range[1],
                neginf=self._y_range[0],
            )

        # Enforce strict monotonicity via sorting
        q = np.sort(q, axis=1)

        n_samples = q.shape[0]

        # Construct per-sample bin edges: extend slightly beyond the outermost quantiles
        left_w = np.maximum(q[:, 1] - q[:, 0], 1e-7)
        right_w = np.maximum(q[:, -1] - q[:, -2], 1e-7)

        # bin_edges shape: (n_samples, n_quantiles + 1)
        bin_edges = np.concatenate(
            [(q[:, 0] - left_w)[:, None], q, (q[:, -1] + right_w)[:, None]],
            axis=1,
        )

        # Mass per bin: alpha_0, diff(alphas), 1 - alpha_last
        masses = np.concatenate(
            [[self._alphas[0]], np.diff(self._alphas), [1.0 - self._alphas[-1]]]
        )
        probas = np.broadcast_to(masses[None, :], (n_samples, len(masses))).copy()

        bin_midpoints = (bin_edges[:, :-1] + bin_edges[:, 1:]) / 2
        mean = np.sum(probas * bin_midpoints, axis=-1)

        return DistributionPrediction(
            probas=probas,
            bin_edges=bin_edges,
            bin_midpoints=bin_midpoints,
            mean=mean,
        )
