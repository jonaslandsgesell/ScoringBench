"""CREPES Conformal Regressor wrapper for ScoringBench."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


from .base import DistributionPrediction, ProbabilisticWrapper
from .quantile_based import quantiles_to_distribution


class CrepesWrapper(ProbabilisticWrapper):
    """CREPES Conformal Regressor for ScoringBench.

    Uses CREPES (Conformal REgression PrEdiction SetS) with a base model to generate
    conformal prediction intervals. Predictions are converted to quantile-based
    distributions using n_quantiles percentile levels.

    The wrapper automatically splits training data into proper training and
    calibration sets, then uses CREPES's Conformal Predictive System (CPS)
    to predict calibrated quantiles.

    Optionally supports DifficultyEstimator for more sophisticated conformal 
    prediction that adapts interval sizes based on instance difficulty.

    Optionally supports Mondrian Categorization to divide the feature space into
    non-overlapping categories based on difficulty, forming separate conformal
    regressors for each category. This leads to more uniformly distributed interval
    sizes across different difficulty regions.

    Args:
        base_model: An sklearn-compatible regressor to use as the base model
            (e.g., RandomForestRegressor, GradientBoostingRegressor).
        n_quantiles: Number of equally-spaced percentile levels in (0, 1).
            Defaults to 99 (levels 0.01, 0.02, …, 0.99).
        calibration_split: Fraction of training data to reserve for calibration.
            Defaults to 0.2 (20%).
        random_state: Random state for train/calibration split. Defaults to 42.
        use_difficulty_estimator: If True, fits a DifficultyEstimator on calibration
            data to estimate per-sample difficulty and normalize intervals.
            Defaults to True.
        use_mondrian_categorizer: If True, fits a MondrianCategorizer on calibration
            data using the DifficultyEstimator to create non-overlapping categories.
            Each category gets its own conformal regressor for more uniform intervals.
            Defaults to False.
        mondrian_no_bins: Number of bins for the MondrianCategorizer. Defaults to 20.
    """

    def __init__(
        self,
        base_model,
        n_quantiles: int = 99,
        calibration_split: float = 0.2,
        random_state: int = 42,
        use_difficulty_estimator: bool = True,
        use_mondrian_categorizer: bool = False,
        mondrian_no_bins: int = 20,
    ):
        self.base_model = base_model
        self.n_quantiles = n_quantiles
        self.calibration_split = calibration_split
        self.random_state = random_state
        self.use_difficulty_estimator = use_difficulty_estimator
        self.use_mondrian_categorizer = use_mondrian_categorizer
        self.mondrian_no_bins = mondrian_no_bins

        self._alphas = np.linspace(1 / (n_quantiles + 1), n_quantiles / (n_quantiles + 1), n_quantiles)
        # Use the user-requested 0.01..0.99 grid when n_quantiles == 99
        if n_quantiles == 99:
            self._alphas = np.array([q / 100 for q in range(1, 100)])

        self._wrapped_model = None
        self._difficulty_estimator = None
        self._mondrian_categorizer = None
        self._y_range: tuple[float, float] = (0.0, 1.0)

    def fit(self, X, y) -> "CrepesWrapper":
        """Fit the CREPES model.

        Automatically splits training data into training and calibration sets,
        fits the base model, and calibrates via CREPES. Optionally fits
        DifficultyEstimator for advanced conformal prediction.
        
        Optionally fits MondrianCategorizer for category-specific conformal prediction
        when use_mondrian_categorizer=True.

        Args:
            X: Training features of shape (n_samples, n_features).
            y: Training targets of shape (n_samples,).

        Returns:
            self
        """
        try:
            from crepes import WrapRegressor
        except ImportError as exc:
            raise ImportError(
                "Failed to import crepes. Install crepes to use this wrapper."
            ) from exc

        y = np.asarray(y, dtype=float)
        self._y_range = (float(y.min()), float(y.max()))

        # Split into training and calibration sets
        X_train, X_cal, y_train, y_cal = train_test_split(
            X, y,
            test_size=self.calibration_split,
            random_state=self.random_state,
        )

        # Wrap the base model
        self._wrapped_model = WrapRegressor(self.base_model)

        # Fit on training set
        self._wrapped_model.fit(X_train, y_train)

        # Optionally fit DifficultyEstimator
        de = None
        if self.use_difficulty_estimator:
            try:
                from crepes.extras import DifficultyEstimator
            except ImportError as exc:
                raise ImportError(
                    "Failed to import DifficultyEstimator from crepes.extras. "
                    "Install crepes with extras support."
                ) from exc
            
            de = DifficultyEstimator()
            # Adaptive split: use more data for DE in low data regimes to avoid
            # insufficient samples for KNN in crepes (n_neighbors default is 25)
            if len(X_cal) < 60:
                # Small calibration set: use 90% for DE fitting
                split_idx = int(len(X_cal) * 0.9)
            else:
                # Normal regime: use 50% for DE fitting (original behavior)
                split_idx = int(len(X_cal) * 0.5)
            de.fit(X_cal[:split_idx], y=y_cal[:split_idx])
            self._difficulty_estimator = de

        # Optionally fit Mondrian Categorizer using difficulty estimator
        mc = None
        if self.use_mondrian_categorizer and de is not None:
            try:
                from crepes.extras import MondrianCategorizer
            except ImportError as exc:
                raise ImportError(
                    "Failed to import MondrianCategorizer from crepes.extras. "
                    "Install crepes with extras support."
                ) from exc
            
            mc = MondrianCategorizer()
            mc.fit(X_cal, de=de, no_bins=self.mondrian_no_bins)
            self._mondrian_categorizer = mc

        # Calibrate with CPS (Conformal Predictive System)
        # Pass difficulty estimator and Mondrian categorizer if enabled
        calibrate_kwargs = {"cps": True}
        if de is not None:
            calibrate_kwargs["de"] = de
        if mc is not None:
            calibrate_kwargs["mc"] = mc
        
        self._wrapped_model.calibrate(X_cal, y_cal, **calibrate_kwargs)

        return self

    def predict_distribution(self, X) -> DistributionPrediction:
        """Predict quantile-based distributions via CREPES.

        Uses the calibrated CREPES model to get Conformal Predictive Distributions,
        then samples the desired quantile levels and converts to a DistributionPrediction.

        When DifficultyEstimator is enabled, the predicted intervals are automatically 
        adjusted by instance difficulty, ensuring more reliable calibration across 
        heterogeneous samples.

        Args:
            X: Features of shape (n_samples, n_features).

        Returns:
            DistributionPrediction with quantile-based bins and uniform masses.
        """
        if self._wrapped_model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Get Conformal Predictive Distributions: (n_samples, n_calibration_samples)
        cpds = self._wrapped_model.predict_cpds(X)
        
        # Robustly handle variable-length returns from Mondrian categorization
        # First try direct conversion; fall back to object array handling if shapes don't match
        cpds_list = None
        try:
            # Try to convert directly to float array
            cpds_float = np.asarray(cpds, dtype=float)
            cpds = cpds_float
        except (ValueError, TypeError):
            # Handle case where CPDs have variable lengths (e.g., from Mondrian categories)
            cpds = np.asarray(cpds, dtype=object)
            cpds_list = list(cpds)
            cpds = None

        # Handle single-sample edge case for regular (non-object) arrays
        if cpds is not None and cpds.ndim == 1:
            cpds = cpds[np.newaxis, :]
            cpds_list = None

        # Convert alphas from [0.01, 0.02, ..., 0.99] to [1, 2, ..., 99] percentiles
        target_percentiles = [a * 100 for a in self._alphas]

        if cpds is None:
            # Mondrian case: per-sample CPDs may have different lengths
            n_samples = len(cpds_list)
            q = np.empty((n_samples, len(self._alphas)), dtype=float)
            for i in range(n_samples):
                cpd_i = np.asarray(cpds_list[i], dtype=float).ravel()
                if len(cpd_i) == 0:
                    # Empty CPD (shouldn't happen but handle gracefully)
                    q[i] = np.nan
                else:
                    q[i] = np.percentile(cpd_i, q=target_percentiles)
        else:
            # Regular case: all CPDs have same shape
            # np.percentile(cpds, q, axis=1) returns (n_quantiles, n_samples)
            # We transpose to get (n_samples, n_quantiles)
            q = np.percentile(cpds, q=target_percentiles, axis=1).T  # (n_samples, n_quantiles)
        
        q = np.asarray(q, dtype=float)
        return quantiles_to_distribution(q, self._alphas, y_range=self._y_range)

    def predict(self, X) -> np.ndarray:
        """Point predictions using the CREPES model.

        Returns the expected value (mean) of the predicted Conformal 
        Predictive Distribution as the point estimate.

        Args:
            X: Features of shape (n_samples, n_features).

        Returns:
            Point predictions of shape (n_samples,).
        """
        # Call predict_distribution to get the distribution object
        dist = self.predict_distribution(X)
        
        # Return the mean calculated by the distribution prediction
        return dist.mean