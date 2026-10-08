"""CREPES Conformal Regressor wrapper for ScoringBench."""

from __future__ import annotations

from numbers import Integral

import numpy as np
from sklearn.model_selection import train_test_split


from .base import DistributionPrediction, ProbabilisticWrapper


class CrepesWrapper(ProbabilisticWrapper):
    """CREPES Conformal Regressor for ScoringBench.

    Fits a base regressor on proper-training data and calibrates a conformal
    predictive system on held-out data. Each returned CPD threshold carries
    equal probability mass; repeated thresholds combine into a single atom.
    CDF-based scores and coverage use this distribution directly. Density-based
    scores use ScoringBench's shared resampling procedure.

    Args:
        base_model: An sklearn-compatible regressor to use as the base model
            (e.g., RandomForestRegressor, GradientBoostingRegressor).
        calibration_split: Fraction of training data to reserve for calibration.
            Defaults to 0.2 (20%).
        random_state: Seed for the split and Mondrian tie-breaking. Defaults to 42.
        use_difficulty_estimator: If True, fits a DifficultyEstimator on proper-training
            data to estimate local label dispersion and normalize residuals.
            Defaults to True.
        use_mondrian_categorizer: If True, fits a MondrianCategorizer on proper-training
            data using the DifficultyEstimator to create non-overlapping categories.
            Each category uses its own calibration residuals.
            Defaults to False.
        mondrian_no_bins: Number of bins for the MondrianCategorizer. Defaults to 20.
    """

    def __init__(
        self,
        base_model,
        calibration_split: float = 0.2,
        random_state: int = 42,
        use_difficulty_estimator: bool = True,
        use_mondrian_categorizer: bool = False,
        mondrian_no_bins: int = 20,
    ):
        if not 0 < calibration_split < 1:
            raise ValueError("calibration_split must be strictly between 0 and 1.")
        if not isinstance(mondrian_no_bins, Integral) or mondrian_no_bins < 1:
            raise ValueError("mondrian_no_bins must be a positive integer.")
        if use_mondrian_categorizer and not use_difficulty_estimator:
            raise ValueError("Mondrian categorization requires the difficulty estimator.")
        self.base_model = base_model
        self.calibration_split = calibration_split
        self.random_state = random_state
        self.use_difficulty_estimator = use_difficulty_estimator
        self.use_mondrian_categorizer = use_mondrian_categorizer
        self.mondrian_no_bins = mondrian_no_bins

        self._wrapped_model = None
        self._difficulty_estimator = None
        self._mondrian_categorizer = None

    def fit(self, X, y) -> "CrepesWrapper":
        """Fit the regressor and optional difficulty/Mondrian components on the
        proper-training split, then calibrate the CPS on the held-out split.

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

        self._wrapped_model = None
        self._difficulty_estimator = None
        self._mondrian_categorizer = None
        y = np.asarray(y, dtype=float)
        self._set_train_range(y)

        X_train, X_cal, y_train, y_cal = train_test_split(
            X, y,
            test_size=self.calibration_split,
            random_state=self.random_state,
        )

        wrapped_model = WrapRegressor(self.base_model)
        wrapped_model.fit(X_train, y_train)

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
            # Calibration labels must not influence the nonconformity function.
            de.fit(X_train, y=y_train, k=min(25, len(X_train)))

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
            # CREPES uses NumPy's global RNG for bin-boundary tie-breaking.
            state = np.random.get_state()
            try:
                if self.random_state is not None:
                    np.random.seed(self.random_state)
                mc.fit(X_train, de=de, no_bins=int(self.mondrian_no_bins))
            finally:
                np.random.set_state(state)

        wrapped_model.calibrate(
            X_cal, y_cal, cps=True, de=de, mc=mc, seed=self.random_state
        )
        self._wrapped_model = wrapped_model
        self._difficulty_estimator = de
        self._mondrian_categorizer = mc

        return self

    def predict_distribution(self, X) -> DistributionPrediction:
        """Return the full CPD as an equally weighted discrete distribution.

        Args:
            X: Features of shape (n_samples, n_features).

        Returns:
            DistributionPrediction preserving all CPD thresholds and their masses.
        """
        if self._wrapped_model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        cpds = self._wrapped_model.predict_cpds(X)
        if not isinstance(cpds, (np.ndarray, list, tuple)):
            raise ValueError("CREPES must return one CPD vector per observation.")
        if (isinstance(cpds, np.ndarray) and cpds.ndim == 1
                and cpds.dtype != object and len(X) == 1):
            cpds = cpds[np.newaxis, :]
        if len(cpds) != len(X):
            raise ValueError(f"CREPES returned {len(cpds)} CPDs for {len(X)} observations.")
        atoms = []
        for i, row in enumerate(cpds):
            try:
                values = np.asarray(row, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid nonnumeric CREPES CPD for observation {i}.") from exc
            if values.ndim != 1 or not values.size or not np.isfinite(values).all():
                raise ValueError(
                    f"Invalid CREPES CPD for observation {i}: expected a nonempty "
                    "finite vector. Its Mondrian group may have no calibration samples."
                )
            support, counts = np.unique(values, return_counts=True)
            atoms.append((support, counts / values.size))

        n_edges = 2 * max(len(support) for support, _ in atoms)
        edges = np.empty((len(atoms), n_edges), dtype=float)
        probas = np.zeros((len(atoms), n_edges - 1), dtype=float)
        for i, (support, mass) in enumerate(atoms):
            # Repeated edges encode atoms; intervening and padding bins have zero mass.
            stop = 2 * len(support)
            edges[i, :stop] = np.repeat(support, 2)
            edges[i, stop:] = support[-1]
            probas[i, :stop - 1:2] = mass
        return DistributionPrediction.from_histogram(
            edges, probas, train_range=self._y_train_range
        )

    def predict(self, X) -> np.ndarray:
        """Return the mean of each CPD."""
        return self.predict_distribution(X).mean