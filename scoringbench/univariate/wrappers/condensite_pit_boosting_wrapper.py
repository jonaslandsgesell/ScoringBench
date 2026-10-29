"""condensité PIT boosting (LightGBM regressor) wrapper for ScoringBench.

A base condensité model estimates the conditional density ``p_0(y | x)``, with
CDF ``F_0``. Its cross-fitted probability integral transforms (PITs)
``U_i = F_{0,-k(i)}(y_i | x_i)``, each computed by the base model fitted
without the fold ``k(i)`` of observation ``i``, are uniform on ``[0, 1]`` if
``F_0`` is the true conditional CDF; otherwise their conditional density ``g``
satisfies ``p(y | x) = p_0(y | x) g(F_0(y | x) | x)``. A second condensité
model on ``[0, 1]`` estimates ``g`` on the same folds, and the prediction is

    p(y | x) = (1/K) Σ_k p_{0,-k}(y | x) [1 - λ + λ ĝ(F_{0,-k}(y | x) | x)],

with ``ĝ`` the average of the ``K`` out-of-fold correction models. The weight
``λ ∈ [0, 1]`` maximizes the out-of-fold log-likelihood
``Σ_i log(1 - λ + λ ĝ_{-k(i)}(U_i | x_i))``, so ``λ = 0`` (base model only)
when the correction does not generalize to held-out PITs. A kernel of fixed
bandwidth ``h`` on the PIT scale acts as a local bandwidth of about
``h / p_0(y | x)`` on the ``y`` scale: the second fit adapts its resolution to
the distribution found by the first.

The method is implemented by ``condensite.PITBoosting`` (condensité >= 0.1.1);
this wrapper supplies LightGBM regressors for both stages and maps the density
on the shared grid onto the ScoringBench PMF via
:func:`grid_density_to_distribution`.

Paper: A. G. Reisach, O. Collier, A. Luedtke, A. Chambaz (2025), "Transforming
Conditional Density Estimation Into a Single Nonparametric Regression Task",
https://arxiv.org/abs/2511.18530. Code: https://github.com/Scriddie/condensite.
License: BSD-3-Clause.
"""

from __future__ import annotations

import numpy as np
import torch

from .base import DistributionPrediction, ProbabilisticWrapper
from .density_based import grid_density_to_distribution

# Default hyperparameters. M*h = 1 for both stages (Mh); min_leaf_mass is
# explained in _LGBMPredictor.fit and u_max_bin in
# CondensitePITBoostingWrapper._predictor.
CONDENSITE_PIT_BOOSTING_PRESET = dict(
    base_h=[0.01],
    correction_h=[0.1],
    Mh=1.0,
    K=5,
    n_grid=400,
    n_grid_u=400,
    min_leaf_mass=0.2,
    u_max_bin=1023,
    regression_params={
        "num_leaves": 100,
        "max_bin": 255,
        "num_iterations": 100,
        "learning_rate": 0.1,
    },
)


class _LGBMPredictor:
    """LightGBM booster as a condensité predictor.

    Duck-types ``condensite.CondensitePredictor`` (``Condensite`` only calls
    ``device``/``to``/``fit``/``predict``), which keeps the condensité import
    lazy.
    """

    def __init__(self, params: dict, min_leaf_mass: float):
        self.params = params
        self.min_leaf_mass = min_leaf_mass
        self.booster = None

    def device(self):
        return "cpu"

    def to(self, device):
        return self

    def forward(self, features):
        return self.booster.predict(features.numpy())

    def predict(self, features):
        return torch.as_tensor(self.forward(features), dtype=torch.float32)

    def fit(self, dataset):
        import lightgbm as lgb

        # Assemble the n*M expanded rows by indexing; the per-item DataLoader
        # path does the same thing ~1000x slower.
        features = torch.cat([dataset.x[dataset.j_map], dataset.y_aux.reshape(-1, 1)], dim=1)
        params = dict(self.params)
        # A leaf spanning k observations and a u-interval of width w holds about
        # k*M*w expanded rows, so min_data_in_leaf = min_leaf_mass*M enforces
        # k*w >= min_leaf_mass. M is that of the bandwidth being fitted, so each
        # candidate h is regularized alike. As for a histogram bin, the number of
        # observations whose ỹ falls into the leaf's u-interval is about
        # k*w*p̃(ỹ | x).
        params.setdefault("min_data_in_leaf", max(1, round(self.min_leaf_mass * dataset.M)))
        num_boost_round = params.pop("num_iterations")
        train = lgb.Dataset(features.numpy(), label=dataset.targets.reshape(-1).numpy(),
                            params={"max_bin_by_feature": params.pop("max_bin_by_feature")})
        self.booster = lgb.train(params, train, num_boost_round=num_boost_round)
        return None  # no train MSE: it would cost a full predict on n*M rows


class CondensitePITBoostingWrapper(ProbabilisticWrapper):
    """condensité corrected by a second condensité fit in PIT space, with a grid-derived PMF.

    Parameters
    ----------
    base_h, correction_h : list of float, optional
        Candidate kernel bandwidths of the base model (on the min-max-scaled
        target) and of the correction (on the PIT scale ``[0, 1]``). With
        several candidates, condensité selects the one with the lowest
        validation ISE on a fraction ``val_frac`` of the rows and refits on all
        rows. Default to the :data:`CONDENSITE_PIT_BOOSTING_PRESET` values.
    Mh : float, optional
        ``M * h``: bandwidth ``h`` uses ``M = round(Mh / h)`` auxiliary points
        per observation.
    K : int, optional
        Number of folds for the cross-fitted PITs and the correction models.
    n_grid : int, optional
        Number of points on the shared ``y``-grid on which the density is
        evaluated, both for the PITs and for prediction.
    n_grid_u : int, optional
        Number of points on ``[0, 1]`` on which the correction densities are
        evaluated.
    min_leaf_mass : float, optional
        Lower bound on ``k * w`` for a leaf spanning ``k`` observations and a
        ``u``-interval of width ``w``, in both stages; sets
        ``min_data_in_leaf = min_leaf_mass * M`` for each bandwidth's ``M``.
    u_max_bin : int, optional
        Number of histogram bins for the auxiliary feature ``u`` (the ``x``
        features use ``regression_params["max_bin"]``).
    regression_params : dict, optional
        LightGBM parameters of both stages; they override the wrapper's own
        settings.
    val_frac : float
        Validation fraction for the selection among several bandwidths; unused
        with a single candidate.
    prediction_batch_size : int, optional
        Number of expanded ``(x, u)`` rows per prediction batch (bounds memory).
    random_seed : int
        Seed for the folds, condensité's auxiliary draws and LightGBM.
    """

    def __init__(
        self,
        base_h: list[float] | None = None,
        correction_h: list[float] | None = None,
        Mh: float | None = None,
        K: int | None = None,
        n_grid: int | None = None,
        n_grid_u: int | None = None,
        min_leaf_mass: float | None = None,
        u_max_bin: int | None = None,
        regression_params: dict | None = None,
        val_frac: float = 0.2,
        prediction_batch_size: int | None = 100_000,
        random_seed: int = 0,
    ):
        preset = CONDENSITE_PIT_BOOSTING_PRESET
        self.base_h = [float(h) for h in (base_h if base_h is not None else preset["base_h"])]
        self.correction_h = [float(h) for h in (correction_h if correction_h is not None else preset["correction_h"])]
        self.Mh = float(Mh if Mh is not None else preset["Mh"])
        self.K = int(K if K is not None else preset["K"])
        self.n_grid = int(n_grid if n_grid is not None else preset["n_grid"])
        self.n_grid_u = int(n_grid_u if n_grid_u is not None else preset["n_grid_u"])
        self.min_leaf_mass = float(min_leaf_mass if min_leaf_mass is not None else preset["min_leaf_mass"])
        self.u_max_bin = int(u_max_bin if u_max_bin is not None else preset["u_max_bin"])
        if regression_params is not None:
            self.regression_params = regression_params
        else:
            self.regression_params = dict(preset["regression_params"])
        self.val_frac = float(val_frac)
        self.prediction_batch_size = prediction_batch_size
        self.random_seed = int(random_seed)

        self._model = None
        self.lambda_ = None

    @staticmethod
    def _sanitize_X(X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return np.nan_to_num(X, nan=0.0, posinf=1e7, neginf=-1e7)

    def _predictor(self, n_features: int) -> _LGBMPredictor:
        """LightGBM predictor; its min_data_in_leaf is set at fit time (see _LGBMPredictor.fit).

        u is binned more finely than x: its bin width 1/u_max_bin must stay well
        below h for the trees to resolve the kernel.
        """
        params = {
            "objective": "regression",
            "seed": self.random_seed,
            "verbosity": -1,
            **self.regression_params,
        }
        x_max_bin = params.pop("max_bin", 255)
        params["max_bin_by_feature"] = [x_max_bin] * n_features + [self.u_max_bin]
        return _LGBMPredictor(params, self.min_leaf_mass)

    def _fit_kwargs(self, hs: list[float]) -> dict:
        return dict(val_frac=self.val_frac, n_grid_ISE=self.n_grid) if len(hs) > 1 else dict(val_frac=0.0)

    def fit(self, X, y) -> "CondensitePITBoostingWrapper":
        try:
            import condensite as cde
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "Failed to import condensite. Install condensite>=0.1.1 to use this wrapper."
            ) from exc

        X = self._sanitize_X(X)
        y = np.asarray(y, dtype=np.float64).reshape(-1)

        valid = np.isfinite(y)
        X, y = X[valid], y[valid]
        if len(y) == 0:
            raise ValueError("No valid (finite) training samples after sanitization")
        self._set_train_range(y)

        def model(hs):
            return cde.Condensite(
                self._predictor(X.shape[1]),
                h=hs,
                Mh=self.Mh,
                prediction_batch_size=self.prediction_batch_size,
            )

        # The folds and condensité's auxiliary points are drawn from the global
        # torch RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.random_seed)
            self._model = cde.PITBoosting(
                model(self.base_h), model(self.correction_h), K=self.K, n_grid_u=self.n_grid_u,
            ).fit(
                torch.as_tensor(X, dtype=torch.float32),
                torch.as_tensor(y, dtype=torch.float32),
                base_fit_kwargs=self._fit_kwargs(self.base_h),
                correction_fit_kwargs=self._fit_kwargs(self.correction_h),
                n_grid_PIT=self.n_grid,
            )
        self.lambda_ = float(self._model.lambda_)
        msg = f"λ = {self.lambda_:.3f}"
        # bandwidth selected by each fold model, when there was a choice
        for stage, hs, models in [("base", self.base_h, self._model.base_models),
                                  ("correction", self.correction_h, self._model.correction_models)]:
            if len(hs) > 1:
                msg += f", {stage} h " + "/".join(f"{m.best_h:g}" for m in models)
        print(f"    [CondensitePITBoostingWrapper] {msg}", flush=True)
        return self

    def predict_distribution(self, X) -> DistributionPrediction:
        X = torch.as_tensor(self._sanitize_X(X), dtype=torch.float32)
        cdes, y_points = self._model.predict(X, n_grid=self.n_grid)
        return grid_density_to_distribution(
            y_points.numpy().astype(np.float64).reshape(-1),
            cdes.numpy().astype(np.float64),
            train_range=self._y_train_range,
        )

    def predict(self, X) -> np.ndarray:
        return self.predict_distribution(X).mean
