"""XGBoost wrapper with custom CRPS objective for ScoringBench."""

from __future__ import annotations

import numpy as np
import xgboost as xgb

from .base import DistributionPrediction, ProbabilisticWrapper


def crps_objective(preds, dtrain):
    """Exact Diagonal Hessian CRPS objective for XGBoost (for equally spaced bins only)"""
    labels = dtrain.get_label().astype(int)
    num_samples = labels.shape[0]
    
    if preds.ndim == 1:
        num_class = preds.size // num_samples
        preds = preds.reshape(num_samples, num_class)

    # Softmax
    shift_preds = preds - np.max(preds, axis=1, keepdims=True)
    exps = np.exp(shift_preds)
    probs = exps / np.sum(exps, axis=1, keepdims=True)

    # One-hot targets
    target_one_hot = np.zeros_like(probs)
    target_one_hot[np.arange(num_samples), labels] = 1

    # CDFs
    pred_cdf = np.cumsum(probs, axis=1)
    target_cdf = np.cumsum(target_one_hot, axis=1)
    cdf_diff = pred_cdf - target_cdf # (P_k - Y_k)

    # Gradient (unchanged)
    grad_p = 2 * np.flip(np.cumsum(np.flip(cdf_diff, axis=1), axis=1), axis=1)
    weighted_grad_p = np.sum(grad_p * probs, axis=1, keepdims=True)
    grad_z = probs * (grad_p - weighted_grad_p)

    # --- EXACT HESSIAN CALCULATION ---
    
    # 1. Calculate A_j (The GN term)
    V = (1.0 - pred_cdf) ** 2
    V_total = np.sum(V, axis=1, keepdims=True)
    diff_A = 2.0 * pred_cdf - 1.0
    A = V_total + np.pad(np.cumsum(diff_A, axis=1)[:, :-1], ((0, 0), (1, 0)), constant_values=0)

    # 2. Calculate B_j (The Residual term)
    # B_j = sum_{k=j}^K (P_k - Y_k)(1 - P_k) + sum_{k=1}^{j-1} (P_k - Y_k)(-P_k)
    term_left = cdf_diff * (1.0 - pred_cdf)
    term_right = -cdf_diff * pred_cdf
    
    # Summing terms for B_j
    sum_left = np.flip(np.cumsum(np.flip(term_left, axis=1), axis=1), axis=1)
    sum_right = np.cumsum(term_right, axis=1)
    exclusive_sum_right = np.pad(sum_right[:, :-1], ((0, 0), (1, 0)), constant_values=0)
    B = sum_left + exclusive_sum_right

    # 3. Combine for Exact Hessian
    hess_z = 2.0 * (probs**2) * A + 2.0 * (probs - 2.0 * (probs**2)) * B

    # Stability: XGBoost split gain formula (G^2 / (H + lambda)) fails if H <= 0.
    # We take the absolute value or a small epsilon to satisfy the solver.
    hess_z = np.maximum(np.abs(hess_z), 1e-6)

    return grad_z, hess_z


class XGBVectorWrapper(ProbabilisticWrapper):
    """XGBoost with a custom CRPS objective, predicting a discretized distribution.

    Discretizes y uniformly into n_bins bins at fit time, trains XGBoost as
    a multi-class classifier using the CRPS gradient/hessian, then reads back
    the per-bin probabilities at predict time.

    Parameters
    ----------
    n_bins : int
        Number of histogram bins for the discretized distribution.
    num_boost_round : int
        Number of XGBoost boosting rounds.
    xgb_params : dict, optional
        Extra XGBoost parameters that override the defaults.
    """

    _DEFAULT_PARAMS = {
        # rank:pairwise is overwritten by the custom CRPS objective at train
        # time; it only determines output dimensionality / XGBoost internals.
        "objective": "rank:pairwise",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "multi_strategy": "multi_output_tree",
        #"eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 0.001,
        "lambda": 1,
        "alpha": 0.5,
        "device": "cuda"
    }



    def __init__(self, n_bins: int = 50, num_boost_round: int = 100, xgb_params: dict | None = None):
        self.n_bins = n_bins
        self.num_boost_round = num_boost_round
        self.xgb_params = xgb_params or {}
        self._model: xgb.Booster | None = None
        self._bin_edges: np.ndarray | None = None
        self._bin_midpoints: np.ndarray | None = None

    def fit(self, X, y) -> "XGBVectorWrapper":
        y = np.asarray(y, dtype=float)

        # Uniform discretization over the observed range
        self._bin_edges = np.linspace(y.min(), y.max(), self.n_bins + 1)
        self._bin_midpoints = (self._bin_edges[:-1] + self._bin_edges[1:]) / 2

        # Map each y value to its bin index (0-indexed, clipped to valid range)
        y_labels = np.clip(
            np.digitize(y, self._bin_edges[1:-1]),  # n_bins-1 interior edges → 0..n_bins-1
            0,
            self.n_bins - 1,
        )

        dtrain = xgb.DMatrix(X, label=y_labels)

        params = {
            **self._DEFAULT_PARAMS,
            "num_class": self.n_bins,
            **self.xgb_params,
        }

        self._model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.num_boost_round,
            obj=crps_objective,
        )
        return self

    def _get_probs(self, X) -> np.ndarray:
        dtest = xgb.DMatrix(X)
        # output_margin=True returns raw logits for all classes rather than
        # class-index scalars (which is what multi:softmax gives by default).
        raw = self._model.predict(dtest, output_margin=True)
        # Depending on XGBoost version / multi_strategy, raw may already be
        # (n_samples, n_bins) or still a flat (n_samples * n_bins,) array.
        if raw.ndim == 2 and raw.shape[1] == self.n_bins:
            logits = raw
        else:
            n_samples = raw.size // self.n_bins
            logits = raw.reshape(n_samples, self.n_bins)

        shift = logits - np.max(logits, axis=1, keepdims=True)
        exps = np.exp(shift)
        return exps / np.sum(exps, axis=1, keepdims=True)

    def predict(self, X) -> np.ndarray:
        probs = self._get_probs(X)
        return (probs * self._bin_midpoints[None, :]).sum(axis=-1)

    def predict_distribution(self, X) -> DistributionPrediction:
        probs = self._get_probs(X)
        mean = (probs * self._bin_midpoints[None, :]).sum(axis=-1)

        return DistributionPrediction(
            probas=probs,
            bin_edges=self._bin_edges,
            bin_midpoints=self._bin_midpoints,
            mean=mean,
        )


class XGBQuantileVectorWrapper(ProbabilisticWrapper):
    """
    XGBoost 1000 adaptive quantiles for continuous CRPS minimization.
    Uses per-sample bin_edges and bin_midpoints to eliminate discretization bias.
    """

    _DEFAULT_PARAMS = {
        "objective": "reg:quantileerror",
        "tree_method": "hist",
        "multi_strategy": "multi_output_tree",
        # "subsample": 0.8,
        # "colsample_bytree": 0.8,
        # "learning_rate": 0.05,
        "device": "cuda"
    }

    def __init__(self, n_bins: int = 200, num_boost_round: int = 100, xgb_params: dict | None = None):
        self.n_quantiles = n_bins
        self.num_boost_round = num_boost_round
        self.xgb_params = xgb_params or {}
        
        # Grid of 1000 probability levels
        self._alphas = np.linspace(0.001, 0.999, n_bins)
        self._model: xgb.Booster | None = None
        self._y_range: tuple[float, float] = (0.0, 1.0)

    def fit(self, X, y) -> "XGBQuantileVectorWrapper":
        y = np.asarray(y, dtype=float)
        self._y_range = (float(y.min()), float(y.max()))

        dtrain = xgb.DMatrix(X, label=y)

        params = {
            **self._DEFAULT_PARAMS,
            "quantile_alpha": self._alphas.tolist(),
            **self.xgb_params,
        }

        self._model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.num_boost_round,
        )
        return self

    def predict_distribution(self, X) -> DistributionPrediction:
        dtest = xgb.DMatrix(X)
        # q shape: (n_samples, 1000)
        q = self._model.predict(dtest)
        
        # 1. Finite-value protection
        if not np.all(np.isfinite(q)):
            q = np.nan_to_num(q, nan=self._y_range[0], 
                             posinf=self._y_range[1], neginf=self._y_range[0])
        
        # 2. Strict monotonicity (Resort)
        q = np.sort(q, axis=1)
        n_samples = q.shape[0]

        # 3. Construct Per-Sample Bin Edges
        # We extend the edges slightly beyond the min/max quantiles to 
        # cover the full probability mass [0, 1].
        left_w = np.maximum(q[:, 1] - q[:, 0], 1e-7)
        right_w = np.maximum(q[:, -1] - q[:, -2], 1e-7)
        
        # bin_edges shape: (n_samples, n_quantiles + 1)
        bin_edges = np.concatenate(
            [(q[:, 0] - left_w)[:, None], q, (q[:, -1] + right_w)[:, None]], 
            axis=1
        )
        
        # 4. Construct Probabilities (Mass per Bin)
        # mass_0 = alpha_0, mass_i = alpha_i - alpha_{i-1}, mass_last = 1 - alpha_last
        masses = np.concatenate([[self._alphas[0]], np.diff(self._alphas), [1.0 - self._alphas[-1]]])
        probas = np.broadcast_to(masses[None, :], (n_samples, len(masses))).copy()

        # 5. Midpoints and Mean
        bin_midpoints = (bin_edges[:, :-1] + bin_edges[:, 1:]) / 2
        mean = np.sum(probas * bin_midpoints, axis=-1)

        return DistributionPrediction(
            probas=probas,
            bin_edges=bin_edges,
            bin_midpoints=bin_midpoints,
            mean=mean,
        )

    def predict(self, X) -> np.ndarray:
        # High-precision mean directly from quantile particles
        dtest = xgb.DMatrix(X)
        q = self._model.predict(dtest)
        return np.mean(q, axis=1)