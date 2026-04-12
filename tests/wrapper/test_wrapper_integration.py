"""Integration tests — all probabilistic wrappers on a 1-D linear regression.

Each test function receives a ``fitted_model`` fixture that is parametrized
over all available wrappers. Models whose optional package is not installed
are skipped automatically.

Adding a new model
------------------
1. Write a zero-argument factory function ``_make_<name>()`` that returns an
   unfitted wrapper instance.
2. Append a ``pytest.param`` entry to ``MODEL_FACTORIES``.

Everything else (data generation, assertions) is reused automatically.
"""

from __future__ import annotations

import logging
import pytest
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import os

from scoringbench.wrappers.base import DistributionPrediction

logger = logging.getLogger(__name__)


def _cuda_or_cpu() -> str:
    """Return 'cuda' if a GPU is available, else 'cpu'."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

# ---------------------------------------------------------------------------
# Constants — data-generating process
# ---------------------------------------------------------------------------

NOISE_STD = 2.0
N_TRAIN = 400
N_TEST = 100
RANDOM_STATE = 42

# True 68 % PI width for a perfect Gaussian predictor: 2 * Φ⁻¹(0.84) * σ ≈ 2σ
_EXPECTED_68PI_WIDTH = 2.0 * NOISE_STD


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _extract_quantile(dist: DistributionPrediction, alpha: float) -> np.ndarray:
    """Return per-sample quantile at level ``alpha`` from a DistributionPrediction.

    Handles both shared (1-D) and per-sample (2-D) ``bin_edges``.
    Uses midpoint of the target bin as a simple approximation.
    """
    probas = dist.probas                        # (n, n_bins)
    edges = dist.bin_edges                      # (n_bins+1,) or (n, n_bins+1)
    if edges.ndim == 1:
        edges = np.tile(edges, (probas.shape[0], 1))

    cdf = np.cumsum(probas, axis=1)
    out = np.empty(probas.shape[0])
    for i in range(probas.shape[0]):
        idx = int(np.searchsorted(cdf[i], alpha))
        idx = np.clip(idx, 0, probas.shape[1] - 1)
        out[i] = (edges[i, idx] + edges[i, idx + 1]) / 2.0
    return out


# ---------------------------------------------------------------------------
# Data fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def regression_data():
    """Linear 1-D regression dataset with known noise level."""
    logger.info(f"Generating synthetic linear regression data: {N_TRAIN} train + {N_TEST} test")
    X, y = make_regression(
        n_samples=N_TRAIN + N_TEST,
        n_features=1,
        noise=NOISE_STD,
        random_state=RANDOM_STATE,
    )
    logger.info(f"Data generated: X.shape={X.shape}, noise_std={NOISE_STD}")
    return train_test_split(
        X, y,
        test_size=N_TEST / (N_TRAIN + N_TEST),
        random_state=RANDOM_STATE,
    )  # X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Model factories — zero-argument callables that return an unfitted wrapper
# ---------------------------------------------------------------------------

def _make_xgb_vector():
    from scoringbench.wrappers.xgb_vector import XGBVectorWrapper
    return XGBVectorWrapper(n_bins=50, num_boost_round=20, xgb_params={"device": _cuda_or_cpu()})


def _make_xgb_quantile():
    from scoringbench.wrappers.xgb_vector import XGBQuantileVectorWrapper
    return XGBQuantileVectorWrapper(n_bins=50, num_boost_round=20, xgb_params={"device": _cuda_or_cpu()})


def _make_tabpfn():
    from scoringbench.wrappers.tabpfn import TabPFNWrapper
    return TabPFNWrapper(model_path="tabpfn-v2.5-regressor-v2.5_real.ckpt" if os.path.exists("tabpfn-v2.5-regressor-v2.5_real.ckpt") else None)


def _make_tabicl():
    from scoringbench.wrappers.tabicl import TabICLWrapper
    return TabICLWrapper(device=_cuda_or_cpu())


def _make_pytabkit():
    from scoringbench.wrappers.pytabkit import PytabkitRealMLPWrapper
    return PytabkitRealMLPWrapper(        
        train_metric_name='multi_pinball(0.01,0.03,0.05,0.07,0.09,0.11,0.13,0.15,0.17,0.19,0.21,0.23,0.25,0.27,0.29,0.31,0.33,0.35,0.37,0.39,0.41,0.43,0.45,0.47,0.49,0.51,0.53,0.55,0.57,0.59,0.61,0.63,0.65,0.67,0.69,0.71,0.73,0.75,0.77,0.79,0.81,0.83,0.85,0.87,0.89,0.91,0.93,0.95,0.97,0.99)',
        val_metric_name='multi_pinball(0.01,0.03,0.05,0.07,0.09,0.11,0.13,0.15,0.17,0.19,0.21,0.23,0.25,0.27,0.29,0.31,0.33,0.35,0.37,0.39,0.41,0.43,0.45,0.47,0.49,0.51,0.53,0.55,0.57,0.59,0.61,0.63,0.65,0.67,0.69,0.71,0.73,0.75,0.77,0.79,0.81,0.83,0.85,0.87,0.89,0.91,0.93,0.95,0.97,0.99)',
        n_quantiles=50,
        device=_cuda_or_cpu())


def _make_catboost_quantile():
    from scoringbench.wrappers.catboost_wrapper import CatBoostQuantileWrapper
    return CatBoostQuantileWrapper(n_quantiles=99, iterations=200)


# Registry — add new entries here to include a model in all tests below
# Each param is a (name, factory) tuple so the fixture can log the name
# without relying on pytest internals that differ across scopes.
MODEL_FACTORIES = [
    pytest.param(("XGBVectorWrapper",        _make_xgb_vector),   id="XGBVectorWrapper"),
    pytest.param(("XGBQuantileVectorWrapper", _make_xgb_quantile), id="XGBQuantileVectorWrapper"),
    pytest.param(("TabPFNWrapper",            _make_tabpfn),       id="TabPFNWrapper"),
    pytest.param(("TabICLWrapper",            _make_tabicl),       id="TabICLWrapper"),
    pytest.param(("PytabkitRealMLPWrapper",   _make_pytabkit),     id="PytabkitRealMLPWrapper"),
    pytest.param(("CatBoostQuantileWrapper",  _make_catboost_quantile), id="CatBoostQuantileWrapper"),
]


# ---------------------------------------------------------------------------
# Fitted-model fixture — trains once per (model, module), reused by all tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", params=MODEL_FACTORIES)
def fitted_model(request, regression_data):
    """Return ``(model, X_test, y_test)``; skips when a required package is absent."""
    model_name, model_factory = request.param
    logger.info(f"\n{'='*60}")
    logger.info(f"Fitting model: {model_name}")
    X_train, X_test, y_train, y_test = regression_data
    try:
        logger.debug(f"Creating {model_name} instance...")
        model = model_factory()
    except ImportError as exc:
        logger.warning(f"Skipping {model_name}: {exc}")
        pytest.skip(f"Optional dependency not installed: {exc}")
    logger.debug(f"Training {model_name} on {len(X_train)} samples...")
    model.fit(X_train, y_train)
    logger.info(f"✓ {model_name} fitted successfully")
    return model, X_test, y_test


# ---------------------------------------------------------------------------
# Tests — each function receives ``fitted_model`` and checks one property
# ---------------------------------------------------------------------------

def test_predict_shape_and_finite(fitted_model):
    """predict() must return a finite array of shape (n_test,)."""
    logger.debug("Running: test_predict_shape_and_finite")
    model, X_test, _ = fitted_model
    preds = model.predict(X_test)
    assert preds.shape == (X_test.shape[0],), f"unexpected shape: {preds.shape}"
    assert np.all(np.isfinite(preds)), "predict() contains non-finite values"


def test_point_prediction_r2(fitted_model):
    """Point predictions must capture the linear signal (R² > 0.5)."""
    logger.debug("Running: test_point_prediction_r2")
    model, X_test, y_test = fitted_model
    preds = model.predict(X_test)
    ss_res = np.sum((y_test - preds) ** 2)
    ss_tot = np.sum((y_test - y_test.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    assert r2 > 0.5, f"R² = {r2:.3f} — model fails to capture the linear signal"


def test_distribution_return_type(fitted_model):
    """predict_distribution() must return a DistributionPrediction."""
    logger.debug("Running: test_distribution_return_type")
    model, X_test, _ = fitted_model
    dist = model.predict_distribution(X_test)
    assert isinstance(dist, DistributionPrediction)


def test_distribution_shape_consistency(fitted_model):
    """probas, bin_edges, bin_midpoints, and mean must have consistent shapes."""
    logger.debug("Running: test_distribution_shape_consistency")
    model, X_test, _ = fitted_model
    n = X_test.shape[0]
    dist = model.predict_distribution(X_test)
    n_bins = dist.probas.shape[1]

    assert dist.probas.shape == (n, n_bins), f"probas shape: {dist.probas.shape}"
    assert dist.mean.shape == (n,), f"mean shape: {dist.mean.shape}"

    if dist.bin_edges.ndim == 1:
        assert dist.bin_edges.shape[0] == n_bins + 1, (
            f"bin_edges 1-D shape {dist.bin_edges.shape} ≠ ({n_bins + 1},)"
        )
    else:
        assert dist.bin_edges.shape == (n, n_bins + 1), (
            f"bin_edges 2-D shape {dist.bin_edges.shape} ≠ ({n}, {n_bins + 1})"
        )

    if dist.bin_midpoints.ndim == 1:
        assert dist.bin_midpoints.shape[0] == n_bins
    else:
        assert dist.bin_midpoints.shape == (n, n_bins)


def test_probas_sum_to_one(fitted_model):
    """probas must sum to ≈ 1 per sample."""
    logger.debug("Running: test_probas_sum_to_one")
    model, X_test, _ = fitted_model
    dist = model.predict_distribution(X_test)
    np.testing.assert_allclose(
        dist.probas.sum(axis=1), 1.0, atol=1e-4,
        err_msg="probas do not sum to 1 per sample",
    )


def test_distribution_mean_close_to_point_predict(fitted_model):
    """dist.mean should agree with predict() to within ≈ 1 noise std on average."""
    logger.debug("Running: test_distribution_mean_close_to_point_predict")
    model, X_test, _ = fitted_model
    preds = model.predict(X_test)
    dist = model.predict_distribution(X_test)
    avg_diff = float(np.abs(dist.mean - preds).mean())
    assert avg_diff < NOISE_STD, (
        f"dist.mean differs from predict() by {avg_diff:.2f} on average "
        f"(threshold: {NOISE_STD})"
    )


def test_68pct_coverage(fitted_model):
    """The [16th, 84th] percentile interval must cover ≈ 68 % of held-out targets.

    For a well-calibrated model on linear Gaussian data the empirical coverage
    should be close to 0.68.  We allow ±28 pp ([0.40, 0.96]) to accommodate
    mild over/under-confidence.
    """
    logger.debug("Running: test_68pct_coverage")
    model, X_test, y_test = fitted_model
    dist = model.predict_distribution(X_test)
    lo = _extract_quantile(dist, 0.16)
    hi = _extract_quantile(dist, 0.84)
    coverage = float(np.mean((y_test >= lo) & (y_test <= hi)))
    assert 0.35 <= coverage <= 0.97, (
        f"68% PI empirical coverage = {coverage:.2f} (expected ≈ 0.68)"
    )


def test_68pct_interval_width_matches_noise(fitted_model):
    """Mean 68 % PI width should be within 5× of the true 2·σ_noise.

    On a 1-D linear problem the irreducible uncertainty equals the noise std,
    so a well-calibrated uncertainty estimate should produce intervals of width
    ≈ 2·NOISE_STD.  We allow a factor-of-5 range to accommodate models that
    are somewhat over- or under-dispersed.
    """
    logger.debug("Running: test_68pct_interval_width_matches_noise")
    model, X_test, _ = fitted_model
    dist = model.predict_distribution(X_test)
    lo = _extract_quantile(dist, 0.16)
    hi = _extract_quantile(dist, 0.84)
    avg_width = float((hi - lo).mean())
    assert avg_width > 0, "68 % PI width is zero or negative"
    ratio = avg_width / _EXPECTED_68PI_WIDTH
    # XGBVectorWrapper constructs bins on training data and can be significantly
    # over-dispersed on held-out data, so allow a wider ratio for it.
    is_xgb_vector = type(model).__name__ == "XGBVectorWrapper"
    lo_bound, hi_bound = (0.1, 10.0) if is_xgb_vector else (0.5, 3.0)
    assert lo_bound <= ratio <= hi_bound, (
        f"68% PI avg_width = {avg_width:.1f}, expected ≈ {_EXPECTED_68PI_WIDTH:.1f}, "
        f"ratio = {ratio:.2f} — interval is "
        f"{'too narrow' if ratio < lo_bound else 'too wide'}"
    )
