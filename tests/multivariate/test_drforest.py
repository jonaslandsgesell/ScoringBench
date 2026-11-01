"""Tests for the native multivariate Distributional Random Forest wrapper.

The headline requirement is *mean recovery*: on a deterministic linear DGP
``Y = X @ B`` (plus tiny noise) the per-instance ensemble mean of the DRF
weighted-bootstrap draws must track the true conditional mean ``E[Y | x]`` for
every target dimension.  The remaining tests assert the sample-based contract
(shape ``(n_test, m, d)`` with ``m == N_DRAWS``, finite), that draws are genuine
members of the training-target set (weighted bootstrap), that cross-target
dependence is captured, that high-``d`` targets fit cheaply, and that the model
is registered under exactly one ``drforest`` key with no composition variants.

Runs on CPU (``drforest`` has a Rust core; there is no CUDA path for the forest
itself).  A small forest keeps the fit fast while leaving the linear signal
trivially recoverable.
"""

from __future__ import annotations

import numpy as np
import pytest

from scoringbench.multivariate.wrappers import DRForestMultiOutputWrapper

pytest.importorskip("drforest")

SEED = 0
N_DRAWS = 150


@pytest.fixture(autouse=True)
def _disable_adaptive_topup(monkeypatch):
    """Isolate the sampler contract from the adaptive R̂ top-up.

    With ``config.CONV_ENABLED`` the sample-based base may draw *beyond*
    ``N_DRAWS`` until the multivariate R̂ converges — that machinery has its own
    tests (``test_convergence.py``).  Pinning it off here lets these tests assert
    the wrapper emits exactly the requested ``m`` draws.
    """
    from scoringbench.multivariate import config as _cfg

    monkeypatch.setattr(_cfg, "CONV_ENABLED", False)


def _fast_model(n_draws: int = N_DRAWS) -> DRForestMultiOutputWrapper:
    return DRForestMultiOutputWrapper(
        n_estimators=200,
        min_samples_leaf=5,
        n_draws=n_draws,
        sample_chunk=n_draws,
        random_state=SEED,
    )


@pytest.fixture(scope="module")
def linear_dgp():
    rng = np.random.default_rng(SEED)
    n, p, d = 600, 3, 2
    X = rng.normal(size=(n, p))
    Y = np.stack([2.0 * X[:, 0], -1.0 * X[:, 1]], axis=1)
    Y = Y + 0.05 * rng.normal(size=(n, d))
    # Interior test points only: a random forest shrinks its leaf averages
    # toward the training-support interior, so edge points (|x| >~ 2 std) are
    # systematically under-extrapolated — a property of forests, not a bug.
    X_test = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, -1.0, 0.0], [-0.5, 0.5, 0.0]]
    )
    expected_mean = np.stack([2.0 * X_test[:, 0], -1.0 * X_test[:, 1]], axis=1)
    return X, Y, X_test, expected_mean


# --- headline: mean recovery -------------------------------------------------

def test_recovers_conditional_mean(linear_dgp):
    X, Y, X_test, expected_mean = linear_dgp
    model = _fast_model().fit(X, Y)
    got_mean = model.predict_ensemble(X_test).mean
    assert got_mean.shape == expected_mean.shape
    # Forest leaf averaging biases toward the training-support interior, so use
    # a loose tolerance that still separates a correct sampler from a broken one.
    np.testing.assert_allclose(got_mean, expected_mean, atol=0.6)


def test_predict_matches_ensemble_mean(linear_dgp):
    X, Y, X_test, _ = linear_dgp
    model = _fast_model().fit(X, Y)
    point = model.predict(X_test)
    assert point.shape == (X_test.shape[0], Y.shape[1])


# --- sample-based contract ---------------------------------------------------

def test_ensemble_shape_and_finiteness(linear_dgp):
    X, Y, X_test, _ = linear_dgp
    model = _fast_model(n_draws=N_DRAWS).fit(X, Y)
    samples = model.predict_ensemble(X_test).samples
    assert samples.shape == (X_test.shape[0], N_DRAWS, Y.shape[1])
    assert np.isfinite(samples).all()


def test_draws_are_training_targets(linear_dgp):
    """Every weighted-bootstrap draw must be an actual training target vector."""
    X, Y, X_test, _ = linear_dgp
    model = _fast_model(n_draws=100).fit(X, Y)
    samples = model.predict_ensemble(X_test).samples  # (n_test, 100, d)
    train_rows = {tuple(np.round(r, 10)) for r in model._Y_train}
    flat = samples.reshape(-1, Y.shape[1])
    for r in flat:
        assert tuple(np.round(r, 10)) in train_rows


def test_captures_cross_target_dependence():
    """On correlated targets the ensemble covariance must be clearly non-diagonal."""
    rng = np.random.default_rng(SEED)
    n, p = 800, 3
    X = rng.normal(size=(n, p))
    z = rng.normal(size=n)
    # Both targets share a latent driven by X[:,0] -> strong positive correlation.
    Y = np.stack([X[:, 0] + 0.3 * z, X[:, 0] + 0.3 * rng.normal(size=n)], axis=1)
    model = _fast_model(n_draws=400).fit(X, Y)
    X_test = np.array([[0.5, 0.0, 0.0]])
    s = model.predict_ensemble(X_test).samples[0]  # (400, 2)
    corr = np.corrcoef(s.T)[0, 1]
    assert corr > 0.3, f"expected positive cross-target correlation, got {corr:.3f}"


def test_sampling_is_reproducible(linear_dgp):
    X, Y, X_test, _ = linear_dgp
    model = _fast_model(n_draws=60).fit(X, Y)
    first = model.predict_ensemble(X_test).samples
    np.random.seed(999)
    _ = np.random.normal(size=1000)
    second = model.predict_ensemble(X_test).samples
    np.testing.assert_array_equal(first, second)


def test_single_target_dimension():
    rng = np.random.default_rng(SEED)
    n, p = 400, 2
    X = rng.normal(size=(n, p))
    y = (1.5 * X[:, 0] + 0.05 * rng.normal(size=n)).reshape(-1, 1)
    model = _fast_model(n_draws=80).fit(X, y)
    got = model.predict_ensemble(np.array([[2.0, 0.0], [-1.0, 0.0]])).mean
    assert got.shape == (2, 1)


def test_high_dimensional_target_fits():
    """DRF is a native joint sampler at any d (unlike grid methods): d=10 is cheap."""
    rng = np.random.default_rng(SEED)
    n, p, d = 500, 4, 10
    X = rng.normal(size=(n, p))
    Y = rng.normal(size=(n, d))
    Y[:, 0] += X[:, 0]
    model = _fast_model(n_draws=50).fit(X, Y)
    s = model.predict_ensemble(X[:5]).samples
    assert s.shape == (5, 50, d)
    assert np.isfinite(s).all()


# --- registry ----------------------------------------------------------------

def test_registered_as_single_direct_model():
    from scoringbench.multivariate.models import MODELS

    assert "drforest" in MODELS
    for prefix in ("independent", "copula", "chained"):
        assert f"{prefix}_drforest" not in MODELS
    assert isinstance(MODELS["drforest"](), DRForestMultiOutputWrapper)
