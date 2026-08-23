"""Tests for the native multivariate Forest-Diffusion wrapper.

The headline requirement is a *mean-recovery* check: on a deterministic linear
DGP ``Y = X @ B`` (plus tiny noise), the per-instance ensemble mean of the
Forest-Diffusion draws must track the true conditional mean ``E[Y | x] = x @ B``
for **every** target dimension.  This is the multivariate analogue of the
univariate mean check and confirms the joint conditional sampler is wired up
correctly (features conditioned via ``X_covs``, all ``d`` columns diffused).

The remaining tests assert the sample-based contract: draws have shape
``(n_test, m, d)`` with ``m == N_DRAWS``, are finite, and the model is
registered under exactly one ``forest_diffusion`` key (no chained / copula
variants).

These run on CPU only (XGBoost trees); no GPU is required, unlike the TabPFN /
TabICL samplers.  A small ``n_t`` / ``duplicate_K`` / ``n_estimators`` keeps the
fit fast while leaving the linear signal trivially recoverable.
"""

from __future__ import annotations

import numpy as np
import pytest

from scoringbench.multivariate.wrappers import ForestDiffusionMultiOutputWrapper

pytest.importorskip("ForestDiffusion")

SEED = 0
N_DRAWS = 120  # light MC budget; the linear signal is huge, so this suffices.


def _fast_model(n_draws: int = N_DRAWS) -> ForestDiffusionMultiOutputWrapper:
    """A deliberately small/fast Forest-Diffusion for unit tests."""
    return ForestDiffusionMultiOutputWrapper(
        n_t=25,
        duplicate_K=50,
        diffusion_type="flow",
        n_estimators=50,
        max_depth=5,
        n_jobs=-1,
        n_draws=n_draws,
        sample_chunk=20,
        random_state=SEED,
    )


@pytest.fixture(scope="module")
def linear_dgp():
    """Deterministic linear DGP with a recoverable conditional mean.

    ``Y[:, 0] = 2 * X[:, 0]`` and ``Y[:, 1] = -1 * X[:, 1]`` (+ tiny noise), so
    the true conditional mean at a test row ``x`` is ``[2*x0, -x1]``.
    """
    rng = np.random.default_rng(SEED)
    n, p, d = 300, 3, 2
    X = rng.normal(size=(n, p))
    Y = np.stack([2.0 * X[:, 0], -1.0 * X[:, 1]], axis=1)
    Y = Y + 0.05 * rng.normal(size=(n, d))

    X_test = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 2.0, 0.0]]
    )
    expected_mean = np.stack([2.0 * X_test[:, 0], -1.0 * X_test[:, 1]], axis=1)
    return X, Y, X_test, expected_mean


# ---------------------------------------------------------------------------
# Headline: mean recovery on every target dimension
# ---------------------------------------------------------------------------

def test_recovers_conditional_mean(linear_dgp):
    X, Y, X_test, expected_mean = linear_dgp
    model = _fast_model().fit(X, Y)

    pred = model.predict_ensemble(X_test)
    got_mean = pred.mean  # (n_test, d)

    assert got_mean.shape == expected_mean.shape
    # The linear signal dominates the tiny noise; a loose tolerance still
    # separates a correct conditional sampler from a broken one.
    np.testing.assert_allclose(got_mean, expected_mean, atol=0.35)


def test_predict_matches_ensemble_mean(linear_dgp):
    X, Y, X_test, expected_mean = linear_dgp
    model = _fast_model().fit(X, Y)
    point = model.predict(X_test)  # base class -> ensemble mean
    assert point.shape == expected_mean.shape
    np.testing.assert_allclose(point, expected_mean, atol=0.35)


# ---------------------------------------------------------------------------
# Sample-based contract
# ---------------------------------------------------------------------------

def test_ensemble_shape_and_finiteness(linear_dgp):
    X, Y, X_test, _ = linear_dgp
    model = _fast_model(n_draws=N_DRAWS).fit(X, Y)
    pred = model.predict_ensemble(X_test)
    samples = pred.samples
    assert samples.shape == (X_test.shape[0], N_DRAWS, Y.shape[1])
    assert np.isfinite(samples).all()


def test_draws_are_distinct_and_spread(linear_dgp):
    """Draws must be genuine samples: distinct, with non-degenerate spread.

    A broken loop that reused the same prior noise would return (near-)identical
    draws and collapse the predictive law.  We assert every draw is distinct and
    the per-instance ensemble std is clearly non-zero on a DGP with real
    residual noise.
    """
    rng = np.random.default_rng(SEED)
    n, p = 300, 3
    X = rng.normal(size=(n, p))
    # heteroscedastic-ish residual so the true predictive spread is non-trivial
    Y = np.stack([2.0 * X[:, 0], -1.0 * X[:, 1]], axis=1) + 0.8 * rng.normal(size=(n, 2))
    model = _fast_model(n_draws=150).fit(X, Y)

    X_test = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    samples = model.predict_ensemble(X_test).samples  # (2, 150, 2)

    for r in range(X_test.shape[0]):
        row = samples[r]  # (150, 2)
        # all draws distinct (no collapse)
        uniq = np.unique(np.round(row, 8), axis=0)
        assert uniq.shape[0] == row.shape[0], "draws collapsed / duplicated"
        # non-degenerate predictive spread in both dimensions
        assert row[:, 0].std() > 0.1
        assert row[:, 1].std() > 0.1


def test_sampling_is_reproducible_and_isolated(linear_dgp):
    """Two predict calls match, and external np.random use does not perturb draws.

    The wrapper snapshots/seeds/restores the global RNG around the generate
    loop, so results are self-contained: identical across calls and immune to
    unrelated ``np.random`` activity in between.
    """
    X, Y, X_test, _ = linear_dgp
    model = _fast_model(n_draws=60).fit(X, Y)

    first = model.predict_ensemble(X_test).samples
    # Perturb the global RNG between calls; must not affect the draws.
    np.random.seed(12345)
    _ = np.random.normal(size=1000)
    second = model.predict_ensemble(X_test).samples

    np.testing.assert_array_equal(first, second)


def test_single_target_dimension():
    """A d=1 target must still work (p_in_one=False single-column path)."""
    rng = np.random.default_rng(SEED)
    n, p = 200, 2
    X = rng.normal(size=(n, p))
    y = (1.5 * X[:, 0] + 0.05 * rng.normal(size=n)).reshape(-1, 1)
    model = _fast_model(n_draws=80).fit(X, y)
    X_test = np.array([[2.0, 0.0], [-1.0, 0.0]])
    got = model.predict_ensemble(X_test).mean
    assert got.shape == (2, 1)
    np.testing.assert_allclose(got[:, 0], [3.0, -1.5], atol=0.5)


# ---------------------------------------------------------------------------
# Registry: exactly one direct model, no composition variants
# ---------------------------------------------------------------------------

def test_registered_as_single_direct_model():
    from scoringbench.multivariate.models import MODELS

    assert "forest_diffusion" in MODELS
    # No composition-prefixed variants should exist for forest diffusion.
    for prefix in ("independent", "copula", "chained"):
        assert f"{prefix}_forest_diffusion" not in MODELS

    factory = MODELS["forest_diffusion"]
    assert isinstance(factory(), ForestDiffusionMultiOutputWrapper)
