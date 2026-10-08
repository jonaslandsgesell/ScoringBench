"""Numerical contracts for Surjectors density-grid evaluation."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("jax")

from scoringbench.univariate.wrappers.surjectors_wrapper import SurjectorsWrapper


def model_with_log_density(values):
    values = np.asarray(values, dtype=float)
    model = SurjectorsWrapper(n_grid=values.shape[1], eval_chunk=2)
    model._x_mean = np.zeros((1, 1))
    model._x_std = np.ones((1, 1))
    model._grid_s = np.arange(values.shape[1], dtype=np.float32)
    model._grid_o = model._grid_s.copy()
    model._params = {}
    model._set_train_range(np.array([0., 3.]))

    def apply(params, rng, *, method, y, x):
        rows = np.asarray(x[:, 0], dtype=int)
        cols = np.asarray(y[:, 0], dtype=int)
        return values[rows, cols]

    model._fn = SimpleNamespace(apply=apply)
    return model


def test_log_density_offsets_preserve_mass_mean_and_median():
    base = np.log(np.array([0.1, 0.2, 0.3, 0.4]))
    offsets = np.array([-10000., 0., 10000.])
    model = model_with_log_density(base[None, :] + offsets[:, None])
    dist = model.predict_distribution(np.arange(3.)[:, None])
    np.testing.assert_allclose(dist.probas, np.tile(np.exp(base), (3, 1)), atol=1e-12)
    np.testing.assert_allclose(dist.probas.sum(axis=1), 1)
    np.testing.assert_allclose(dist.mean, 2.)
    assert np.isfinite(dist.median).all()
    np.testing.assert_allclose(dist.median, dist.median[0])


def test_negative_infinity_allows_zero_density_where_other_support_exists():
    model = model_with_log_density([[-np.inf, -1000., -1001., -np.inf]])
    dist = model.predict_distribution(np.zeros((1, 1)))
    np.testing.assert_array_equal(dist.probas[0, [0, 3]], 0)
    np.testing.assert_allclose(dist.probas.sum(axis=1), 1)
    assert np.isfinite(dist.median).all()


@pytest.mark.parametrize("values", [
    [[np.nan, 0., 1.]], [[0., np.inf, 1.]], [[-np.inf, -np.inf, -np.inf]],
])
def test_invalid_log_density_fails_before_distribution_conversion(values):
    model = model_with_log_density(values)
    with pytest.raises(ValueError, match="invalid log-densities.*rows"):
        model.predict_distribution(np.zeros((1, 1)))
