from types import SimpleNamespace

import numpy as np

from scoringbench.multivariate.wrappers import CausiloSampler


def test_causilo_sampler_uses_quantile_cdf_nodes():
    sampler = CausiloSampler(device="cpu")
    support = np.tile(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]), (2, 1))
    cdf = np.tile(np.linspace(0.0, 1.0, support.shape[1]), (2, 1))
    sampler._model = SimpleNamespace(
        predict_distribution=lambda X: SimpleNamespace(cdf_nodes=(support, cdf))
    )
    X = np.zeros((2, 1))
    levels = np.array([0.25, 0.75])

    values = sampler.quantile(X, levels)
    recovered_levels = sampler.cdf(X, values)
    draws = sampler.sample(X, 4, rng=np.random.default_rng(0))

    np.testing.assert_allclose(recovered_levels, levels)
    assert draws.shape == (2, 4)
    assert np.all(np.isfinite(draws))
