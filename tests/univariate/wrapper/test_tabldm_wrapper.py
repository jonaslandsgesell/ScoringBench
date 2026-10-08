"""TabLDM wrapper contracts without checkpoint downloads or GPU inference."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scoringbench.univariate.wrappers.tabldm import TabLDMWrapper


@pytest.fixture
def regressor(monkeypatch):
    class Regressor:
        def __init__(self, enhance_candidates=True, **kwargs):
            self.enhance_candidates = enhance_candidates
            self.kwargs = kwargs

        def fit(self, X, y):
            return self

        def predict(self, X, output_type="mean", alphas=None):
            if self.enhance_candidates or output_type == "mean":
                return np.arange(len(X), dtype=float)
            return np.arange(len(X))[:, None] + np.asarray(alphas)[None, :]

    monkeypatch.setitem(sys.modules, "tabldm", SimpleNamespace(TabLDMRegressor=Regressor))
    return Regressor


def test_defaults_select_quantile_capable_path_for_600_samples(regressor):
    model = TabLDMWrapper(device="cuda:0").fit(np.zeros((4, 2)), np.arange(4.))
    dist = model.predict_distribution(np.zeros((600, 2)))
    assert model._model.enhance_candidates is False
    assert model._model.kwargs["device"] == "cuda:0"
    assert dist.probas.shape[0] == 600
    assert dist.cdf_nodes[0].shape == (600, 200)
    np.testing.assert_allclose(dist.probas.sum(axis=1), 1)
    np.testing.assert_array_equal(model.predict(np.zeros((600, 2))), np.arange(600.))


def test_enhanced_mean_only_path_rejected_before_model_construction(regressor):
    with pytest.raises(ValueError, match="enhance_candidates"):
        TabLDMWrapper(enhance_candidates=True)


@pytest.mark.parametrize("n_quantiles", [0, -1, 2.5])
def test_invalid_quantile_count_rejected(regressor, n_quantiles):
    with pytest.raises(ValueError, match="n_quantiles"):
        TabLDMWrapper(n_quantiles=n_quantiles)


@pytest.mark.parametrize("kind", ["array", "dict", "list", "transposed", "single"])
def test_supported_quantile_outputs(regressor, kind):
    model = TabLDMWrapper(n_quantiles=3, enhance_candidates=False)
    model._set_train_range(np.array([0., 10.]))
    expected = np.array([[1., 2., 3.], [4., 5., 6.]])
    if kind == "single":
        expected = expected[:1]
        output = expected[0]
    elif kind == "dict":
        output = {"quantiles": expected}
    elif kind == "list":
        output = expected.tolist()
    elif kind == "transposed":
        output = expected.T
    else:
        output = expected
    model._model.predict = lambda *args, **kwargs: output
    dist = model.predict_distribution(np.zeros((len(expected), 2)))
    np.testing.assert_array_equal(dist.cdf_nodes[0], expected)


@pytest.mark.parametrize("n_samples", [600, 200])
def test_point_predictions_cannot_be_mistaken_for_quantiles(regressor, n_samples):
    model = TabLDMWrapper()
    model._set_train_range(np.array([0., 10.]))
    model._model.predict = lambda *args, **kwargs: np.arange(n_samples, dtype=float)
    with pytest.raises(ValueError, match="quantiles.*shape"):
        model.predict_distribution(np.zeros((n_samples, 2)))


@pytest.mark.parametrize("output", [
    {}, {"mean": np.zeros(2)}, np.zeros((2, 4)), np.zeros((1, 3)),
    np.zeros((2, 1, 3)), np.full((2, 3), np.nan), np.full((2, 3), np.inf),
])
def test_invalid_quantile_output_fails_explicitly(regressor, output):
    model = TabLDMWrapper(n_quantiles=3)
    model._set_train_range(np.array([0., 10.]))
    model._model.predict = lambda *args, **kwargs: output
    with pytest.raises(ValueError, match="quantiles"):
        model.predict_distribution(np.zeros((2, 2)))
