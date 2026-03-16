import numpy as np

from scoringbench.wrappers.base import ProbabilisticWrapper, DistributionPrediction
from scoringbench.wrappers.tabicl import TabICLWrapper
from scoringbench.wrappers.xgb_vector import XGBVectorWrapper, XGBQuantileVectorWrapper


def test_probabilistic_wrapper_contract():
    pw = ProbabilisticWrapper()
    # Base class should force subclasses to implement methods
    try:
        pw.fit(None, None)
        raised = False
    except NotImplementedError:
        raised = True
    assert raised

    try:
        pw.predict(None)
        raised = False
    except NotImplementedError:
        raised = True
    assert raised

    try:
        pw.predict_distribution(None)
        raised = False
    except NotImplementedError:
        raised = True
    assert raised


def test_tabicl_wrapper_predict_distribution_conversion():
    # Create instance without invoking __init__ (avoids external dependency)
    w = TabICLWrapper.__new__(TabICLWrapper)
    # Use a small, test-friendly quantile grid
    w._ALPHAS = [0.25, 0.5, 0.75]

    class Model:
        def predict(self, X_arr, output_type=None, alphas=None):
            # Return a simple per-sample quantile matrix for two samples
            return np.array([[0.0, 0.5, 1.0], [0.1, 0.6, 0.9]])

    w._model = Model()

    X = np.zeros((2, 1))
    dist = w.predict_distribution(X)

    assert isinstance(dist, DistributionPrediction)
    assert dist.probas.shape[0] == 2
    # bin_edges may be shared (1-D) or per-sample (2-D)
    expected_edges = len(w._ALPHAS) + 2
    if dist.bin_edges.ndim == 1:
        assert dist.bin_edges.shape[0] == expected_edges
    else:
        assert dist.bin_edges.shape == (2, expected_edges)
    expected_mids = expected_edges - 1
    if dist.bin_midpoints.ndim == 1:
        assert dist.bin_midpoints.shape[0] == expected_mids
    else:
        assert dist.bin_midpoints.shape == (2, expected_mids)


def test_xgb_vector_wrapper_predicts_and_distribution():
    w = XGBVectorWrapper.__new__(XGBVectorWrapper)
    w.n_bins = 4
    # synthetic midpoints / edges
    w._bin_midpoints = np.array([0.0, 1.0, 2.0, 3.0])
    w._bin_edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    class FakeModel:
        def predict(self, *args, **kwargs):
            # return logits for 3 samples
            return np.array([[0.1, 0.2, 0.3, 0.4], [1.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 0.0]])

    w._model = FakeModel()

    X = np.zeros((3, 2))
    preds = w.predict(X)
    dist = w.predict_distribution(X)

    assert preds.shape[0] == 3
    assert isinstance(dist, DistributionPrediction)
    assert dist.probas.shape == (3, w.n_bins)


def test_xgb_quantile_vector_wrapper_predict_distribution():
    w = XGBQuantileVectorWrapper.__new__(XGBQuantileVectorWrapper)
    # small alpha grid for testing
    w._alphas = np.array([0.2, 0.5, 0.8])

    class FakeModel:
        def predict(self, *args, **kwargs):
            # return quantiles for 2 samples
            return np.array([[0.0, 0.5, 1.0], [0.2, 0.6, 0.9]])

    w._model = FakeModel()

    X = np.zeros((2, 1))
    dist = w.predict_distribution(X)

    assert isinstance(dist, DistributionPrediction)
    assert dist.probas.shape == (2, len(w._alphas) + 1)
    # bin_edges can be shared (1-D) or per-sample (2-D)
    expected_edges = len(w._alphas) + 2
    if dist.bin_edges.ndim == 1:
        assert dist.bin_edges.shape[0] == expected_edges
    else:
        assert dist.bin_edges.shape == (2, expected_edges)
