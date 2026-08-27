import numpy as np

from scoringbench.univariate.wrappers.base import ProbabilisticWrapper, DistributionPrediction
from scoringbench.univariate.wrappers.tabicl import TabICLWrapper
from scoringbench.univariate.wrappers.xgb_vector import XGBVectorWrapper, XGBQuantileVectorWrapper


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
    w._y_train_range = (0.0, 1.0)

    class Model:
        def predict(self, X_arr, output_type=None, alphas=None):
            # Return a simple per-sample quantile matrix for two samples
            return np.array([[0.0, 0.5, 1.0], [0.1, 0.6, 0.9]])

    w._model = Model()

    X = np.zeros((2, 1))
    dist = w.predict_distribution(X)

    assert isinstance(dist, DistributionPrediction)
    assert dist.probas.shape[0] == 2

    # TabICL uses the shared quantile->distribution mapping
    # (``quantiles_to_distribution``): the predicted quantiles are used verbatim
    # as the native edges, so K levels give K edges and K-1 bins on a
    # per-sample (2-D) grid.
    n_bins = len(w._ALPHAS) - 1
    assert dist.probas.shape == (2, n_bins)
    assert dist.bin_edges.shape == (2, len(w._ALPHAS))
    assert dist.bin_midpoints.shape == (2, n_bins)

    # Each row is a valid PMF on a non-decreasing grid.
    assert np.allclose(dist.probas.sum(axis=1), 1.0)
    assert np.all(dist.probas >= 0.0)
    assert np.all(np.diff(dist.bin_edges, axis=1) >= 0)


def test_xgb_vector_wrapper_predicts_and_distribution():
    w = XGBVectorWrapper.__new__(XGBVectorWrapper)
    w.n_bins = 4
    # synthetic midpoints / edges
    w._bin_midpoints = np.array([0.0, 1.0, 2.0, 3.0])
    w._bin_edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    w._y_train_range = (0.0, 1.0)

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
    # __new__ bypasses __init__, so set the range that predict_distribution
    # forwards to quantiles_to_distribution (default is (0.0, 1.0)).
    w._y_range = (0.0, 1.0)
    w._y_train_range = (0.0, 1.0)

    class FakeModel:
        def predict(self, *args, **kwargs):
            # return quantiles for 2 samples
            return np.array([[0.0, 0.5, 1.0], [0.2, 0.6, 0.9]])

    w._model = FakeModel()

    X = np.zeros((2, 1))
    dist = w.predict_distribution(X)

    assert isinstance(dist, DistributionPrediction)
    # Nodes-as-edges: K levels -> K-1 bins, K edges.
    assert dist.probas.shape == (2, len(w._alphas) - 1)
    # bin_edges can be shared (1-D) or per-sample (2-D)
    expected_edges = len(w._alphas)
    if dist.bin_edges.ndim == 1:
        assert dist.bin_edges.shape[0] == expected_edges
    else:
        assert dist.bin_edges.shape == (2, expected_edges)
