import math

import numpy as np

from scoringbench import __version__
from scoringbench.metrics import compute_point_metrics
from scoringbench.wrappers import DistributionPrediction


def test_scoringbench_imports():
    # basic smoke-test: package imports and exposes a version
    assert isinstance(__version__, str)


def test_compute_point_metrics_basic():
    y_true = np.array([0.0, 1.0])
    y_pred = np.array([0.0, 2.0])

    res = compute_point_metrics(y_true, y_pred)

    assert math.isclose(res["mae"], 0.5, rel_tol=1e-9)
    assert math.isclose(res["rmse"], math.sqrt(0.5), rel_tol=1e-9)
    # For this simple example R^2 is -1.0
    assert math.isclose(res["r2"], -1.0, rel_tol=1e-9)


def test_distribution_prediction_container():
    # Ensure DistributionPrediction dataclass accepts arrays and exposes fields
    probas = np.array([[0.5, 0.5], [0.2, 0.8]])
    bin_edges = np.array([0.0, 0.5, 1.0])
    bin_mids = np.array([0.25, 0.75])
    mean = np.array([0.25, 0.8])

    dist = DistributionPrediction(probas=probas, bin_edges=bin_edges, bin_midpoints=bin_mids, mean=mean)

    assert dist.probas.shape == (2, 2)
    assert dist.bin_edges.shape[0] == 3
    assert dist.bin_midpoints.shape == (2,)
    assert dist.mean.shape == (2,)
