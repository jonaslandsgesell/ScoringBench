"""Explicit CREPES contracts, using the real library and lightweight regressors."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

crepes = pytest.importorskip("crepes")
from crepes.extras import DifficultyEstimator, MondrianCategorizer
from crepes.base import calculate_crps
from scoringbench.univariate.metrics import COVERAGE_LEVELS, ENERGY_BETAS, compute_scoring_rules
from scoringbench.univariate.wrappers.crepes_wrapper import CrepesWrapper


def data(n=500):
    rng = np.random.default_rng(12)
    x = rng.uniform(-1, 1, (n, 2))
    y = 2 * x[:, 0] + (0.2 + (x[:, 1] + 1) / 2) * rng.normal(size=n)
    return x, y


@pytest.mark.parametrize("mondrian", [False, True])
@pytest.mark.parametrize("dataframe", [False, True])
def test_calibration_is_held_out_from_all_fitted_components(monkeypatch, mondrian, dataframe):
    x, y = data()
    if dataframe:
        x = pd.DataFrame(x, index=np.arange(len(x)) * 3 + 7)
    observed = {}
    originals = (DifficultyEstimator.fit, MondrianCategorizer.fit, crepes.WrapRegressor.calibrate)

    def de_fit(self, X=None, *args, **kwargs):
        observed["de_x"], observed["de_y"] = np.asarray(X), kwargs["y"]
        return originals[0](self, X, *args, **kwargs)

    def mc_fit(self, X=None, *args, **kwargs):
        observed["mc_x"] = np.asarray(X)
        return originals[1](self, X, *args, **kwargs)

    def calibrate(self, X, y, **kwargs):
        observed["cal_x"], observed["cal_y"] = np.asarray(X), y
        return originals[2](self, X, y, **kwargs)

    monkeypatch.setattr(DifficultyEstimator, "fit", de_fit)
    monkeypatch.setattr(MondrianCategorizer, "fit", mc_fit)
    monkeypatch.setattr(crepes.WrapRegressor, "calibrate", calibrate)
    m = CrepesWrapper(LinearRegression(), random_state=0,
                      use_mondrian_categorizer=mondrian, mondrian_no_bins=3).fit(x, y)
    xt, xc, yt, yc = train_test_split(x, y, test_size=0.2, random_state=0)
    np.testing.assert_array_equal(observed["de_x"], xt)
    np.testing.assert_array_equal(observed["de_y"], yt)
    np.testing.assert_array_equal(observed["cal_x"], xc)
    np.testing.assert_array_equal(observed["cal_y"], yc)
    if mondrian:
        np.testing.assert_array_equal(observed["mc_x"], xt)
    assert m.predict_distribution(x[:4]).probas.shape[0] == 4


def test_changing_calibration_labels_cannot_change_difficulty_or_bins():
    x, y = data()
    _, cal = train_test_split(np.arange(len(y)), test_size=0.2, random_state=0)
    changed = y.copy()
    changed[cal] += np.linspace(-100, 100, len(cal))
    models = [CrepesWrapper(LinearRegression(), random_state=0,
                           use_mondrian_categorizer=True, mondrian_no_bins=3).fit(x, t)
              for t in (y, changed)]
    np.testing.assert_allclose(models[0].base_model.predict(x), models[1].base_model.predict(x))
    np.testing.assert_allclose(models[0]._difficulty_estimator.apply(x),
                               models[1]._difficulty_estimator.apply(x))
    np.testing.assert_allclose(models[0]._mondrian_categorizer.bin_thresholds,
                               models[1]._mondrian_categorizer.bin_thresholds)


@pytest.mark.parametrize("n", [20, 30, 100])
def test_small_training_sets_bound_neighbor_count(n):
    x, y = data(n)
    m = CrepesWrapper(DummyRegressor(), random_state=0).fit(x, y)
    assert m._difficulty_estimator.k == min(25, int(0.8 * n))
    dist = m.predict_distribution(x[:3])
    assert np.isfinite(dist.mean).all()
    np.testing.assert_allclose(dist.probas.sum(axis=1), 1)


def test_mondrian_requires_difficulty_instead_of_silently_disabling():
    with pytest.raises(ValueError, match="difficulty"):
        CrepesWrapper(DummyRegressor(), use_difficulty_estimator=False,
                      use_mondrian_categorizer=True)


@pytest.mark.parametrize("kwargs", [
    {"calibration_split": 0}, {"calibration_split": 1},
    {"mondrian_no_bins": 0}, {"mondrian_no_bins": 2.5},
])
def test_invalid_configuration_fails_explicitly(kwargs):
    with pytest.raises(ValueError):
        CrepesWrapper(DummyRegressor(), **kwargs)


@pytest.mark.parametrize("difficulty,mondrian", [(False, False), (True, False), (True, True)])
def test_cpds_match_manual_calibration_residual_construction(difficulty, mondrian):
    x, y = data()
    m = CrepesWrapper(LinearRegression(), random_state=0, use_difficulty_estimator=difficulty,
                      use_mondrian_categorizer=mondrian, mondrian_no_bins=3).fit(x, y)
    _, xc, _, yc = train_test_split(x, y, test_size=0.2, random_state=0)
    query = x[:11]
    sc = m._difficulty_estimator.apply(xc) if difficulty else np.ones(len(xc))
    sq = m._difficulty_estimator.apply(query) if difficulty else np.ones(len(query))
    residuals = (yc - m.base_model.predict(xc)) / sc
    state = np.random.get_state()
    try:
        # Replay CREPES's seed-controlled randomized tie-breaking.
        np.random.seed(0)
        bc = m._mondrian_categorizer.apply(xc) if mondrian else np.zeros(len(xc))
        np.random.seed(0)
        bq = m._mondrian_categorizer.apply(query) if mondrian else np.zeros(len(query))
    finally:
        np.random.set_state(state)
    actual = m._wrapped_model.predict_cpds(query)
    for i, prediction in enumerate(m.base_model.predict(query)):
        expected = prediction + sq[i] * np.sort(residuals[bc == bq[i]])
        np.testing.assert_allclose(actual[i], expected)
    dist = m.predict_distribution(query)
    np.testing.assert_allclose(dist.probas.sum(axis=1), 1)
    np.testing.assert_allclose(m.predict(query), dist.mean)
    np.testing.assert_allclose(dist.mean, [np.mean(row) for row in actual])
    scores = compute_scoring_rules(dist, y[:len(query)])
    reference = np.mean([
        calculate_crps(np.asarray(row)[None, :], np.sort(row), np.ones(1), y[i:i+1])
        for i, row in enumerate(actual)
    ])
    assert scores["crps"] == pytest.approx(reference, abs=1e-12)


@pytest.mark.parametrize("cpds", [
    np.array([[1., 2., 4.], [3., 3., 5.]]),
    [np.array([1., 2.]), np.array([3., 3., 5.])],
    [np.array([3.])],
    np.array([1., 2., 4.]),
    [np.array([4., 1., 2., 1.]), np.repeat(3., 7)],
])
def test_cpd_conversion_preserves_every_atom_and_its_mass(cpds):
    rows = [cpds] if isinstance(cpds, np.ndarray) and cpds.ndim == 1 else cpds
    m = CrepesWrapper(DummyRegressor())
    m._wrapped_model = SimpleNamespace(predict_cpds=lambda X: cpds)
    m._set_train_range(np.array([0., 10.]))
    dist = m.predict_distribution(np.zeros((len(rows), 2)))
    assert dist.probas.shape[0] == len(rows)
    np.testing.assert_allclose(dist.probas.sum(axis=1), 1)
    np.testing.assert_allclose(dist.mean, [np.mean(row) for row in rows])
    for i, row in enumerate(rows):
        support, counts = np.unique(row, return_counts=True)
        positive = dist.probas[i] > 0
        np.testing.assert_array_equal(dist.bin_edges[i, :-1][positive], support)
        np.testing.assert_array_equal(dist.bin_edges[i, 1:][positive], support)
        np.testing.assert_allclose(dist.probas[i, positive], counts / len(row))
        probes = np.sort(np.concatenate((support - 0.01, support, support + 0.01)))
        actual_cdf = (dist.probas[i] * (
            dist.bin_edges[i, 1:][None, :] <= probes[:, None])).sum(axis=1)
        expected_cdf = (np.asarray(row)[None, :] <= probes[:, None]).mean(axis=1)
        np.testing.assert_allclose(actual_cdf, expected_cdf)
    assert not dist.is_grid_native
    assert np.isfinite(dist.resampled.probas).all()
    assert (np.diff(dist.resampled.bin_edges, axis=-1) > 0).all()
    np.testing.assert_allclose(dist.resampled.probas.sum(axis=1), 1)


@pytest.mark.parametrize("cpds", [
    [np.array([]), np.array([1., 2.])],
    [np.array([np.nan]), np.array([1., 2.])],
    [np.array([np.inf]), np.array([1., 2.])],
    [None, np.array([1., 2.])],
    [np.array([1., 2.])],
    [np.array(["invalid"]), np.array([1., 2.])],
    None,
])
def test_invalid_cpd_does_not_become_a_fake_finite_distribution(cpds):
    m = CrepesWrapper(DummyRegressor())
    m._wrapped_model = SimpleNamespace(predict_cpds=lambda X: cpds)
    m._set_train_range(np.array([0., 10.]))
    with pytest.raises(ValueError, match="CPD"):
        m.predict_distribution(np.zeros((2, 2)))


def test_mondrian_unseen_calibration_group_fails_explicitly():
    x, y = data(20)
    m = CrepesWrapper(DummyRegressor(), random_state=0,
                      use_mondrian_categorizer=True, mondrian_no_bins=10).fit(x, y)
    with pytest.raises(ValueError, match="Mondrian group"):
        m.predict_distribution(x)


def test_seeded_mondrian_ties_are_reproducible_and_preserve_numpy_rng():
    x, y = data(30)
    state = np.random.get_state()
    models = [
        CrepesWrapper(DummyRegressor(), random_state=0,
                      use_mondrian_categorizer=True, mondrian_no_bins=2).fit(x, y)
        for _ in range(2)
    ]
    first = models[0].predict_distribution(x).mean
    np.testing.assert_array_equal(first, models[0].predict_distribution(x).mean)
    np.testing.assert_array_equal(first, models[1].predict_distribution(x).mean)
    after = np.random.get_state()
    assert state[0] == after[0]
    np.testing.assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]


def test_unfitted_and_failed_refit_do_not_predict_stale_distribution():
    x, y = data(100)
    m = CrepesWrapper(DummyRegressor())
    with pytest.raises(ValueError, match="not fitted"):
        m.predict_distribution(x[:1])
    m.fit(x, y)
    with pytest.raises(ValueError):
        m.fit(x[:1], y[:1])
    with pytest.raises(ValueError, match="not fitted"):
        m.predict_distribution(x[:1])


@pytest.mark.parametrize("values", [
    np.arange(1., 11.), np.array([1., 2., 2., 4.]), np.array([3.]),
])
def test_scores_and_coverage_use_the_cpd_distribution(values):
    targets = np.arange(0., 11., 0.5)
    cpds = np.tile(values, (len(targets), 1))
    m = CrepesWrapper(DummyRegressor())
    m._wrapped_model = SimpleNamespace(predict_cpds=lambda X: cpds)
    m._set_train_range(np.array([0., 10.]))
    dist = m.predict_distribution(np.zeros((len(targets), 2)))
    scores = compute_scoring_rules(dist, targets)
    expected_crps = calculate_crps(cpds, values, np.ones(len(targets)), targets)
    assert scores["crps"] == pytest.approx(expected_crps, abs=1e-12)
    for beta in ENERGY_BETAS:
        expected = (np.abs(values[None, :] - targets[:, None]) ** beta).mean()
        expected -= 0.5 * (np.abs(values[:, None] - values[None, :]) ** beta).mean()
        assert scores[f"energy_score_beta_{beta}"] == pytest.approx(expected, abs=1e-12)
    for level in COVERAGE_LEVELS:
        alpha = 1 - level / 100
        low, high = np.quantile(values, [alpha / 2, 1 - alpha / 2], method="inverted_cdf")
        expected_coverage = np.mean((targets >= low) & (targets <= high))
        expected_score = np.mean(high - low + 2 / alpha * (
            np.maximum(low - targets, 0) + np.maximum(targets - high, 0)))
        assert scores[f"coverage_{level}"] == pytest.approx(expected_coverage)
        assert scores[f"interval_score_{level}"] == pytest.approx(expected_score)
