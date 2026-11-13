"""Sample forecasts: native view = Hazen eCDF nodes, density view = sqrt(N)-quantile summary.

Two defects of the former equal-width sample route are pinned here:

* the NATIVE view cut the draws' hull ``[min, max]`` into equal-width bins, so a
  single outlying draw coarsened every bin; CRPS and PIT then scored a blurred
  forecast (heavy/skewed tails: +10% .. +2000% CRPS, calibrated PIT-KS rejected);
* the DENSITY view interpolated linearly through every draw, i.e. the 1-spacing
  density estimator, which is inconsistent: ``int f^2`` grows without bound as
  bins shrink, so CDE/DPD drifted with the bin count instead of converging.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from scoringbench.univariate.metrics import _score_one_view, compute_scoring_rules
from scoringbench.univariate.wrappers.base import DistributionPrediction
from scoringbench.univariate.wrappers.resampling_grid import samples_to_quantile_nodes


def _exact_ecdf_crps(samples, y):
    n = samples.shape[1]
    ordered = np.sort(samples, axis=1)
    weights = (2 * np.arange(1, n + 1) - n - 1) / n**2
    return np.abs(samples - y[:, None]).mean(1) - (ordered * weights).sum(1)


def _density(prediction, y):
    return _score_one_view(prediction.resampled, y, compute_density=True, compute_grid_robust=False)


def test_full_hazen_nodes_are_the_sorted_draws():
    draws = np.random.default_rng(0).normal(size=(3, 50))
    support, levels = samples_to_quantile_nodes(draws, 50)
    np.testing.assert_allclose(support, np.sort(draws, axis=1), rtol=0, atol=1e-12)
    np.testing.assert_allclose(levels, (np.arange(50) + 0.5) / 50)


@pytest.mark.parametrize("law", [stats.cauchy(), stats.lognorm(2.0), stats.t(2)],
                         ids=["cauchy", "lognormal", "t2"])
def test_native_crps_tracks_exact_sample_crps_under_heavy_tails(law):
    rng = np.random.default_rng(1)
    y = law.rvs(200, random_state=rng)
    draws = law.rvs((200, 300), random_state=rng)
    prediction = DistributionPrediction.from_samples(draws, train_range=(float(y.min()), float(y.max())))
    crps = compute_scoring_rules(prediction, y, representation="native")["crps"]
    exact = _exact_ecdf_crps(draws, y).mean()
    assert crps == pytest.approx(exact, rel=0.01)


def test_calibrated_skewed_samples_are_not_rejected_by_pit():
    rng = np.random.default_rng(2)
    law = stats.lognorm(2.0)
    y = law.rvs(400, random_state=rng)
    draws = law.rvs((400, 300), random_state=rng)
    prediction = DistributionPrediction.from_samples(draws, train_range=(float(y.min()), float(y.max())))
    result = compute_scoring_rules(prediction, y, representation="native")
    assert result["pit_ks_pvalue"] > 0.01


@pytest.mark.parametrize("n_draws", [100, 300, 1000])
def test_sample_density_scores_converge_instead_of_diverging_with_bins(n_draws):
    rng = np.random.default_rng(3)
    y = rng.normal(size=400)
    draws = rng.normal(size=(400, n_draws))
    truth = 1.0 / (2.0 * np.sqrt(np.pi)) - 2.0 * stats.norm.pdf(y).mean()
    cde = []
    for bins in (100, 400, 1600):
        prediction = DistributionPrediction.from_samples(
            draws, train_range=(-4.0, 4.0), num_equally_sized_bins=bins)
        cde.append(_density(prediction, y)["cde_loss"])
    assert max(cde) - min(cde) < 0.03
    assert abs(cde[-1] - truth) < 0.08


def test_tied_draws_keep_atoms_natively_and_a_finite_density():
    draws = np.tile(np.array([0.0, 0.0, 0.0, 10.0, 10.0]), (4, 20))
    y = np.array([0.0, 3.0, 10.0, 12.0])
    prediction = DistributionPrediction.from_samples(draws, train_range=(-1.0, 11.0))
    result = compute_scoring_rules(prediction, y)
    assert all(np.isfinite(v) for v in result.values())
    np.testing.assert_allclose(prediction.native.probas.sum(axis=1), 1.0, atol=1e-12)
    assert result["crps"] == pytest.approx(_exact_ecdf_crps(draws, y).mean(), rel=0.02)
