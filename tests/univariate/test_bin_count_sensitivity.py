"""Bin-count sensitivity of the density view, stratified by native output type.

A fast regression version of the Q2(c) study (rebuttal, Reviewer tJHH): the
forecast *is* the true law, written as a native histogram, as quantiles or as
draws, so every deviation from the exact continuous-density score is a
representation error.  Pinned claims:

1. native (grid) histograms are never regridded: density scores do not depend on B;
2. realistic quantile heads (99 levels) are within 0.5% of the exact CDE / DPD
   at the production B = 200, and the error shrinks from B = 25 to B = 200;
3. DPD_0.01 stays nearly representation-free (tail mass spread to the support edge);
4. PIT-KS lives on the native view, so it is identical for every B and close to the
   KS statistic of the exact CDF.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from scipy import stats

from scoringbench.univariate.metrics import _score_one_view, compute_pit_ks
from scoringbench.univariate.wrappers.base import DistributionPrediction

ROWS = 256
PRODUCTION_B = 200


class _Bimodal:
    """0.55 N(-1.3, 0.25^2) + 0.45 N(1.0, 0.55^2): two modes of different width."""

    w, a, b = 0.55, stats.norm(-1.3, 0.25), stats.norm(1.0, 0.55)

    def pdf(self, x):
        return self.w * self.a.pdf(x) + (1 - self.w) * self.b.pdf(x)

    def cdf(self, x):
        return self.w * self.a.cdf(x) + (1 - self.w) * self.b.cdf(x)

    def ppf(self, u):
        grid = np.linspace(-5.0, 5.0, 20001)
        return np.interp(u, self.cdf(grid), grid)


def _law(name):
    return _Bimodal() if name == "bimodal" else stats.norm(0.0, 0.75)


def _setup(name):
    law = _law(name)
    rng = np.random.default_rng(7)
    y = law.ppf(rng.random(ROWS))
    train = law.ppf(rng.random(2048))
    return law, y, (float(train.min()), float(train.max()))


def _exact(law, y):
    grid = np.linspace(-6.0, 6.0, 120001)
    f = law.pdf(grid)
    out = {}
    for beta in (0.01, 0.5, 1.0):
        out[f"dpd_beta_{beta}"] = float((np.trapezoid(f ** (1 + beta), grid)
                                         - (1 + 1 / beta) * law.pdf(y) ** beta).mean())
    return out


def _forecast(law, kind, y, train_range):
    if kind == "histogram_5000":
        lo, hi = train_range
        pad = 0.1 * (hi - lo)
        edges = np.linspace(lo - pad, hi + pad, 5001)
        masses = np.diff(law.cdf(edges))
        masses = np.tile(masses / masses.sum(), (y.size, 1))
        return DistributionPrediction.from_histogram(edges, masses, train_range=train_range,
                                                     is_grid_native=True)
    levels = np.arange(1, 100) / 100
    q = np.tile(law.ppf(levels), (y.size, 1))
    return DistributionPrediction.from_multi_quantile(q, levels, train_range=train_range)


def _density_scores(prediction, y, bins):
    view = replace(prediction, num_equally_sized_bins=bins).resampled
    return _score_one_view(view, y, compute_density=True, compute_grid_robust=False)


def _pit_ks(prediction, y, bins):
    """Randomized PIT-KS statistic as computed by the benchmark, on the native view."""
    view = replace(prediction, num_equally_sized_bins=bins).native
    probas = torch.as_tensor(np.atleast_2d(view.probas), dtype=torch.float64)
    edges = torch.as_tensor(np.asarray(view.bin_edges), dtype=torch.float64)
    rows = probas.shape[0]
    return float(compute_pit_ks(probas, probas.cumsum(dim=-1), edges, edges.diff(dim=-1),
                                torch.zeros(rows, dtype=torch.long),
                                torch.as_tensor(y, dtype=torch.float64), edges.ndim == 1,
                                torch.arange(rows))["pit_ks_stat"])


def _rel(got, exact):
    return abs(got - exact) / abs(exact)


@pytest.mark.parametrize("name", ["smooth", "bimodal"])
def test_native_histogram_density_scores_do_not_depend_on_bin_count(name):
    law, y, train_range = _setup(name)
    prediction = _forecast(law, "histogram_5000", y, train_range)
    reference = _density_scores(prediction, y, PRODUCTION_B)
    for bins in (25, 1600):
        scores = _density_scores(prediction, y, bins)
        for metric in ("cde_loss", "dpd_beta_0.01", "pseudospherical_alpha_3.0"):
            assert scores[metric] == reference[metric]


@pytest.mark.parametrize("name", ["smooth", "bimodal"])
def test_quantile_density_scores_are_converged_at_production_bin_count(name):
    law, y, train_range = _setup(name)
    exact = _exact(law, y)
    prediction = _forecast(law, "quantiles_99", y, train_range)
    coarse = _density_scores(prediction, y, 25)
    production = _density_scores(prediction, y, PRODUCTION_B)
    assert _rel(production["cde_loss"], exact["dpd_beta_1.0"]) < 0.005
    assert _rel(production["dpd_beta_0.5"], exact["dpd_beta_0.5"]) < 0.005
    assert _rel(production["dpd_beta_0.01"], exact["dpd_beta_0.01"]) < 0.005
    assert _rel(production["cde_loss"], exact["dpd_beta_1.0"]) < _rel(coarse["cde_loss"], exact["dpd_beta_1.0"])


@pytest.mark.parametrize("kind", ["histogram_5000", "quantiles_99"])
def test_pit_ks_is_native_and_independent_of_bin_count(kind):
    law, y, train_range = _setup("bimodal")
    prediction = _forecast(law, kind, y, train_range)
    oracle = stats.kstest(law.cdf(y), "uniform").statistic
    stats_by_bins = [_pit_ks(prediction, y, b) for b in (25, PRODUCTION_B, 1600)]
    assert stats_by_bins[0] == stats_by_bins[1] == stats_by_bins[2]
    assert abs(stats_by_bins[1] - oracle) < 0.02
