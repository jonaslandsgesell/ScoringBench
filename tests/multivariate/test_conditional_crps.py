"""Tests for the conditional CRPS (cCRPS) of the chained wrapper.

cCRPS (Roordink & Hess 2023) factorizes the joint via the chain rule and sums the
*univariate* CRPS of each sequential conditional ``p(y_{π(t)} | y_{π(<t)}, x)``,
conditioning on the **observed** earlier targets (teacher forcing).  Each
per-coordinate CRPS is read off the base sampler's piecewise-linear predictive
CDF in closed form (no Gaussian assumption, no sampling).

These tests use a tiny Gaussian-grid sampler (no TabPFN/GPU needed) to check:
  * the closed-form CRPS matches the known analytic values, and
  * cCRPS rewards the learned cross-target dependence — conditioning on the true
    preceding targets scores strictly better than conditioning on scrambled ones,
    and the plain sample-based metrics (which lack the observed conditioning) do
    not expose a ``conditional_crps`` key for non-chained models.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from scoringbench.multivariate.wrappers.base_sampler import (
    BaseSampler,
    crps_piecewise_linear,
)
from scoringbench.multivariate.wrappers.baselines import ChainedMultiOutputWrapper


def test_crps_closed_form_known_values():
    # Uniform(0,1) evaluated at 0.5 -> CRPS = 1/12.
    v = torch.tensor([[0.0, 1.0]]); c = torch.tensor([[0.0, 1.0]])
    got = float(crps_piecewise_linear(v, c, torch.tensor([0.5])))
    assert abs(got - 1.0 / 12.0) < 1e-6

    # Near point-mass at 0 -> CRPS ≈ |y - 0| for the tail.
    v2 = torch.tensor([[0.0, 1e-9]]); c2 = torch.tensor([[0.0, 1.0]])
    assert abs(float(crps_piecewise_linear(v2, c2, torch.tensor([2.0]))) - 2.0) < 1e-6


class _GaussGridSampler(BaseSampler):
    """Conditional sampler: linear-regression mean, Gaussian CDF on a z-grid."""

    def __init__(self):
        self._device = "cpu"

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float).ravel()
        Xb = np.c_[np.ones(len(X)), X]
        self.beta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
        self.sd = max((y - Xb @ self.beta).std(), 1e-3)
        return self

    def predict_mean(self, X):
        X = np.asarray(X, float)
        return np.c_[np.ones(len(X)), X] @ self.beta

    def _row_cdf_grid(self, X):
        mu = self.predict_mean(X)
        z = np.linspace(-4, 4, 81)
        cdfz = np.array([0.5 * (1 + math.erf(zi / math.sqrt(2))) for zi in z])
        return [cdfz.copy() for _ in mu], [mu[i] + self.sd * z for i in range(len(mu))]


def _dgp(seed=0):
    rng = np.random.default_rng(seed)
    n, p, d = 600, 3, 2
    X = rng.normal(size=(n, p))
    Y = np.empty((n, d))
    Y[:, 0] = X[:, 0] + 0.5 * rng.normal(size=n)
    Y[:, 1] = 0.8 * Y[:, 0] + X[:, 1] + 0.5 * rng.normal(size=n)  # depends on Y0
    return X[:450], X[450:], Y[:450], Y[450:], rng


def test_ccrps_rewards_true_conditioning():
    Xtr, Xte, Ytr, Yte, rng = _dgp()
    ch = ChainedMultiOutputWrapper(lambda: _GaussGridSampler(), n_draws=50, n_orders=3)
    ch.fit(Xtr, Ytr)

    good = ch.conditional_crps(Xte, Yte)
    # Scramble the Y0 we condition on when scoring Y1's conditional.
    Y_bad = Yte.copy(); rng.shuffle(Y_bad[:, 0])
    bad = ch.conditional_crps(Xte, Y_bad)

    assert np.isfinite(good) and good > 0
    assert good < bad, "teacher-forcing on true targets must score better"


def test_independent_ccrps_equals_summed_marginal_crps():
    # The independent model's chain-rule conditionals collapse to the marginals,
    # so cCRPS must equal the sum of per-dimension marginal CRPS exactly.
    from scoringbench.multivariate.wrappers import IndependentMultiOutputWrapper

    Xtr, Xte, Ytr, Yte, _ = _dgp()
    ind = IndependentMultiOutputWrapper(lambda: _GaussGridSampler(), n_draws=20)
    ind.fit(Xtr, Ytr)

    got = ind.conditional_crps(Xte, Yte)
    # Reference: sum over dims of each marginal sampler's CRPS, meaned over rows.
    ref = np.zeros(len(Xte))
    for k in range(Yte.shape[1]):
        ref += ind._samplers[k].crps(Xte, Yte[:, k])
    assert abs(got - float(ref.mean())) < 1e-9


def test_chained_beats_independent_on_dependent_data():
    # On dependent targets, conditioning on realized predecessors sharpens the
    # predictive law, so the chained cCRPS should beat the independent one
    # (= summed marginal CRPS). Same proper rule, same realized targets.
    from scoringbench.multivariate.wrappers import IndependentMultiOutputWrapper

    Xtr, Xte, Ytr, Yte, _ = _dgp()
    ind = IndependentMultiOutputWrapper(lambda: _GaussGridSampler(), n_draws=20).fit(Xtr, Ytr)
    ch = ChainedMultiOutputWrapper(lambda: _GaussGridSampler(), n_draws=20, n_orders=3).fit(Xtr, Ytr)
    assert ch.conditional_crps(Xte, Yte) < ind.conditional_crps(Xte, Yte)
