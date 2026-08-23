"""Correlation-structure tests for the three composition modes.

These tests verify that the *copula* and *chained* wrappers recover
cross-target dependence that the *independent* wrapper structurally cannot, and
that this shows up as a lower (better) multivariate score.

Why a mock sampler
------------------
The real ``TabPFNSampler`` / ``TabICLSampler`` require a GPU that is
incompatible with the installed PyTorch build on this node (any ``.to('cuda')``
raises a "no kernel image" error).  So we plug the composition wrappers into a
tiny **covariate-aware Gaussian mock** that subclasses
:class:`~scoringbench.multivariate.wrappers.base_sampler.BaseSampler` and
implements *only* ``_row_cdf_grid``.  Crucially this means ``cdf`` / ``quantile``
/ ``sample`` all come from the *real* base-class monotone-PCHIP machinery — the
exact code path used in production — so the tests exercise the genuine copula
PIT / inverse-PIT plumbing, not a stand-in.

The mock's marginals depend only on ``X`` (as the copula wrapper requires).  The
data-generating process adds *residual* cross-target correlation that ``X`` does
not explain, so:

* **independent** must miss it (its joint is a product of marginals), while
* **copula** recovers it through the vine copula on the PITs, and
* **chained** recovers it through the product rule ``p(y1|x) p(y2|x, y1)``.

Speed
-----
The base class builds one PCHIP per row, so cost scales with
``n_test * n_draws``.  Tests keep those small (``N_DRAWS`` draws, coarse PCHIP
grid, modest test sets) — the dependence signal at ``|rho| = 0.9`` is huge, so
a light Monte-Carlo budget is plenty to separate the modes.  Fitted ensembles
are cached per DGP via module-scoped fixtures so each wrapper is scored once.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.stats import norm

from scoringbench.multivariate.metrics import compute_scoring_rules
from scoringbench.multivariate.wrappers import (
    ChainedMultiOutputWrapper,
    CopulaMultiOutputWrapper,
    IndependentMultiOutputWrapper,
)
from scoringbench.multivariate.wrappers.base_sampler import BaseSampler

# Light Monte-Carlo budget: |rho|=0.9 gives a large, easily-detected signal.
N_DRAWS = 150
N_NODES = 60
SEED = 0


# ---------------------------------------------------------------------------
# Covariate-aware Gaussian mock sampler (GPU-free, real PCHIP path)
# ---------------------------------------------------------------------------

class GaussianMockSampler(BaseSampler):
    """Conditional-Gaussian sampler exposed only through a CDF grid.

    Fits ``E[y | X] = X @ beta`` by least squares and a homoscedastic residual
    std, then presents the resulting conditional-Gaussian CDF to the base class
    as ``(alpha, quantile)`` nodes.  ``cdf`` / ``quantile`` / ``sample`` are then
    the base class's PCHIP derivations of that grid.
    """

    def __init__(self, n_nodes: int = N_NODES):
        self._device = torch.device("cpu")
        self._alphas = np.linspace(1e-3, 1 - 1e-3, n_nodes)
        # Standard-normal quantiles of the grid, cached once (row-independent).
        self._z = norm.ppf(self._alphas)
        self._beta = None
        self._sd = 1.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        A = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        self._beta = coef
        resid = y - A @ coef
        self._sd = float(max(np.std(resid), 1e-3))
        return self

    def predict_mean(self, X):
        X = np.asarray(X, dtype=np.float64)
        A = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        return A @ self._beta

    def _row_cdf_grid(self, X):
        mu = self.predict_mean(X)
        # Value grid = mu_i + sd * z (vectorised); alphas shared across rows.
        v = mu[:, None] + self._sd * self._z[None, :]
        c_rows = [self._alphas] * len(mu)
        v_rows = [v[i] for i in range(len(mu))]
        return c_rows, v_rows


def _factory():
    return GaussianMockSampler()


# ---------------------------------------------------------------------------
# Data-generating process
# ---------------------------------------------------------------------------

def _make_correlated_data(n, rho, seed):
    """Two targets sharing residual correlation NOT explained by X.

    ``mu_k(x)`` is linear in X (so the mock marginals are well-specified) and
    the residuals ``(e0, e1)`` are jointly Gaussian with correlation ``rho``.
    The residual correlation is orthogonal to X, so an X-only marginal model
    cannot recover it — only the joint structure (copula / chain) can.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    mu0 = 1.5 * X[:, 0] - 0.5 * X[:, 1]
    mu1 = -1.0 * X[:, 0] + 2.0 * X[:, 1]
    L = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))
    e = rng.normal(size=(n, 2)) @ L.T
    Y = np.stack([mu0 + e[:, 0], mu1 + e[:, 1]], axis=1)
    return X, Y


def _split(X, Y, n_train):
    return X[:n_train], Y[:n_train], X[n_train:], Y[n_train:]


def _fit_predict(wrapper_cls, X_tr, Y_tr, X_te, **kw):
    w = wrapper_cls(sampler_factory=_factory, n_draws=N_DRAWS, seed=SEED, **kw)
    w.fit(X_tr, Y_tr)
    return w.predict_ensemble(X_te).samples


def _mean_conditional_corr(S):
    """Mean over test rows of the per-row cross-target sample correlation."""
    return float(np.mean([np.corrcoef(S[i, :, 0], S[i, :, 1])[0, 1]
                          for i in range(S.shape[0])]))


class _Pred:
    """Minimal duck-typed stand-in exposing ``.samples`` for the metrics."""

    __slots__ = ("samples",)

    def __init__(self, samples):
        self.samples = np.asarray(samples, dtype=np.float64)


def _fit_all_modes(rho, seed, n=220, n_train=150):
    """Fit all three modes on one DGP, returning cached samples + scores."""
    X, Y = _make_correlated_data(n=n, rho=rho, seed=seed)
    X_tr, Y_tr, X_te, Y_te = _split(X, Y, n_train)
    S = {
        "ind": _fit_predict(IndependentMultiOutputWrapper, X_tr, Y_tr, X_te),
        "cop": _fit_predict(CopulaMultiOutputWrapper, X_tr, Y_tr, X_te),
        "cha": _fit_predict(ChainedMultiOutputWrapper, X_tr, Y_tr, X_te),
    }
    scores = {k: compute_scoring_rules(_Pred(v), Y_te) for k, v in S.items()}
    return {"samples": S, "scores": scores}


# ---------------------------------------------------------------------------
# Shared fixtures: fit each mode ONCE per DGP and reuse across assertions.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def strong_pos():
    """Strong positive residual dependence (rho=+0.9)."""
    return _fit_all_modes(rho=0.9, seed=1)


@pytest.fixture(scope="module")
def strong_neg():
    """Strong negative residual dependence (rho=-0.9)."""
    return _fit_all_modes(rho=-0.9, seed=2)


@pytest.fixture(scope="module")
def independent_truth():
    """No residual dependence (rho=0)."""
    return _fit_all_modes(rho=0.0, seed=3)


# ---------------------------------------------------------------------------
# 1. The independent joint is (near) a product of marginals
# ---------------------------------------------------------------------------

def test_independent_has_no_conditional_dependence(strong_pos):
    """Independent draws have ~zero conditional cross-target correlation."""
    assert abs(_mean_conditional_corr(strong_pos["samples"]["ind"])) < 0.1


def test_copula_recovers_conditional_dependence(strong_pos):
    """Copula draws reproduce the positive residual dependence."""
    assert _mean_conditional_corr(strong_pos["samples"]["cop"]) > 0.5


def test_chained_recovers_conditional_dependence(strong_pos):
    """Chained draws also reproduce the positive residual dependence."""
    assert _mean_conditional_corr(strong_pos["samples"]["cha"]) > 0.5


def test_copula_recovers_negative_dependence(strong_neg):
    """Copula recovers the sign of the dependence (rho=-0.9)."""
    assert _mean_conditional_corr(strong_neg["samples"]["cop"]) < -0.5


def test_chained_recovers_negative_dependence(strong_neg):
    assert _mean_conditional_corr(strong_neg["samples"]["cha"]) < -0.5


# ---------------------------------------------------------------------------
# 2. Superiority on dependence-sensitive scoring rules
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["cop", "cha"])
@pytest.mark.parametrize("dgp", ["strong_pos", "strong_neg"])
def test_beats_independent_on_energy(mode, dgp, request):
    s = request.getfixturevalue(dgp)["scores"]
    assert s[mode]["energy_score_beta_1"] < s["ind"]["energy_score_beta_1"]


@pytest.mark.parametrize("mode", ["cop", "cha"])
def test_beats_independent_on_variogram(mode, strong_pos):
    s = strong_pos["scores"]
    assert s[mode]["variogram_score_p_0.5"] < s["ind"]["variogram_score_p_0.5"]


@pytest.mark.parametrize("mode", ["cop", "cha"])
def test_beats_independent_on_dawid_sebastiani(mode, strong_pos):
    """DSS uses the ensemble covariance, so it directly penalises the missing
    off-diagonal term of the independent joint."""
    s = strong_pos["scores"]
    assert s[mode]["dawid_sebastiani"] < s["ind"]["dawid_sebastiani"]


# ---------------------------------------------------------------------------
# 3. No spurious dependence when the truth is independent
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["cop", "cha"])
def test_no_penalty_when_targets_independent(mode, independent_truth):
    """With truly independent residuals (rho=0), copula/chained must not be
    meaningfully worse than independent — they must not invent dependence."""
    s = independent_truth["scores"]
    base = s["ind"]["energy_score_beta_1"]
    assert s[mode]["energy_score_beta_1"] < 1.10 * base


# ---------------------------------------------------------------------------
# 4. Shape / invariant sanity for all three modes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "wrapper_cls",
    [IndependentMultiOutputWrapper, CopulaMultiOutputWrapper, ChainedMultiOutputWrapper],
)
def test_ensemble_shape_and_finiteness(wrapper_cls):
    X, Y = _make_correlated_data(n=120, rho=0.7, seed=6)
    X_tr, Y_tr, X_te, Y_te = _split(X, Y, 90)
    S = _fit_predict(wrapper_cls, X_tr, Y_tr, X_te)
    assert S.shape == (X_te.shape[0], N_DRAWS, Y.shape[1])
    assert np.all(np.isfinite(S))


@pytest.mark.parametrize(
    "wrapper_cls",
    [IndependentMultiOutputWrapper, CopulaMultiOutputWrapper, ChainedMultiOutputWrapper],
)
def test_reproducible_given_seed(wrapper_cls):
    X, Y = _make_correlated_data(n=120, rho=0.7, seed=6)
    X_tr, Y_tr, X_te, Y_te = _split(X, Y, 90)
    a = _fit_predict(wrapper_cls, X_tr, Y_tr, X_te)
    b = _fit_predict(wrapper_cls, X_tr, Y_tr, X_te)
    np.testing.assert_allclose(a, b)
