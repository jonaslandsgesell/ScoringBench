"""Tests for the synthetic copula-coupled target source.

Two groups:

1. **Generator contract & determinism** — cheap structural checks that the
   ``(X, Y)`` contract matches the feature-promotion source, that the same seed
   reproduces identical arrays byte-for-byte, that different seeds diverge, and
   that the frozen-artifact cache round-trips through parquet.

2. **Independence-failure quality** (the tough tests) — the whole *point* of
   this source is to produce data an independent (product-of-marginals) model
   provably cannot fit. These tests fit the real composition wrappers (via the
   GPU-free ``GaussianMockSampler`` reused from ``test_baselines``) on the
   synthetic data and assert that the **independent** wrapper is beaten by
   **copula** and **chained** wrappers on every dependence-sensitive multivariate
   scoring rule (energy, variogram, Dawid–Sebastiani), and that the gap widens
   as the copula's Kendall tau increases. If an independent model could match a
   copula model here, the synthetic source would be worthless — so these
   assertions guard the source's reason to exist.
"""

from __future__ import annotations

import numpy as np
import pytest

from scoringbench.multivariate import config as cfg
from scoringbench.multivariate import synthetic_targets as st
from scoringbench.multivariate.metrics import compute_scoring_rules
from scoringbench.multivariate.wrappers import (
    ChainedMultiOutputWrapper,
    CopulaMultiOutputWrapper,
    IndependentMultiOutputWrapper,
)

# Reuse the GPU-free, real-PCHIP-path mock sampler from the baseline suite so
# these tests exercise the genuine copula PIT / inverse-PIT machinery.
from .test_baselines import GaussianMockSampler, _Pred  # noqa: E402

N_DRAWS = 150
SEED = 0


def _factory():
    return GaussianMockSampler()


# ===========================================================================
# 1. Generator contract & determinism
# ===========================================================================

def test_enumerate_is_deterministic():
    a = st.enumerate_synthetic(target_dim=3, sample_size=500)
    b = st.enumerate_synthetic(target_dim=3, sample_size=500)
    assert a == b
    # Total is exactly SYNTHETIC_N_DATASETS, distributed across family x tau cells.
    assert len(a) == cfg.SYNTHETIC_N_DATASETS
    # Every (family, tau) cell is represented, replicates differ by at most one.
    from collections import Counter
    per_cell = Counter((c["family"], c["tau"]) for c in a)
    assert len(per_cell) == len(cfg.SYNTHETIC_FAMILIES) * len(cfg.SYNTHETIC_TAUS)
    assert max(per_cell.values()) - min(per_cell.values()) <= 1
    # Names and seeds are unique across the grid.
    assert len({c["name"] for c in a}) == len(a)
    assert len({c["seed"] for c in a}) == len(a)


def test_config_carries_provenance_keys():
    cfgs = st.enumerate_synthetic(target_dim=2, sample_size=100)
    c = cfgs[0]
    for key in ("name", "id", "source", "family", "tau", "seed",
                "n_samples", "n_features", "target_dim"):
        assert key in c
    assert c["source"] == st.SOURCE_NAME


@pytest.mark.parametrize("d", [2, 3, 4])
def test_generate_shape_and_contract(d):
    cfgs = st.enumerate_synthetic(target_dim=d, sample_size=300)
    X, Y = st._generate(cfgs[0], target_dim=d)
    # X non-empty, Y has exactly d targets.
    assert X.shape == (300, cfg.SYNTHETIC_N_FEATURES)
    assert Y.shape == (300, d)
    assert X.shape[1] >= 1
    # target_0 must be the FIRST target column (mirrors promote_features_to_targets).
    assert list(Y.columns)[0] == "target_0"
    assert list(Y.columns) == [f"target_{k}" for k in range(d)]
    assert np.all(np.isfinite(X.values))
    assert np.all(np.isfinite(Y.values))


def test_same_seed_reproduces_identical_arrays():
    cfgs = st.enumerate_synthetic(target_dim=3, sample_size=400)
    X1, Y1 = st._generate(cfgs[0], target_dim=3)
    X2, Y2 = st._generate(cfgs[0], target_dim=3)
    np.testing.assert_array_equal(X1.values, X2.values)
    np.testing.assert_array_equal(Y1.values, Y2.values)


def test_different_seeds_diverge():
    cfgs = st.enumerate_synthetic(target_dim=3, sample_size=400)
    # Same family/tau, different replicate => different seed.
    same_cell = [c for c in cfgs
                 if c["family"] == cfgs[0]["family"] and c["tau"] == cfgs[0]["tau"]]
    assert len(same_cell) >= 2
    _, Ya = st._generate(same_cell[0], target_dim=3)
    _, Yb = st._generate(same_cell[1], target_dim=3)
    assert not np.allclose(Ya.values, Yb.values)


def test_cache_round_trip(tmp_path, monkeypatch):
    """A written parquet artifact loads back to the exact generated arrays.

    Artifacts live in a per-(d, n) subfolder under ``SYNTHETIC_DIR``; the
    generator creates it, so the test creates it explicitly here.
    """
    monkeypatch.setattr(st, "SYNTHETIC_DIR", tmp_path)

    d, n = 3, 250
    cfgs = st.enumerate_synthetic(target_dim=d, sample_size=n)
    ds = cfgs[0]
    X, Y = st._generate(ds, target_dim=d)
    st._shape_subdir(d, n).mkdir(parents=True, exist_ok=True)
    X.join(Y).to_parquet(st._artifact_path(ds["name"], d, n), index=False)

    Xl, Yl = st.load_synthetic(ds, target_dim=d)
    np.testing.assert_array_equal(X.values, Xl.values)
    np.testing.assert_array_equal(Y.values, Yl.values)
    assert list(Yl.columns) == [f"target_{k}" for k in range(d)]


def test_load_raises_on_cache_miss(tmp_path, monkeypatch):
    """The synthetic loader must FAIL LOUDLY on a cache miss (no on-the-fly
    regeneration) and point the user at the generator command."""
    monkeypatch.setattr(st, "SYNTHETIC_DIR", tmp_path)
    ds = st.enumerate_synthetic(target_dim=2, sample_size=120)[0]
    with pytest.raises(FileNotFoundError, match="generate_synthetic.py"):
        st.load_synthetic(ds, target_dim=2)


def test_load_uses_per_shape_subdir(tmp_path, monkeypatch):
    """Artifacts for one (d, n) must not satisfy a load for a different (d, n)."""
    monkeypatch.setattr(st, "SYNTHETIC_DIR", tmp_path)

    d = 2
    ds = st.enumerate_synthetic(target_dim=d, sample_size=120)[0]
    # Freeze the n=120 artifact only.
    X, Y = st._generate(ds, target_dim=d)
    st._shape_subdir(d, 120).mkdir(parents=True, exist_ok=True)
    X.join(Y).to_parquet(st._artifact_path(ds["name"], d, 120), index=False)

    # Same dataset name/family/tau/replicate but a DIFFERENT sample size => a
    # different subfolder => must still be a cache miss (raises).
    ds_other_n = dict(ds, n_samples=250)
    with pytest.raises(FileNotFoundError):
        st.load_synthetic(ds_other_n, target_dim=d)


def test_residuals_are_dependent():
    """The residual copula must inject genuine cross-target dependence.

    We regenerate with tau=0 impossible (grid starts at 0.5), so at the smallest
    grid tau the empirical target correlation must be clearly non-zero for a
    Gaussian family (whose Pearson correlation ~= sin(pi/2 * tau))."""
    cfgs = st.enumerate_synthetic(target_dim=2, sample_size=4000)
    # Smallest tau in the grid, Gaussian family: rho ~= sin(pi/2 * tau).
    gauss = next(c for c in cfgs
                 if c["family"] == "gaussian" and c["tau"] == min(cfg.SYNTHETIC_TAUS))
    _, Y = st._generate(gauss, target_dim=2)
    # Residual dependence dominates the (small, feature-driven) mean, so the raw
    # target correlation is high even at the weakest grid tau.
    corr = np.corrcoef(Y.values[:, 0], Y.values[:, 1])[0, 1]
    assert abs(corr) > 0.6


# ===========================================================================
# 2. Independence-failure quality (tough tests)
# ===========================================================================

def _split(X, Y, n_train):
    return X[:n_train], Y[:n_train], X[n_train:], Y[n_train:]


def _fit_predict(wrapper_cls, X_tr, Y_tr, X_te):
    w = wrapper_cls(sampler_factory=_factory, n_draws=N_DRAWS, seed=SEED)
    w.fit(X_tr, Y_tr)
    return w.predict_ensemble(X_te).samples


def _score_all_modes(family: str, tau: float, n: int = 1400, n_train: int = 1000):
    """Fit all three modes on ONE synthetic (family, tau) dataset.

    The default (n, n_train) leaves 400 held-out points; the Dawid-Sebastiani
    assertion below relies on the ensemble/empirical covariance, whose off-
    diagonal estimate needs enough test points to be stable for the thinner-
    covariance families (e.g. Frank). 400 keeps every family's DSS margin well
    inside the strict thresholds while the variogram margin stays ~0.1.
    """
    cfgs = st.enumerate_synthetic(target_dim=2, sample_size=n)
    ds = next(c for c in cfgs if c["family"] == family and c["tau"] == tau)
    Xdf, Ydf = st._generate(ds, target_dim=2)
    X, Y = Xdf.values, Ydf.values
    X_tr, Y_tr, X_te, Y_te = _split(X, Y, n_train)
    samples = {
        "ind": _fit_predict(IndependentMultiOutputWrapper, X_tr, Y_tr, X_te),
        "cop": _fit_predict(CopulaMultiOutputWrapper, X_tr, Y_tr, X_te),
        "cha": _fit_predict(ChainedMultiOutputWrapper, X_tr, Y_tr, X_te),
    }
    scores = {k: compute_scoring_rules(_Pred(v), Y_te)
              for k, v in samples.items()}
    return scores


# Fit each (family, tau) cell once and reuse across parametrized assertions.
_STRONG_TAU = max(cfg.SYNTHETIC_TAUS)
_WEAK_TAU = min(cfg.SYNTHETIC_TAUS)


@pytest.fixture(scope="module")
def gaussian_strong():
    return _score_all_modes(family="gaussian", tau=_STRONG_TAU)


@pytest.fixture(scope="module")
def clayton_strong():
    return _score_all_modes(family="clayton", tau=_STRONG_TAU)


@pytest.fixture(scope="module")
def gumbel_strong():
    return _score_all_modes(family="gumbel", tau=_STRONG_TAU)


@pytest.fixture(scope="module")
def frank_strong():
    return _score_all_modes(family="frank", tau=_STRONG_TAU)


# Every enumerated family, at the strongest tau, must break independence hard.
_STRONG_FIXTURES = ["gaussian_strong", "clayton_strong",
                    "gumbel_strong", "frank_strong"]


@pytest.mark.parametrize("dgp", _STRONG_FIXTURES)
@pytest.mark.parametrize("mode", ["cop", "cha"])
def test_independent_loses_on_energy(dgp, mode, request):
    """An independent model must score WORSE (higher energy) than copula/chain."""
    s = request.getfixturevalue(dgp)
    assert s[mode]["energy_score_beta_1"] < s["ind"]["energy_score_beta_1"]


@pytest.mark.parametrize("dgp", _STRONG_FIXTURES)
@pytest.mark.parametrize("mode", ["cop", "cha"])
def test_independent_loses_hard_on_variogram(dgp, mode, request):
    """The variogram score isolates pairwise dependence, so with a
    dependence-dominated DGP the independent model is CRUSHED: copula/chained
    must recover the vast majority of the score (>=70% here). This is a strict
    margin, not a mere sign check."""
    s = request.getfixturevalue(dgp)
    ind = s["ind"]["variogram_score_p_0.5"]
    assert s[mode]["variogram_score_p_0.5"] < 0.30 * ind


@pytest.mark.parametrize("dgp", _STRONG_FIXTURES)
@pytest.mark.parametrize("mode", ["cop", "cha"])
def test_independent_loses_hard_on_dawid_sebastiani(dgp, mode, request):
    """DSS uses the ensemble covariance, directly penalising the missing
    off-diagonal that an independent joint zeroes out. In the
    dependence-dominated regime the near-comonotone residual induces a large
    linear covariance for EVERY family (even the tail-concentrated Clayton /
    Gumbel), so a dependence-aware model must beat independent by a wide margin
    (>=25%)."""
    s = request.getfixturevalue(dgp)
    ind = s["ind"]["dawid_sebastiani"]
    assert s[mode]["dawid_sebastiani"] < 0.75 * ind


def test_independent_gap_is_substantial(gaussian_strong):
    """The failure is not marginal: on the strong-dependence dataset the copula
    recovers the overwhelming majority of the variogram score."""
    s = gaussian_strong
    ind = s["ind"]["variogram_score_p_0.5"]
    cop = s["cop"]["variogram_score_p_0.5"]
    # tau=0.9, dependence-dominated => independent is badly misspecified.
    assert cop < 0.20 * ind


def test_gap_widens_with_dependence_strength():
    """Stronger copula dependence => bigger independent-vs-copula gap.

    This proves the failure is *driven by the copula coupling* rather than an
    artifact: increasing Kendall tau must increase how much the independent
    model loses by."""
    weak = _score_all_modes(family="gaussian", tau=_WEAK_TAU)
    strong = _score_all_modes(family="gaussian", tau=_STRONG_TAU)

    def rel_gap(s):
        # Variogram isolates pairwise dependence, so the widening is clean.
        ind = s["ind"]["variogram_score_p_0.5"]
        cop = s["cop"]["variogram_score_p_0.5"]
        return (ind - cop) / ind

    assert rel_gap(strong) > rel_gap(weak)
