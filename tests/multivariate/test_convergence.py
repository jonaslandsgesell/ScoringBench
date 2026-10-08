"""Tests for the multivariate MPSRF convergence diagnostic.

Covers:
* ``compute_multivariate_rhat`` on a known multivariate Gaussian DGP with
  elongated covariance (eigenvalues [10, 1, 1]) — R̂ should be close to 1
  when n is large and well above 1 when n is tiny.
* ``assess_sample_prediction`` on a MultivariateSamplePrediction drawn from
  the same DGP.
* ``adaptive_sample_until_converged`` terminates and returns R̂ < epsilon.
* The adaptive wiring in ``SampleBasedWrapper`` actually draws more samples
  when the initial batch is too small to converge.
"""

from __future__ import annotations

import numpy as np
import pytest

from scoringbench.multivariate.convergence import (
    ConvergenceInfo,
    SampleDiagnostic,
    adaptive_sample_until_converged,
    assess_sample_prediction,
    compute_multivariate_rhat,
)
from scoringbench.multivariate.prediction import MultivariateSamplePrediction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _elongated_cov(p: int = 3, stretch: float = 10.0) -> np.ndarray:
    """Covariance with eigenvalues [stretch, 1, 1, …, 1]."""
    cov = np.eye(p)
    cov[0, 0] = stretch
    return cov


def _draw_gaussian_groups(
    m: int, n: int, p: int, cov: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Return ``(m, n, p)`` i.i.d. draws from N(0, cov), split into m groups."""
    total = m * n
    samples = rng.multivariate_normal(np.zeros(p), cov, size=total)
    return samples.reshape(m, n, p)


def _make_pred(
    n_test: int, m: int, p: int, cov: np.ndarray, rng: np.random.Generator
) -> MultivariateSamplePrediction:
    """``(n_test, m, p)`` draws from N(0, cov) as a MultivariateSamplePrediction."""
    samples = np.stack(
        [rng.multivariate_normal(np.zeros(p), cov, size=m) for _ in range(n_test)],
        axis=0,
    )  # (n_test, m, p)
    return MultivariateSamplePrediction(samples=samples)


# ---------------------------------------------------------------------------
# compute_multivariate_rhat
# ---------------------------------------------------------------------------

class TestComputeMultivariateRhat:
    """Unit tests for the det-form MPSRF statistic."""

    def test_converges_to_one_large_n(self):
        """With many i.i.d. draws R̂ should be very close to 1."""
        rng = np.random.default_rng(0)
        cov = _elongated_cov(p=3, stretch=10.0)
        groups = _draw_gaussian_groups(m=4, n=5000, p=3, cov=cov, rng=rng)
        rhat = compute_multivariate_rhat(groups)
        assert rhat < 1.01, f"Expected R̂ < 1.01 with n=5000; got {rhat:.4f}"

    def test_above_one_small_n(self):
        """With very few draws the between-group variance dominates → R̂ > 1."""
        rng = np.random.default_rng(1)
        cov = _elongated_cov(p=3, stretch=10.0)
        groups = _draw_gaussian_groups(m=4, n=5, p=3, cov=cov, rng=rng)
        rhat = compute_multivariate_rhat(groups)
        # Not guaranteed to be huge, but should be > 1 in expectation.
        assert rhat >= 1.0, f"R̂ should be >= 1; got {rhat:.4f}"

    def test_scalar_case(self):
        """p=1 should reduce to the classic scalar Gelman–Rubin statistic."""
        rng = np.random.default_rng(2)
        groups = rng.normal(size=(4, 2000, 1))
        rhat = compute_multivariate_rhat(groups)
        assert rhat < 1.01, f"Scalar R̂ should be ~1 with n=2000; got {rhat:.4f}"

    def test_elongated_vs_isotropic(self):
        """Elongated covariance should not prevent convergence with enough draws."""
        rng = np.random.default_rng(3)
        p = 5
        cov_iso = np.eye(p)
        cov_elo = _elongated_cov(p=p, stretch=100.0)
        n = 3000
        rhat_iso = compute_multivariate_rhat(
            _draw_gaussian_groups(4, n, p, cov_iso, rng)
        )
        rhat_elo = compute_multivariate_rhat(
            _draw_gaussian_groups(4, n, p, cov_elo, rng)
        )
        assert rhat_iso < 1.01, f"Isotropic R̂={rhat_iso:.4f}"
        assert rhat_elo < 1.01, f"Elongated R̂={rhat_elo:.4f}"

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError, match="3-D"):
            compute_multivariate_rhat(np.ones((4, 10)))

    def test_too_few_groups_raises(self):
        with pytest.raises(ValueError, match="m=2"):
            compute_multivariate_rhat(np.ones((1, 10, 3)))

    def test_too_few_samples_raises(self):
        with pytest.raises(ValueError, match="n=2"):
            compute_multivariate_rhat(np.ones((4, 1, 3)))

    def test_nonfinite_raises(self):
        groups = np.ones((4, 10, 3))
        groups[0, 0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            compute_multivariate_rhat(groups)


# ---------------------------------------------------------------------------
# assess_sample_prediction
# ---------------------------------------------------------------------------

class TestAssessSamplePrediction:
    """Tests for the per-instance MPSRF summary over a sample prediction."""

    def test_converged_large_m(self):
        """With m=1000 draws R̂ should be well below 1.01."""
        rng = np.random.default_rng(10)
        cov = _elongated_cov(p=3, stretch=10.0)
        pred = _make_pred(n_test=50, m=1000, p=3, cov=cov, rng=rng)
        diag = assess_sample_prediction(pred, n_groups=4, epsilon=1.01)
        assert isinstance(diag, SampleDiagnostic)
        assert diag.rhat < 1.01, f"median R̂={diag.rhat:.4f}"
        assert diag.rhat_max < 1.05, f"max R̂={diag.rhat_max:.4f}"
        assert diag.frac_converged > 0.9, f"frac_converged={diag.frac_converged:.3f}"
        assert diag.method == "det"
        assert diag.n_draws == 1000

    def test_not_converged_small_m(self):
        """With m=20 draws (5 per group) frac_converged should be low."""
        rng = np.random.default_rng(11)
        cov = _elongated_cov(p=3, stretch=10.0)
        pred = _make_pred(n_test=50, m=20, p=3, cov=cov, rng=rng)
        diag = assess_sample_prediction(pred, n_groups=4, epsilon=1.01)
        # With only 5 draws per group the between-group variance is noisy;
        # most instances should NOT be flagged as converged.
        assert diag.frac_converged < 0.5, (
            f"Expected low frac_converged with m=20; got {diag.frac_converged:.3f}"
        )

    def test_max_instances_subsampling(self):
        """max_instances subsampling must not crash and should return a valid diag."""
        rng = np.random.default_rng(12)
        cov = np.eye(2)
        pred = _make_pred(n_test=200, m=400, p=2, cov=cov, rng=rng)
        diag = assess_sample_prediction(pred, n_groups=4, max_instances=30)
        assert 0.0 <= diag.frac_converged <= 1.0

    def test_too_few_draws_raises(self):
        """m too small to form n_groups groups of >= 2 must raise ValueError."""
        rng = np.random.default_rng(13)
        pred = _make_pred(n_test=10, m=7, p=2, cov=np.eye(2), rng=rng)
        with pytest.raises(ValueError, match="not enough draws"):
            assess_sample_prediction(pred, n_groups=4)  # needs m >= 4*2 = 8


# ---------------------------------------------------------------------------
# adaptive_sample_until_converged
# ---------------------------------------------------------------------------

class TestAdaptiveSampleUntilConverged:
    """Tests for the standalone adaptive sampling loop."""

    def _make_dgp(self, cov: np.ndarray):
        """Return a sample_dgp callable for the given covariance."""
        rng = np.random.default_rng(99)
        p = cov.shape[0]

        def dgp(m: int, n: int, p_: int) -> np.ndarray:
            assert p_ == p
            return rng.multivariate_normal(np.zeros(p), cov, size=m * n).reshape(m, n, p)

        return dgp

    def test_converges_on_gaussian(self):
        """Adaptive loop must converge on a well-behaved Gaussian."""
        p = 3
        cov = _elongated_cov(p=p, stretch=10.0)
        dgp = self._make_dgp(cov)
        groups, info = adaptive_sample_until_converged(
            dgp, p=p, n_groups=4, n_start=200, batch_size=200,
            epsilon=1.01, max_samples=20_000, verbose=False,
        )
        assert isinstance(info, ConvergenceInfo)
        assert info.converged, (
            f"Expected convergence; last R̂={info.rhat:.4f} after {info.n_samples} samples"
        )
        assert info.rhat < 1.01
        assert groups.shape[0] == 4
        assert groups.shape[2] == p
        assert len(info.history) >= 1

    def test_history_is_monotone_n(self):
        """n_samples in history must be strictly increasing."""
        p = 2
        cov = np.eye(p)
        dgp = self._make_dgp(cov)
        _, info = adaptive_sample_until_converged(
            dgp, p=p, n_groups=4, n_start=100, batch_size=100,
            epsilon=1.01, max_samples=5_000, verbose=False,
        )
        ns = [h[0] for h in info.history]
        assert ns == sorted(ns), f"history n_samples not monotone: {ns}"

    def test_max_samples_respected(self):
        """Loop must stop at max_samples even if not converged."""
        p = 2
        # Degenerate DGP: all zeros — covariance is singular, R̂ may be noisy
        # but the loop must still terminate.
        def dgp(m, n, p_):
            return np.zeros((m, n, p_))

        _, info = adaptive_sample_until_converged(
            dgp, p=p, n_groups=4, n_start=50, batch_size=50,
            epsilon=1.01, max_samples=200, verbose=False,
        )
        assert info.n_samples <= 200


# ---------------------------------------------------------------------------
# Adaptive wiring in SampleBasedWrapper
# ---------------------------------------------------------------------------

class _ToyDGPWrapper:
    """Minimal SampleBasedWrapper-like object for testing the adaptive loop.

    Draws i.i.d. samples from a multivariate Gaussian; tracks total draws.
    """

    def __init__(self, cov: np.ndarray, n_test: int = 20):
        from scoringbench.multivariate.wrappers.sample_based import SampleBasedWrapper

        self._cov = cov
        self._p = cov.shape[0]
        self._n_test = n_test
        self._rng = np.random.default_rng(42)
        self.draw_count = 0

        class _Impl(SampleBasedWrapper):
            def fit(self_, X, y=None):
                return self_

            def _draw_samples(self_, X, n_samples):
                self.draw_count += n_samples
                return self._rng.multivariate_normal(
                    np.zeros(self._p), self._cov,
                    size=(self._n_test * n_samples),
                ).reshape(self._n_test, n_samples, self._p)

        self._impl = _Impl()
        self._impl.fit(np.zeros((n_test, 1)))

    def predict_ensemble(self, X):
        return self._impl.predict_ensemble(X)


class TestAdaptiveWiringSampleBased:
    """Integration tests: SampleBasedWrapper draws more when R̂ not converged."""

    def test_draws_more_when_not_converged(self, monkeypatch):
        """With a tiny N_SAMPLES the wrapper should top up to convergence."""
        import scoringbench.multivariate.wrappers.sample_based as sb_mod
        import scoringbench.multivariate.config as cfg_mod
        import scoringbench.multivariate.convergence as conv_mod

        # Force a small initial target so we definitely need top-ups.
        monkeypatch.setattr(sb_mod, "N_DRAWS", 40)
        monkeypatch.setattr(cfg_mod, "CONV_ENABLED", True)
        # Patch the DEFAULT_* constants in convergence (and their aliases in the
        # wrapper module) so the test runs fast with a small cap.
        monkeypatch.setattr(conv_mod, "DEFAULT_EPSILON", 1.01)
        monkeypatch.setattr(conv_mod, "DEFAULT_N_GROUPS", 4)
        monkeypatch.setattr(conv_mod, "DEFAULT_MAX_INSTANCES", 20)
        monkeypatch.setattr(conv_mod, "DEFAULT_MAX_SAMPLES", 5000)
        monkeypatch.setattr(conv_mod, "DEFAULT_BATCH_SIZE", 200)
        monkeypatch.setattr(sb_mod, "DEFAULT_EPSILON", 1.01)
        monkeypatch.setattr(sb_mod, "DEFAULT_N_GROUPS", 4)
        monkeypatch.setattr(sb_mod, "DEFAULT_MAX_INSTANCES", 20)
        monkeypatch.setattr(sb_mod, "DEFAULT_MAX_SAMPLES", 5000)
        monkeypatch.setattr(sb_mod, "DEFAULT_BATCH_SIZE", 200)

        p = 3
        cov = _elongated_cov(p=p, stretch=10.0)
        toy = _ToyDGPWrapper(cov=cov, n_test=20)
        toy._impl.N_SAMPLES = 40
        toy._impl.SAMPLE_CHUNK = 40

        X = np.zeros((20, 1))
        pred = toy.predict_ensemble(X)

        # The wrapper should have drawn more than the initial 40.
        assert toy.draw_count > 40, (
            f"Expected adaptive top-up; total draws = {toy.draw_count}"
        )
        # The final ensemble should be substantially converged.
        # We use rhat_max < 1.05 (not 1.01) because the post-hoc re-evaluation
        # uses a fresh shuffle RNG and only 20 instances, so the worst-case
        # instance can land slightly above the 1.01 stopping threshold by chance.
        diag = assess_sample_prediction(
            pred, n_groups=4, epsilon=1.01, max_instances=20
        )
        assert diag.rhat_max < 1.05, (
            f"Expected R̂_max < 1.05 after adaptive top-up; got {diag.rhat_max:.4f}"
        )

    def test_no_topup_when_disabled(self, monkeypatch):
        """When CONV_ENABLED=False the wrapper must return exactly N_SAMPLES draws."""
        import scoringbench.multivariate.config as cfg_mod

        monkeypatch.setattr(cfg_mod, "CONV_ENABLED", False)

        p = 2
        cov = np.eye(p)
        toy = _ToyDGPWrapper(cov=cov, n_test=10)
        toy._impl.N_SAMPLES = 100
        toy._impl.SAMPLE_CHUNK = 100

        X = np.zeros((10, 1))
        pred = toy.predict_ensemble(X)
        assert pred.samples.shape[1] == 100, (
            f"Expected exactly 100 draws when disabled; got {pred.samples.shape[1]}"
        )
