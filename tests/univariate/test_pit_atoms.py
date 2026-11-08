"""PIT at probability atoms is the randomized PIT of Czado et al. (2009).

    u = F(y-) + V * (F(y) - F(y-)),   V ~ Uniform(0, 1),

which is exactly Uniform(0, 1) under an ideal forecast even when the forecast
has atoms, and equals F(y) for continuous forecasts.  V comes from a fixed-seed
generator so the benchmark stays reproducible.
"""
import numpy as np
import pytest
import torch

from scoringbench.univariate import metrics as M

# name -> (edges, bin masses, y, F(y-), F(y))
ATOM_CASES = {
    "atom_asymmetric": ([0.0, 0.5, 0.5, 1.0], [0.1, 0.4, 0.5], 0.5, 0.1, 0.5),
    "two_coincident_atoms": ([0.0, 0.5, 0.5, 0.5, 1.0], [0.1, 0.2, 0.2, 0.5], 0.5, 0.1, 0.5),
    "point_mass": ([0.5, 0.5], [1.0], 0.5, 0.0, 1.0),
    "atom_at_upper_support_end": ([0.0, 1.0, 1.0], [0.6, 0.4], 1.0, 0.6, 1.0),
    "atom_at_lower_support_end": ([0.0, 0.0, 1.0], [0.3, 0.7], 0.0, 0.0, 0.3),
}


def _run(edges, masses, y, n, monkeypatch):
    """Call compute_pit_ks on ``n`` identical rows; return (PIT values, stat, p-value)."""
    captured = {}
    real_kstest = M.stats.kstest

    def spy(x, *args, **kwargs):
        captured["pit"] = np.asarray(x)
        return real_kstest(x, *args, **kwargs)

    monkeypatch.setattr(M.stats, "kstest", spy)
    e = torch.tensor(edges, dtype=torch.float64)
    p = torch.tensor([masses] * n, dtype=torch.float64)
    ys = y if torch.is_tensor(y) else torch.full((n,), float(y), dtype=torch.float64)
    out = M.compute_pit_ks(p, torch.cumsum(p, dim=1), e, e[1:] - e[:-1], None, ys, True, None)
    return captured["pit"], out["pit_ks_stat"], out["pit_ks_pvalue"]


@pytest.mark.parametrize("name", ATOM_CASES)
def test_pit_at_atom_is_uniform_on_the_cdf_jump(name, monkeypatch):
    edges, masses, y, lo, hi = ATOM_CASES[name]
    pit, _, _ = _run(edges, masses, y, 4000, monkeypatch)
    assert pit.min() >= lo - 1e-12 and pit.max() <= hi + 1e-12
    if hi > lo:
        assert pit.mean() == pytest.approx((lo + hi) / 2, abs=0.02 * (hi - lo) + 1e-3)
        assert pit.std() == pytest.approx((hi - lo) / np.sqrt(12.0), rel=0.05)
        assert pit.min() < lo + 0.02 * (hi - lo) and pit.max() > hi - 0.02 * (hi - lo)


def test_pit_is_deterministic(monkeypatch):
    edges, masses, y, _, _ = ATOM_CASES["atom_asymmetric"]
    a, _, _ = _run(edges, masses, y, 200, monkeypatch)
    b, _, _ = _run(edges, masses, y, 200, monkeypatch)
    np.testing.assert_array_equal(a, b)


def test_pit_continuous_interior_equals_cdf(monkeypatch):
    pit, _, _ = _run([0.0, 0.5, 1.0], [0.5, 0.5], 0.25, 50, monkeypatch)
    np.testing.assert_allclose(pit, 0.25, atol=1e-12)


def test_pit_outside_support_is_clamped(monkeypatch):
    below, _, _ = _run([0.0, 1.0], [1.0], -3.0, 20, monkeypatch)
    above, _, _ = _run([0.0, 1.0], [1.0], 3.0, 20, monkeypatch)
    np.testing.assert_array_equal(below, 0.0)
    np.testing.assert_array_equal(above, 1.0)


def test_ideal_atomic_forecast_gives_uniform_pit(monkeypatch):
    """y drawn from the forecast itself (atoms at 0, 1, 2) must not be rejected;
    the midpoint PIT would sit on three values and be rejected."""
    rng = np.random.default_rng(12345)  # must differ from the PIT randomization seed
    probs = np.array([0.2, 0.5, 0.3])
    n = 4000
    y = torch.tensor(rng.choice([0.0, 1.0, 2.0], size=n, p=probs), dtype=torch.float64)
    edges = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0]
    masses = [0.2, 0.0, 0.5, 0.0, 0.3]
    _, _, pvalue = _run(edges, masses, y, n, monkeypatch)
    assert pvalue > 0.01


# ---------------------------------------------------------------------------
# Closed-form expectations for the PIT distribution
# ---------------------------------------------------------------------------
# If the truth Y has law Q and the forecast F is atomic with atoms p_k at x_k, then
# u | (Y = x_k) ~ Uniform(F(x_k-), F(x_k)), so the PIT CDF is the piecewise linear
#     G(t) = sum_k q_k * clip((t - F(x_k-)) / p_k, 0, 1).
# The empirical CDF of N draws stays within the DKW bound of G with prob. >= 1 - ALPHA.
N_DRAWS = 40_000
ALPHA = 1e-6
DKW_EPS = np.sqrt(np.log(2.0 / ALPHA) / (2.0 * N_DRAWS))
T_GRID = np.linspace(0.0, 1.0, 2001)


def _atomic_forecast(p):
    """Histogram of atoms at 0, 1, ..., len(p)-1 (zero-width bins) with masses ``p``."""
    k = len(p)
    edges = [0.0]
    masses = []
    for i in range(k):
        edges += [float(i), float(i)] if i > 0 else [0.0]
        masses += [0.0, p[i]] if i > 0 else [p[i]]
    return edges, masses


def _analytic_pit_cdf(p, q):
    cum = np.concatenate([[0.0], np.cumsum(p)])
    g = np.zeros_like(T_GRID)
    for k in range(len(p)):
        g += q[k] * np.clip((T_GRID - cum[k]) / p[k], 0.0, 1.0)
    return g


def _empirical_cdf(pit):
    return np.searchsorted(np.sort(pit), T_GRID, side="right") / len(pit)


@pytest.mark.parametrize(
    "p, q",
    [
        ([0.2, 0.5, 0.3], [0.2, 0.5, 0.3]),            # ideal -> G(t) = t
        ([0.2, 0.5, 0.3], [0.5, 0.3, 0.2]),            # miscalibrated
        ([0.25, 0.25, 0.25, 0.25], [0.7, 0.1, 0.1, 0.1]),
    ],
)
def test_atomic_forecast_pit_cdf_matches_closed_form(p, q, monkeypatch):
    rng = np.random.default_rng(12345)
    y = torch.tensor(rng.choice(len(p), size=N_DRAWS, p=q), dtype=torch.float64)
    edges, masses = _atomic_forecast(p)
    pit, stat, _ = _run(edges, masses, y, N_DRAWS, monkeypatch)

    g = _analytic_pit_cdf(p, q)
    assert np.abs(_empirical_cdf(pit) - g).max() < DKW_EPS
    # the reported KS statistic is the analytic distance to Uniform(0, 1)
    assert stat == pytest.approx(np.abs(g - T_GRID).max(), abs=2 * DKW_EPS)


def test_ideal_atomic_forecast_pit_has_uniform_moments(monkeypatch):
    p = [0.2, 0.5, 0.3]
    rng = np.random.default_rng(12345)
    y = torch.tensor(rng.choice(3, size=N_DRAWS, p=p), dtype=torch.float64)
    pit, _, _ = _run(*_atomic_forecast(p), y, N_DRAWS, monkeypatch)
    # Uniform(0, 1): mean 1/2, variance 1/12; standard errors ~ 1/sqrt(12 N), 0.3/sqrt(N)
    assert pit.mean() == pytest.approx(0.5, abs=5 / np.sqrt(12 * N_DRAWS))
    assert pit.var() == pytest.approx(1 / 12, abs=0.3 * 5 / np.sqrt(N_DRAWS))


def test_mixed_atom_and_continuous_forecast(monkeypatch):
    """Forecast: atom of mass 0.3 at 0 plus Uniform(0, 1) with mass 0.7."""
    edges, masses = [0.0, 0.0, 1.0], [0.3, 0.7]
    rng = np.random.default_rng(12345)

    # ideal truth -> PIT ~ Uniform(0, 1)
    at_zero = rng.random(N_DRAWS) < 0.3
    y = torch.tensor(np.where(at_zero, 0.0, rng.random(N_DRAWS)), dtype=torch.float64)
    pit, _, _ = _run(edges, masses, y, N_DRAWS, monkeypatch)
    assert np.abs(_empirical_cdf(pit) - T_GRID).max() < DKW_EPS

    # purely continuous truth Uniform(0, 1): u = 0.3 + 0.7 y ~ Uniform(0.3, 1)
    y = torch.tensor(rng.random(N_DRAWS), dtype=torch.float64)
    pit, _, _ = _run(edges, masses, y, N_DRAWS, monkeypatch)
    g = np.clip((T_GRID - 0.3) / 0.7, 0.0, 1.0)
    assert np.abs(_empirical_cdf(pit) - g).max() < DKW_EPS


def test_continuous_forecast_pit_cdf_matches_closed_form(monkeypatch):
    """Forecast Uniform(0, 1), truth Uniform(0, 0.5): u = y ~ Uniform(0, 0.5)."""
    rng = np.random.default_rng(12345)
    y = torch.tensor(0.5 * rng.random(N_DRAWS), dtype=torch.float64)
    pit, _, _ = _run([0.0, 1.0], [1.0], y, N_DRAWS, monkeypatch)
    g = np.minimum(2.0 * T_GRID, 1.0)
    assert np.abs(_empirical_cdf(pit) - g).max() < DKW_EPS


# ---------------------------------------------------------------------------
# Gaussian forecasts (continuous CDF, analytic PIT)
# ---------------------------------------------------------------------------
# Forecast N(0, 1) discretised on a fine histogram.  The piecewise-linear CDF of a
# bin of width h is off by <= max|phi'| h^2 / 8 ~ 3e-5 here, so a tolerance of 1e-4
# on top of the DKW bound separates implementation error from interpolation error.
from scipy.stats import norm  # noqa: E402

N_GAUSS = 20_000
DKW_GAUSS = np.sqrt(np.log(2.0 / ALPHA) / (2.0 * N_GAUSS))
INTERP_TOL = 1e-4
GAUSS_GRID = np.linspace(-6.0, 6.0, 401)


def _gaussian_histogram():
    masses = np.diff(norm.cdf(GAUSS_GRID))
    return list(GAUSS_GRID), list(masses / masses.sum())


def test_gaussian_forecast_pit_equals_normal_cdf_pointwise(monkeypatch):
    """Continuous forecast: no randomization, u = Phi(y) for every observation."""
    y_np = np.random.default_rng(12345).uniform(-5.0, 5.0, size=2000)
    edges, masses = _gaussian_histogram()
    pit, _, _ = _run(edges, masses, torch.tensor(y_np), len(y_np), monkeypatch)
    np.testing.assert_allclose(pit, norm.cdf(y_np), atol=INTERP_TOL)


@pytest.mark.parametrize("mu, sigma", [(0.0, 1.0), (0.7, 1.3), (-0.5, 0.6)])
def test_gaussian_forecast_pit_cdf_matches_closed_form(mu, sigma, monkeypatch):
    """Truth N(mu, sigma^2) vs forecast N(0, 1):
    P(Phi(Y) <= t) = Phi((Phi^{-1}(t) - mu) / sigma)."""
    y = np.random.default_rng(12345).normal(mu, sigma, size=N_GAUSS)
    edges, masses = _gaussian_histogram()
    pit, stat, _ = _run(edges, masses, torch.tensor(y), N_GAUSS, monkeypatch)

    t = np.clip(T_GRID, 1e-12, 1 - 1e-12)
    g = norm.cdf((norm.ppf(t) - mu) / sigma)
    assert np.abs(_empirical_cdf(pit) - g).max() < DKW_GAUSS + INTERP_TOL
    assert stat == pytest.approx(np.abs(g - T_GRID).max(), abs=2 * DKW_GAUSS)


def _censored_gaussian_histogram():
    """Forecast max(0, Z), Z ~ N(0, 1): atom of mass 1/2 at 0 plus the density on (0, 6]."""
    grid = np.linspace(0.0, 6.0, 301)
    edges = [0.0, 0.0] + list(grid[1:])
    masses = [norm.cdf(0.0)] + list(np.diff(norm.cdf(grid)))
    return edges, masses


def test_censored_gaussian_forecast_ideal_truth_gives_uniform_pit(monkeypatch):
    z = np.random.default_rng(12345).normal(size=N_GAUSS)
    y = torch.tensor(np.maximum(z, 0.0))
    pit, _, _ = _run(*_censored_gaussian_histogram(), y, N_GAUSS, monkeypatch)
    assert np.abs(_empirical_cdf(pit) - T_GRID).max() < DKW_GAUSS + INTERP_TOL


def test_censored_gaussian_forecast_shifted_truth_matches_closed_form(monkeypatch):
    """Truth max(0, Z + s): atom of mass Phi(-s) at 0 maps to u ~ Uniform(0, 1/2);
    for y > 0, u = Phi(y).  Hence G(t) = 2 t Phi(-s) for t <= 1/2 and
    G(t) = Phi(Phi^{-1}(t) - s) for t > 1/2."""
    s = 0.5
    z = np.random.default_rng(12345).normal(size=N_GAUSS)
    y = torch.tensor(np.maximum(z + s, 0.0))
    pit, _, _ = _run(*_censored_gaussian_histogram(), y, N_GAUSS, monkeypatch)

    t = np.clip(T_GRID, 1e-12, 1 - 1e-12)
    g = np.where(t <= 0.5, 2.0 * t * norm.cdf(-s), norm.cdf(norm.ppf(t) - s))
    assert np.abs(_empirical_cdf(pit) - g).max() < DKW_GAUSS + INTERP_TOL
