"""Integration tests for the sample-/parametric-distribution wrappers.

Each model is exercised on a small but *learnable* synthetic regression task.
A model only "passes" if it can draw conditional predictions well enough that
its predicted mean tracks the true signal and its CRPS is finite and beats a
trivial constant-variance baseline. Tests skip automatically when the backing
library is absent.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from scoringbench.univariate.metrics import compute_metrics
from scoringbench.univariate.wrappers.base import DistributionPrediction
from scoringbench.univariate.wrappers.quantile_based import quantiles_to_distribution
from scoringbench.univariate.wrappers.sample_based import (
    SampleBasedWrapper,
    samples_to_distribution,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _train_range(values: np.ndarray) -> tuple[float, float]:
    """A valid shared grid range for a block of draws / quantiles.

    ``DistributionPrediction`` requires a finite ``(y_lo, y_hi)`` with
    ``y_hi > y_lo``.  These unit tests only exercise the grid/mass structure --
    they never compare models -- so the global finite min/max of the block is a
    fine range; the guard widens it when every value ties so ``y_hi > y_lo``.
    """
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    lo = float(finite.min()) if finite.size else 0.0
    hi = float(finite.max()) if finite.size else 1.0
    if hi <= lo:
        hi = lo + 1.0
    return (lo, hi)


def _make_data(n_train=400, n_test=120, n_features=5, seed=0):
    rng = np.random.default_rng(seed)
    Xtr = rng.normal(size=(n_train, n_features))
    Xte = rng.normal(size=(n_test, n_features))

    def signal(X):
        return 2.0 * X[:, 0] + np.sin(2.0 * X[:, 1]) - X[:, 2]

    ytr = signal(Xtr) + rng.normal(scale=0.3, size=n_train)
    yte = signal(Xte) + rng.normal(scale=0.3, size=n_test)
    return Xtr.astype(np.float64), ytr, Xte.astype(np.float64), yte


def _validate_distribution(dist: DistributionPrediction, n_test: int):
    assert isinstance(dist, DistributionPrediction)
    assert dist.probas.shape[0] == n_test
    # PMF rows sum to 1.
    np.testing.assert_allclose(dist.probas.sum(axis=1), 1.0, rtol=1e-6, atol=1e-6)
    assert np.all(dist.probas >= -1e-9)
    assert np.all(np.isfinite(dist.bin_edges))
    assert np.all(np.isfinite(dist.bin_midpoints))
    assert np.all(np.isfinite(dist.mean))
    # Edges monotone non-decreasing per sample (per-sample 2-D grid expected).
    if dist.bin_edges.ndim == 2:
        assert np.all(np.diff(dist.bin_edges, axis=1) >= -1e-9)


def _assert_learns(dist: DistributionPrediction, y_test: np.ndarray):
    """Conditional draws are 'good enough': mean tracks signal, CRPS beats baseline."""
    _validate_distribution(dist, len(y_test))
    metrics = compute_metrics(dist, y_test)

    # Predicted mean correlates with the truth.
    r = np.corrcoef(dist.mean, y_test)[0, 1]
    assert r > 0.6, f"predicted mean barely correlates with target (r={r:.3f})"

    crps = metrics["crps"]
    assert np.isfinite(crps), "CRPS is not finite"

    # Trivial baseline: a single wide Gaussian PMF (mean 0, std = std(y_test))
    # encoded on a shared grid — sample-based model should be sharper/better.
    std = float(np.std(y_test)) + 1e-6
    grid = np.linspace(y_test.mean() - 5 * std, y_test.mean() + 5 * std, 200)
    mids = 0.5 * (grid[:-1] + grid[1:])
    from scipy.stats import norm
    pdf = norm.pdf(mids, loc=float(np.mean(y_test)), scale=std)
    probas = np.tile(pdf / pdf.sum(), (len(y_test), 1))
    baseline = DistributionPrediction(
        probas=probas, bin_edges=grid, bin_midpoints=mids,
        mean=np.full(len(y_test), float(np.mean(y_test))),
        train_range=(float(np.asarray(grid).min()), float(np.asarray(grid).max())),
    )
    base_crps = compute_metrics(baseline, y_test)["crps"]
    assert crps < base_crps, f"CRPS {crps:.3f} not better than baseline {base_crps:.3f}"


# ---------------------------------------------------------------------------
# Unit tests for the shared machinery
# ---------------------------------------------------------------------------

def test_quantiles_to_distribution_shapes():
    # Nodes-as-edges: the quantile values ARE the bin edges (no invented tail),
    # with the end CDF values pinned to C = 0 / C = 1.  K levels therefore give
    # K edges and K - 1 bins, and the masses are diff(alphas) renormalised to 1.
    alphas = np.array([0.25, 0.5, 0.75])
    q = np.array([[0.0, 1.0, 2.0], [1.0, 1.5, 4.0]])
    dist = quantiles_to_distribution(q, alphas, train_range=_train_range(q))
    assert dist.probas.shape == (2, len(alphas) - 1)
    assert dist.bin_edges.shape == (2, len(alphas))
    np.testing.assert_allclose(dist.probas.sum(axis=1), 1.0)

    # Edges are the quantiles verbatim -- no tail extension.
    np.testing.assert_allclose(dist.bin_edges, np.sort(q, axis=1), rtol=1e-12)
    # Masses are diff(alphas) renormalised, independent of the quantile values.
    d = np.diff(np.sort(alphas))
    expected = d / d.sum()
    np.testing.assert_allclose(dist.probas, np.broadcast_to(expected, dist.probas.shape), rtol=1e-12)
    # Support is exactly the quantile hull -- no invented tail either side.
    np.testing.assert_allclose(dist.bin_edges[:, 0], q[:, 0])
    np.testing.assert_allclose(dist.bin_edges[:, -1], q[:, -1])


def test_samples_to_distribution_recovers_mean():
    rng = np.random.default_rng(1)
    samples = rng.normal(loc=3.0, scale=1.0, size=(4, 5000))
    dist = samples_to_distribution(samples, n_bins=99, train_range=_train_range(samples))
    _validate_distribution(dist, 4)
    np.testing.assert_allclose(dist.mean, 3.0, atol=0.1)


# ---------------------------------------------------------------------------
# eCDF -> uniform grid: explicit guarantee tests
#
# The native PMF grid is the draws' own hull cut into ``n_bins`` EQUAL-WIDTH bins,
# so every bin has width ``span / n_bins`` -- strictly positive whenever the
# draws are not all identical.  A row whose draws are ALL identical is a genuine
# point mass: its hull collapses and the native PMF grid becomes a Dirac (zero-width
# bins) rather than being widened by an invented, arbitrary pad.  That is the
# correct native representation of an atom -- CRPS and the CDF-based rules score
# a Dirac exactly -- and the density rules read ``.resampled`` (which always has
# positive width from ``train_range``) instead of this grid.
#
# The trade is that a bin may be empty (an atom's mass lands wholly in the one
# bin containing it, leaving its neighbours dry).  That is fine: the mass is a
# valid PMF summing to 1, and no density rule reads this grid directly.
#
# These tests exercise pathological *discrete / heavily tied* draws -- the case
# that collapsed the old adaptive quantile grid -- across a range of n_bins.
# ---------------------------------------------------------------------------

# A gauntlet of adversarial sample rows (each row = draws for one test point).
_TIED_ROWS = [
    np.array([1.0] * 100),                                  # all identical (degenerate)
    np.array([0.0] * 50 + [1.0] * 50),                      # two atoms
    np.repeat(np.array([-2.0, -2.0, 0.0, 5.0, 5.0]), 20),   # 3 unique, tied
    np.array([7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 8.0]),     # near-degenerate
    np.array([-1e6, 0.0, 0.0, 0.0, 1e6]),                   # huge dynamic range + ties
    np.round(np.random.default_rng(3).normal(size=200)),    # integer-rounded (many ties)
]


@pytest.mark.parametrize("n_bins", [2, 10, 50, 100, 257])
@pytest.mark.parametrize("row_idx", range(len(_TIED_ROWS)))
def test_ecdf_grid_bins_positive_unless_the_draws_are_all_identical(row_idx, n_bins):
    """Equal-width bins are strictly positive unless the draws all coincide.

    The native PMF grid is the draws' hull cut into ``n_bins`` equal bins, so widths
    are positive whenever the hull is non-degenerate.  An all-identical row is a
    point mass whose hull collapses; its native PMF grid is a Dirac (zero-width bins)
    -- the correct atom representation, not an invented pad.
    """
    row = _TIED_ROWS[row_idx][None, :]
    dist = samples_to_distribution(row, n_bins=n_bins, train_range=_train_range(row))

    assert dist.bin_edges.shape == (1, n_bins + 1)
    widths = np.diff(dist.bin_edges, axis=1)
    assert np.all(np.isfinite(widths))
    assert np.all(widths >= 0.0)
    if np.unique(row).size > 1:
        assert np.all(widths > 0.0), (
            f"zero-width bin for non-degenerate row {row_idx}, n_bins={n_bins}: "
            f"min width {widths.min():.3e}"
        )
    else:  # all draws identical -> Dirac: the whole hull collapses to a point.
        assert np.all(widths == 0.0), (
            f"all-identical row {row_idx} should collapse to a Dirac"
        )


@pytest.mark.parametrize("n_bins", [2, 10, 50, 100, 257])
@pytest.mark.parametrize("row_idx", range(len(_TIED_ROWS)))
def test_ecdf_grid_is_regular_and_holds_all_the_mass(row_idx, n_bins):
    """Widths are uniform and the masses are a valid PMF summing to exactly 1.

    Empty bins are *allowed* here (that is the equal-width trade), so the
    guarantee is non-negativity plus exact total mass, not positivity: the eCDF
    is anchored at ``C = 0`` / ``C = 1`` at the extreme draws, so no mass leaks
    off the grid however heavily the draws tie.  A fully-degenerate (all
    identical) row collapses to a Dirac: every width is exactly 0 and one bin
    carries all the mass, which is still uniform (target span/n = 0) and valid.
    """
    row = _TIED_ROWS[row_idx][None, :]
    dist = samples_to_distribution(row, n_bins=n_bins, train_range=_train_range(row))

    assert dist.probas.shape == (1, n_bins)
    assert np.all(dist.probas >= 0.0), f"negative mass: {dist.probas.min():.3e}"
    np.testing.assert_allclose(dist.probas.sum(axis=1), 1.0, rtol=1e-12, atol=1e-12)

    # Regularity is checked in *absolute* terms against the resolution of the
    # edge coordinates, not as a relative spread of the widths.  For a
    # degenerate row the hull collapses (target width 0) and every edge sits at
    # the single draw value, so the deviation is 0 to within a linspace ULP.
    edges = dist.bin_edges
    widths = np.diff(edges, axis=1)
    target = (edges[:, -1] - edges[:, 0]) / n_bins
    tol = 4.0 * np.spacing(np.abs(edges).max())
    dev = np.abs(widths - target[:, None]).max()
    assert dev <= tol, (
        f"grid is not regular for row {row_idx}, n_bins={n_bins}: "
        f"max |w - span/n| = {dev:.3e} > {tol:.3e}"
    )


@pytest.mark.parametrize("n_bins", [10, 100])
@pytest.mark.parametrize("row_idx", range(len(_TIED_ROWS)))
def test_ecdf_grid_mass_lands_where_the_draws_are(row_idx, n_bins):
    """An atom's mass ends up in the bin that contains it.

    Equal width means the shape of the draws is carried by the *masses*, so the
    representation is only faithful if mass tracks the draws.  For every distinct
    draw value, the bin bracketing it must carry mass -- and the bins carrying
    mass must together account for essentially all of it.
    """
    row = _TIED_ROWS[row_idx][None, :]
    dist = samples_to_distribution(row, n_bins=n_bins, train_range=_train_range(row))
    edges, probas = dist.bin_edges[0], dist.probas[0]

    # Interior draws (the outermost ones sit on a bin boundary by construction).
    for value in np.unique(row):
        k = int(np.clip(np.searchsorted(edges, value, side="right") - 1, 0, n_bins - 1))
        lo = max(k - 1, 0)
        hi = min(k + 2, n_bins)
        assert probas[lo:hi].sum() > 0.0, (
            f"row {row_idx}: no mass near draw {value} (bin {k})"
        )


def test_ecdf_grid_metrics_finite_on_discrete_draws():
    """Every score (CRPS and the density-power-divergence rules alike) stays
    finite on heavy ties, since the regular-grid PMF has strictly positive bin
    widths (a zero-width or zero-mass bin would blow the density up)."""
    rng = np.random.default_rng(7)
    # 20 test points, 100 draws each, only ~5 unique values -> heavy ties.
    base = np.array([-3.0, -1.0, 0.0, 2.0, 4.0])
    samples = rng.choice(base, size=(20, 100))
    dist = samples_to_distribution(samples, n_bins=64, train_range=_train_range(samples))
    _validate_distribution(dist, 20)

    # y lands exactly on the atoms (the worst case for a fixed-width histogram,
    # where the surrounding bin could be empty).
    y = rng.choice(base, size=20)
    metrics = compute_metrics(dist, y)
    # Density/DPD-based scores are now computed for every prediction and must be
    # finite even on heavy ties.  ``log_score`` was dropped from the metric set
    # (it is sensitive to the reported support); ``dpd_beta_0.01`` is its
    # support-insensitive stand-in and is covered by the ``dpd`` loop below.
    for key in ("crps", "cde_loss"):
        assert np.isfinite(metrics[key]), f"{key} not finite: {metrics[key]}"
    for key, val in metrics.items():
        if key.startswith("dpd"):
            assert np.isfinite(val), f"{key} not finite: {val}"


def test_ecdf_grid_edges_bracket_support():
    """Edges span the observed support (outer bins padded, never inverted)."""
    row = np.array([[2.0, 2.0, 2.0, 5.0, 9.0, 9.0]])
    dist = samples_to_distribution(row, n_bins=8, train_range=_train_range(row))
    edges = dist.bin_edges[0]
    assert edges[0] <= 2.0 < 9.0 <= edges[-1]
    assert np.all(np.diff(edges) > 0.0)


def test_ecdf_grid_energy_score_nonzero_on_tied_draws():
    """Energy score stays finite and positive on discrete draws with tied values.

    Tied values would collapse an adaptive quantile grid to zero-width interior
    bins.  Resampling the eCDF onto a *regular* grid removes the degeneracy at
    the source -- there is no zero width to repair -- so the energy score, which
    reads bin midpoints and widths, stays well defined."""
    # 5 unique values, tied so an adaptive quantile grid collapses interior bins.
    base = np.array([0.0, 0.0, 0.0, 10.0, 10.0])
    samples = np.tile(base, (8, 20))            # (8, 100), heavy ties
    dist = samples_to_distribution(samples, n_bins=50, train_range=_train_range(samples))
    y = np.full(8, 3.0)                          # y away from the atoms
    metrics = compute_metrics(dist, y)

    energy_keys = [k for k in metrics if k.startswith("energy_score")]
    assert energy_keys, "no energy score keys produced"
    for k in energy_keys:
        assert np.isfinite(metrics[k]), f"{k} not finite"
        assert metrics[k] > 0.0, f"{k} == {metrics[k]} (expected > 0)"


def test_sample_based_timeout_is_respected():
    class SlowWrapper(SampleBasedWrapper):
        N_SAMPLES = 1000
        SAMPLE_CHUNK = 50
        MAX_SAMPLE_SECONDS = 0.3

        def _draw_samples(self, X, n_samples):
            time.sleep(0.1)
            return np.zeros((len(X), n_samples))

    w = SlowWrapper()
    X = np.zeros((3, 2))
    t0 = time.monotonic()
    samples = w._collect_samples(X)
    elapsed = time.monotonic() - t0
    # Stopped early: fewer than the full target, and not absurdly over budget.
    assert samples.shape[1] < 1000
    assert elapsed < 2.0

# ---------------------------------------------------------------------------
# Per-model integration tests (skip if library missing)
# ---------------------------------------------------------------------------


def test_nflows_integration():
    pytest.importorskip("nflows")
    pytest.importorskip("torch")
    from scoringbench.univariate.wrappers.nflows_wrapper import NFlowsWrapper

    Xtr, ytr, Xte, yte = _make_data()
    model = NFlowsWrapper(
        n_layers=4, hidden_features=64, num_bins=8,
        n_epochs=300, batch_size=256, lr=1e-3, device="cpu", n_samples=300,
    )
    model.fit(Xtr, ytr)
    dist = model.predict_distribution(Xte)
    _assert_learns(dist, yte)


@pytest.mark.slow
def test_bart_integration():
    pytest.importorskip("pymc")
    pytest.importorskip("pymc_bart")
    from scoringbench.univariate.wrappers.bart_wrapper import BARTWrapper

    Xtr, ytr, Xte, yte = _make_data(n_train=200, n_test=80)
    model = BARTWrapper(num_trees=30, draws=150, tune=150, chains=2, cores=1)
    model.fit(Xtr, ytr)
    dist = model.predict_distribution(Xte)
    _assert_learns(dist, yte)


@pytest.mark.slow
def test_forest_diffusion_integration():
    pytest.importorskip("ForestDiffusion")
    from scoringbench.univariate.wrappers.forest_diffusion_wrapper import ForestDiffusionWrapper

    Xtr, ytr, Xte, yte = _make_data(n_train=200, n_test=60)
    model = ForestDiffusionWrapper(
        n_t=20,
        duplicate_K=100,
        diffusion_type="flow",
        n_estimators=100,
        max_depth=5,
        n_jobs=-1,
        n_samples=100,
        sample_chunk=20,
        random_state=0,
    )
    model.fit(Xtr, ytr)
    dist = model.predict_distribution(Xte)
    _assert_learns(dist, yte)
