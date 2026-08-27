"""Contract for grid-native models (``is_grid_native=True``).

Background
----------
``test_dual_prediction.py`` established that a prediction carries two views:

* ``native``     -- the raw model grid, read by the grid-robust rules.
* ``resampled``  -- the density-rule view.  For an ORDINARY model this is the
  prediction PCHIP-resampled onto the shared common grid (``num_equally_sized_bins`` bins
  over ``train_range``), so the width-dividing density rules are comparable.

Some models own a FIXED predictive grid (e.g. TabPFN's bar-distribution
borders).  Forcing that grid onto the coarse common grid would blur away the
density resolution the head actually provides.  ``is_grid_native=True`` marks
such a model: the density view IS the native PMF grid, verbatim -- no resample and
no train-range widening.  Scoring pads every grid to cover the observed target
on its own (``metrics.pad_to_common_grid``), so no support widening is needed at
construction time and the density resolution is preserved exactly.

What these tests pin down
-------------------------
* the flag defaults to ``False`` and TabPFN sets it ``True``;
* the resampled view is the native PMF grid VERBATIM (same edges, same masses),
  regardless of how the native PMF grid relates to the train range;
* density rules are read off the native PMF grid, so they are invariant to the
  ``train_range`` a grid-native model is handed (they are NOT for an ordinary
  model);
* grid-robust rules are unaffected by the flag (they always read the native
  view verbatim).
"""

import numpy as np
import pytest

from scoringbench.univariate.metrics import (
    DENSITY_BASED_KEYS,
    compute_scoring_rules,
)
from scoringbench.univariate.wrappers import DistributionPrediction


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _make(edges, probas, train_range, *, is_grid_native, num_equally_sized_bins=200):
    """Build a DistributionPrediction from a shared 1-D grid + per-row PMF."""
    edges = np.asarray(edges, dtype=float)
    probas = np.atleast_2d(np.asarray(probas, dtype=float))
    mids = 0.5 * (edges[:-1] + edges[1:])
    mean = (probas * mids[None, :]).sum(axis=-1)
    return DistributionPrediction(
        probas=probas,
        bin_edges=edges,
        bin_midpoints=mids,
        mean=mean,
        train_range=train_range,
        num_equally_sized_bins=num_equally_sized_bins,
        is_grid_native=is_grid_native,
    )


# A native PMF grid deliberately much WIDER than the train range on both sides
# (mirrors TabPFN's bar borders vs. the train-target hull).
WIDE_EDGES = np.array([-10.0, -1.0, -0.25, 0.25, 1.0, 10.0])
WIDE_PROBAS = np.array([0.05, 0.20, 0.50, 0.20, 0.05])
TRAIN_RANGE = (-1.0, 1.0)


# ---------------------------------------------------------------------------
# Flag defaults / propagation
# ---------------------------------------------------------------------------

def test_flag_defaults_to_false():
    dp = _make(WIDE_EDGES, WIDE_PROBAS, TRAIN_RANGE, is_grid_native=False)
    assert dp.is_grid_native is False
    assert dp.native.is_grid_native is False
    assert dp.resampled.is_grid_native is False


def test_flag_propagates_to_both_views():
    dp = _make(WIDE_EDGES, WIDE_PROBAS, TRAIN_RANGE, is_grid_native=True)
    assert dp.native.is_grid_native is True
    assert dp.resampled.is_grid_native is True


def test_tabpfn_wrappers_declare_grid_native():
    """The two TabPFN wrappers must construct grid-native predictions.

    Read at the source so the contract is enforced without importing the heavy
    ``tabpfn`` package (unavailable in the unit-test environment).
    """
    import inspect

    from scoringbench.univariate.wrappers import tabpfn as tabpfn_mod

    for cls in (tabpfn_mod.TabPFNWrapper, tabpfn_mod.FinetuneTabPFNWrapper):
        src = inspect.getsource(cls.predict_distribution)
        assert "is_grid_native=True" in src, (
            f"{cls.__name__}.predict_distribution must set is_grid_native=True"
        )


def test_discretized_xgb_head_declares_grid_native():
    """``XGBVectorWrapper`` is a DISCRETIZED head, so it must be grid-native.

    It trains a softmax over ``n_bins`` classes on a uniform grid laid out at fit
    time, i.e. its output already IS a PMF over fixed borders -- exactly TabPFN's
    situation.  Its sibling ``XGBQuantileVectorWrapper`` predicts QUANTILES, so it
    must NOT be grid-native: that grid is an artefact of the adapter and has to be
    resampled to stay comparable.
    """
    import inspect

    from scoringbench.univariate.wrappers import xgb_vector as xgb_mod

    src = inspect.getsource(xgb_mod.XGBVectorWrapper.predict_distribution)
    assert "is_grid_native=True" in src, (
        "XGBVectorWrapper.predict_distribution must set is_grid_native=True"
    )

    q_src = inspect.getsource(
        xgb_mod.XGBQuantileVectorWrapper.predict_distribution
    )
    assert "is_grid_native=True" not in q_src, (
        "XGBQuantileVectorWrapper is quantile-based, not grid-native"
    )


# ---------------------------------------------------------------------------
# Pass-through when the native PMF grid is wider than the train range
# ---------------------------------------------------------------------------

def test_wider_native_grid_passes_through_verbatim():
    dp = _make(WIDE_EDGES, WIDE_PROBAS, TRAIN_RANGE, is_grid_native=True)
    rg = dp.resampled
    # No resample: the density view keeps the native edges and masses exactly.
    assert np.array_equal(rg.bin_edges, WIDE_EDGES)
    assert np.array_equal(rg.probas, WIDE_PROBAS[None, :])


def test_grid_native_never_drops_tail_mass():
    """Neither a grid-native NOR an ordinary forecast drops tail mass now.

    Grid-native passes its native PMF grid through; an ordinary model is resampled
    onto its GROW-ONLY grid (train range widened to its own hull).  Both keep all
    their mass and both reach the native tails -- the old truncation of ordinary
    models to the train range is gone.
    """
    native = _make(WIDE_EDGES, WIDE_PROBAS, TRAIN_RANGE, is_grid_native=True)
    ordinary = _make(WIDE_EDGES, WIDE_PROBAS, TRAIN_RANGE, is_grid_native=False)

    # Both keep unit mass: nothing beyond the train range is dropped.
    assert np.isclose(native.resampled.probas.sum(), 1.0)
    assert np.isclose(ordinary.resampled.probas.sum(), 1.0)

    # Both supports reach the hull tails (grow-only for the ordinary model,
    # pass-through for the grid-native one).
    assert native.resampled.bin_edges[0] == WIDE_EDGES[0]
    assert native.resampled.bin_edges[-1] == WIDE_EDGES[-1]
    assert np.isclose(ordinary.resampled.bin_edges[0], WIDE_EDGES[0])
    assert np.isclose(ordinary.resampled.bin_edges[-1], WIDE_EDGES[-1])


# ---------------------------------------------------------------------------
# Extension when the native PMF grid is narrower than the train range
# ---------------------------------------------------------------------------

def test_narrower_native_grid_passes_through_verbatim():
    """A grid narrower than the train range is NOT widened at construction.

    Scoring pads every grid to cover the observed target on its own
    (``metrics.pad_to_common_grid``), so the resampled view keeps the native
    grid verbatim -- preserving the head's density resolution.
    """
    edges = np.array([-0.5, 0.0, 0.5])
    probas = np.array([0.5, 0.5])
    dp = _make(edges, probas, (-2.0, 3.0), is_grid_native=True)
    rg = dp.resampled
    assert np.array_equal(rg.bin_edges, edges)
    assert np.array_equal(rg.probas, probas[None, :])
    assert np.isclose(rg.probas.sum(), 1.0)


def test_grid_native_resampled_is_native_verbatim():
    """The resampled view equals the native PMF grid regardless of the train range.

    Asymmetric case (train straddles the grid on one side only): the grid is
    still returned untouched -- no widening on either side.
    """
    edges = np.array([-0.5, 0.0, 0.5])          # grid = [-0.5, 0.5]
    probas = np.array([0.5, 0.5])
    train_lo, train_hi = -2.0, 0.2              # train = [-2.0, 0.2]
    dp = _make(edges, probas, (train_lo, train_hi), is_grid_native=True)
    rg = dp.resampled

    assert np.array_equal(rg.bin_edges, edges)
    assert np.array_equal(rg.probas, probas[None, :])


# ---------------------------------------------------------------------------
# Density rules read the native PMF grid for a grid-native model
# ---------------------------------------------------------------------------

def _targets(n, lo=-0.4, hi=0.4, seed=0):
    return np.random.default_rng(seed).uniform(lo, hi, size=n)


def test_density_rules_invariant_to_train_range_when_grid_native():
    """A grid-native model's density scores don't depend on the train range.

    Because the density view keeps the native PMF grid, handing the SAME forecast a
    wider vs. narrower ``train_range`` (both already inside the native hull)
    leaves the density-rule inputs -- and therefore the scores -- unchanged.
    """
    n = 64
    probas = np.tile(WIDE_PROBAS, (n, 1))
    y = _targets(n)

    tight = _make(WIDE_EDGES, probas, (-1.0, 1.0), is_grid_native=True)
    loose = _make(WIDE_EDGES, probas, (-2.0, 2.0), is_grid_native=True)

    s_tight = compute_scoring_rules(tight, y, representation="auto")
    s_loose = compute_scoring_rules(loose, y, representation="auto")
    for k in DENSITY_BASED_KEYS:
        assert np.isclose(s_tight[k], s_loose[k]), k


def test_density_rules_invariant_to_train_range_when_hull_exceeds():
    """An ordinary model's density is invariant to train_range when its hull
    exceeds it: grow-only grows the support to the (same) hull either way.
    """
    n = 64
    probas = np.tile(WIDE_PROBAS, (n, 1))       # hull [-10, 10] exceeds both ranges
    y = _targets(n)

    tight = _make(WIDE_EDGES, probas, (-1.0, 1.0), is_grid_native=False)
    loose = _make(WIDE_EDGES, probas, (-2.0, 2.0), is_grid_native=False)

    s_tight = compute_scoring_rules(tight, y, representation="auto")
    s_loose = compute_scoring_rules(loose, y, representation="auto")
    for k in DENSITY_BASED_KEYS:
        assert np.isclose(s_tight[k], s_loose[k]), k


def test_density_rules_depend_on_train_range_when_hull_inside():
    """When the hull is INSIDE the train range the support floors at the train
    range, so an ordinary model's density still depends on train_range.
    """
    n = 64
    # Narrow hull [-0.5, 0.5], inside both train ranges below.
    edges = np.array([-0.5, -0.25, 0.25, 0.5])
    probas = np.tile(np.array([0.25, 0.5, 0.25]), (n, 1))
    y = _targets(n)

    tight = _make(edges, probas, (-3.0, 3.0), is_grid_native=False)
    loose = _make(edges, probas, (-6.0, 6.0), is_grid_native=False)

    s_tight = compute_scoring_rules(tight, y, representation="auto")
    s_loose = compute_scoring_rules(loose, y, representation="auto")
    # The floored support equals the (differing) train ranges, so the regrid
    # width -- and at least one density rule -- must move.
    assert any(
        not np.isclose(s_tight[k], s_loose[k]) for k in DENSITY_BASED_KEYS
    )


def test_grid_native_density_matches_score_on_native_grid():
    """The ``auto`` density scores equal those computed on the native PMF grid.

    For a grid-native model ``dist.resampled`` IS the native PMF grid, so scoring the
    resampled view directly (``representation="resampled"``) must reproduce the
    density keys the merged ``auto`` path reports.
    """
    n = 64
    probas = np.tile(WIDE_PROBAS, (n, 1))
    y = _targets(n)
    dp = _make(WIDE_EDGES, probas, TRAIN_RANGE, is_grid_native=True)

    auto = compute_scoring_rules(dp, y, representation="auto")
    resampled = compute_scoring_rules(dp, y, representation="resampled")
    for k in DENSITY_BASED_KEYS:
        assert np.isclose(auto[k], resampled[k]), k


# ---------------------------------------------------------------------------
# Grid-robust rules are untouched by the flag
# ---------------------------------------------------------------------------

def test_grid_robust_rules_unaffected_by_flag():
    """CRPS et al. read the verbatim native view, identical regardless of flag."""
    n = 64
    probas = np.tile(WIDE_PROBAS, (n, 1))
    y = _targets(n)

    on = _make(WIDE_EDGES, probas, TRAIN_RANGE, is_grid_native=True)
    off = _make(WIDE_EDGES, probas, TRAIN_RANGE, is_grid_native=False)

    s_on = compute_scoring_rules(on, y, representation="auto")
    s_off = compute_scoring_rules(off, y, representation="auto")
    for k in s_on:
        if k in DENSITY_BASED_KEYS:
            continue
        assert np.isclose(s_on[k], s_off[k]), k


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
