"""Tests for target-discreteness detection and density-rule skipping.

Two concerns:

1. ``target_discreteness`` correctly separates continuous targets (density
   rules valid) from lattice / ordinal / spike targets (density rules skipped),
   using the lattice-OR-atom-mass detector, and is invariant to repetitions.
2. ``compute_scoring_rules(..., score_density=False)`` omits exactly the
   DENSITY_BASED_KEYS and keeps the grid-robust family -- so a flagged dataset
   drops out of the density leaderboards only.
"""

import numpy as np
import pytest
import torch

# Force CPU so the suite runs anywhere.
torch.cuda.is_available = lambda: False

from scoringbench.univariate.discreteness import (  # noqa: E402
    target_discreteness,
    DISCRETE_OVERRIDES,
)
from scoringbench.univariate.metrics import (  # noqa: E402
    compute_scoring_rules,
    DENSITY_BASED_KEYS,
    GRID_ROBUST_KEYS,
    PIT_KEYS,
)
from scoringbench.univariate.wrappers import DistributionPrediction  # noqa: E402


# ---------------------------------------------------------------------------
# Detector: continuous targets are NOT flagged
# ---------------------------------------------------------------------------

def test_gaussian_is_continuous():
    rng = np.random.default_rng(0)
    y = rng.normal(size=5000)
    out = target_discreteness(y)
    assert out["is_discrete"] is False
    assert out["rule"] is None


def test_uniform_is_continuous():
    rng = np.random.default_rng(1)
    y = rng.uniform(-3, 7, size=4000)
    assert target_discreteness(y)["is_discrete"] is False


def test_finely_rounded_continuous_is_not_flagged():
    """Continuous data rounded to a fine grid is a lattice at float resolution
    but each site carries negligible mass (frac_unique high) -> NOT discrete."""
    rng = np.random.default_rng(2)
    y = np.round(rng.normal(0, 10, size=8000), 2)  # thousands of 0.01-sites
    out = target_discreteness(y)
    assert out["frac_unique"] > 0.10
    assert out["is_discrete"] is False


# ---------------------------------------------------------------------------
# Detector: discrete targets ARE flagged
# ---------------------------------------------------------------------------

def test_ordinal_few_levels():
    rng = np.random.default_rng(3)
    y = rng.integers(1, 6, size=3000).astype(float)  # 5-level Likert
    out = target_discreteness(y)
    assert out["is_discrete"] is True
    assert out["rule"] == "few_unique"


def test_integer_lattice_many_levels():
    """Integer counts with many levels but heavy repetition -> lattice branch."""
    rng = np.random.default_rng(4)
    y = rng.poisson(50, size=20000).astype(float)  # >20 unique, span 1, ties
    out = target_discreteness(y)
    assert out["is_discrete"] is True
    assert out["rule"] in ("lattice", "few_unique")
    assert out["ecf_peak"] == pytest.approx(1.0, abs=1e-6)
    assert out["span"] == pytest.approx(1.0, rel=1e-3)


def test_non_integer_lattice():
    """A lattice with a non-integer span (multiples of 0.37) is still caught."""
    rng = np.random.default_rng(5)
    y = 0.37 * rng.integers(0, 40, size=20000).astype(float)
    out = target_discreteness(y)
    assert out["is_discrete"] is True
    assert out["ecf_peak"] == pytest.approx(1.0, abs=1e-6)
    assert out["span"] == pytest.approx(0.37, rel=1e-3)


def test_spike_plus_continuum():
    """A dominant atom (e.g. censoring at 0) on otherwise continuous data."""
    rng = np.random.default_rng(6)
    cont = rng.normal(5, 2, size=10000)
    cont[:2000] = 0.0  # 20% mass at a single value, rest continuous
    out = target_discreteness(cont)
    assert out["is_discrete"] is True
    assert out["rule"] == "atom_mass"


def test_constant_target_is_discrete():
    out = target_discreteness(np.full(100, 3.0))
    assert out["is_discrete"] is True


def test_empty_target_is_discrete():
    out = target_discreteness(np.array([]))
    assert out["is_discrete"] is True
    assert out["rule"] == "empty"


# ---------------------------------------------------------------------------
# Invariance to repetitions / subsampling (the CV-folds concern)
# ---------------------------------------------------------------------------

def test_flag_invariant_to_duplication():
    """Duplicating every row (a crude stand-in for resampling) must not flip
    the flag: the lattice signal is multiplicity-weighted, so ties only
    reinforce it."""
    rng = np.random.default_rng(7)
    y = rng.integers(0, 100, size=5000).astype(float)
    base = target_discreteness(y)
    dup = target_discreteness(np.concatenate([y, y, y]))
    assert base["is_discrete"] == dup["is_discrete"] is True
    assert dup["ecf_peak"] == pytest.approx(base["ecf_peak"], abs=1e-9)


def test_continuous_flag_invariant_to_subsample():
    rng = np.random.default_rng(8)
    y = rng.normal(size=10000)
    full = target_discreteness(y)
    sub = target_discreteness(y[:2000])
    assert full["is_discrete"] == sub["is_discrete"] is False


# ---------------------------------------------------------------------------
# Manual override wins over the auto-detector
# ---------------------------------------------------------------------------

def test_override_forces_flag(monkeypatch):
    rng = np.random.default_rng(9)
    y_cont = rng.normal(size=4000)
    # Force a continuous target to be treated as discrete.
    monkeypatch.setitem(DISCRETE_OVERRIDES, "forced_discrete", True)
    out = target_discreteness(y_cont, name="forced_discrete")
    assert out["is_discrete"] is True
    assert out["overridden"] is True

    # Force a discrete target to be treated as continuous.
    y_ord = rng.integers(1, 5, size=4000).astype(float)
    monkeypatch.setitem(DISCRETE_OVERRIDES, "forced_continuous", False)
    out2 = target_discreteness(y_ord, name="forced_continuous")
    assert out2["is_discrete"] is False
    assert out2["overridden"] is True


def test_no_override_for_unlisted_name():
    rng = np.random.default_rng(10)
    y = rng.normal(size=2000)
    out = target_discreteness(y, name="not_in_map")
    assert out["overridden"] is False


# ---------------------------------------------------------------------------
# End-to-end: score_density=False omits exactly the density keys
# ---------------------------------------------------------------------------

def _simple_dist():
    bin_edges = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    bin_mids = np.array([0.5, 1.5, 2.5], dtype=np.float64)
    probas = np.array([[0.3, 0.4, 0.3]], dtype=np.float64)
    return DistributionPrediction(
        probas=probas,
        bin_edges=bin_edges,
        bin_midpoints=bin_mids,
        mean=np.array([1.5], dtype=np.float64),
        train_range=(0.0, 3.0),
    )


def test_score_density_false_drops_density_and_pit_keys():
    dist = _simple_dist()
    y = np.array([1.5], dtype=np.float64)

    full = compute_scoring_rules(dist, y, score_density=True)
    skipped = compute_scoring_rules(dist, y, score_density=False)

    # No density keys when skipped ...
    for k in DENSITY_BASED_KEYS:
        assert k in full
        assert k not in skipped
    # ... and the PIT KS diagnostic is dropped too: its continuity null is
    # invalid on a discrete / lattice target (it false-rejects a perfect
    # forecast), so it must not be reported there.
    for k in PIT_KEYS:
        assert k in full
        assert k not in skipped
    # ... but every grid-robust key survives, with identical values.
    for k in GRID_ROBUST_KEYS:
        assert k in skipped
        assert skipped[k] == pytest.approx(full[k], rel=1e-9, abs=1e-12)
