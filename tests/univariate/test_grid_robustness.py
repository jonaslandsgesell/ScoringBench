"""Grid-robustness contract for the univariate scoring rules.

Background
----------
``xgb_vector_quantile`` scored implausibly well on the density-based rules
(worst on ``pseudospherical_alpha_3.0``) on a subset of datasets.  The cause is
NOT a bug in the rules and NOT an impropriety: it is the interaction of two
independent facts.

1. A rule that reads the predictive *density* ``f(y)`` is unbounded on a
   histogram, because ``f = p_k / w_k`` and ``w_k`` may be made arbitrarily
   small.  For a forecast uniform on a window of width ``eps`` centred on the
   realisation the rules have exact closed forms::

       DPD_beta      = eps**-beta - (1 + 1/beta) * eps**-beta
                     = -eps**-beta / beta                        -> -inf
       PseudoS_alpha = -(eps**-((alpha-1)/alpha) - 1) / (alpha-1) -> -inf

2. The model is allowed to *choose its own grid*, so it controls ``eps``.

Neither fact alone is exploitable.  Against a continuous target the divergence
is exactly cancelled in expectation by the ``P(y outside window) -> 1`` misses,
so the rules stay proper and shrinking the support is self-defeating.  The
exploit needs genuine **atoms** in ``y``: only then does ``P(y = c) > 0``
survive ``eps -> 0``, so a model that puts its spike on the lattice collects an
unbounded reward.

The precise condition, for DPD with a tie rate ``q = P(y lands on the spike)``::

    E[S] = q * (-eps**-beta / beta) + (1 - q) * (+eps**-beta)

whose ``eps**-beta`` coefficient is ``(1 - q) - q/beta``, i.e. the score
diverges DOWNWARD (exploitable) iff

    q > beta / (1 + beta).

For ``beta = 0.5`` the threshold is ``q* = 1/3``.

Consequences encoded below
--------------------------
* ``test_classification_covers_reported_metrics`` -- the
  DENSITY_BASED_KEYS / GRID_ROBUST_KEYS split in ``metrics.py`` stays in sync
  with what ``compute_scoring_rules`` actually reports.
* ``test_grid_robust_rules_cannot_be_gamed`` -- CDF-based rules improve toward
  their perfect-forecast value instead of diverging.
* ``test_density_rules_diverge_on_free_grid`` -- density rules do diverge, at
  the analytic rate, when scored on the model's own (native) grid, so the risk
  is real and measured, not hypothetical.  The production path never scores them
  there; this test reaches the native density directly to exhibit the danger.
* ``test_default_path_neutralises_the_free_grid_attack`` -- the SAME attack run
  through ``compute_scoring_rules`` (which regrids density rules onto the shared
  common grid) stays finite: the remedy is on by default.
* ``test_continuous_target_is_not_exploitable`` -- with continuous ``y`` the
  same attack makes every rule WORSE: propriety is intact and "a finite test
  set is a bag of Dirac atoms" is NOT the mechanism.
* ``test_tie_rate_threshold`` -- the sign flip happens at ``beta/(1+beta)``,
  pinning the mechanism to the tie rate.
* ``test_fixed_shared_grid_caps_density_rules`` -- the remedy: a shared
  evaluation grid of width ``h`` caps ``f`` at ``1/h`` and hence caps every
  density rule at a finite plateau for all ``eps <= h``.

The upshot for the benchmark: rules in ``GRID_ROBUST_KEYS`` are safe to report
on native model grids; rules in ``DENSITY_BASED_KEYS`` are only comparable once
all models are resampled onto one fixed grid.
"""

import logging

import numpy as np
import pytest

from scoringbench.univariate.metrics import (
    BOUNDED_DIAGNOSTIC_KEYS,
    DENSITY_BASED_KEYS,
    DPD_BETAS,
    GRID_ROBUST_KEYS,
    PSEUDOS_ALPHAS,
    _score_one_view,
    compute_scoring_rules,
)
from scoringbench.univariate.wrappers import DistributionPrediction


def _density_on_native(dist, ys):
    """Score the density rules on the model's *own* (native) grid.

    ``compute_scoring_rules`` routes density rules onto the shared common grid
    (``dist.resampled``), which is exactly the remedy that CAPS the divergence.
    To exhibit the un-capped risk the remedy defends against we have to score
    the raw native density directly, which is what ``_score_one_view`` with
    ``compute_density=True`` does.  This is never the benchmark path -- it is
    only the counterfactual "what a model with a free grid could grab".
    """
    return _score_one_view(dist.native, ys, compute_density=True)

logger = logging.getLogger(__name__)

# Lattice a model could plausibly learn from the training targets.  Crucially
# the attack below NEVER reads the test y -- an adversary that centres its
# spike on the realisation is an oracle, and an oracle beats every proper rule,
# so such a probe would prove nothing.
LATTICE = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
EPSS = [1e-1, 1e-2, 1e-3, 1e-4]
NB = 8


def _spike(centres, eps, n_bins=NB):
    """Forecast uniform on ``[c - eps/2, c + eps/2]``, on the model's own grid."""
    centres = np.asarray(centres, dtype=float)
    offs = np.linspace(-0.5, 0.5, n_bins + 1)[None, :] * eps
    edges = centres[:, None] + offs
    return DistributionPrediction(
        probas=np.full((len(centres), n_bins), 1.0 / n_bins),
        bin_edges=edges,
        bin_midpoints=0.5 * (edges[:, :-1] + edges[:, 1:]),
        mean=centres,
        train_range=(float(np.asarray(edges).min()), float(np.asarray(edges).max())),
    )


def _on_fixed_grid(centres, eps, lo=-1.0, hi=2.0, n_bins=300):
    """Same spike, but resampled onto ONE grid shared by every model.

    The shared grid is declared to the mother via ``num_equally_sized_bins``/``train_range``
    so that ``dist.resampled`` -- the view the density rules actually score -- is
    exactly this ``n_bins``-bin grid over ``[lo, hi]``.  Passing the spike in
    already on that grid makes the regrid an identity, so the density caps at the
    grid scale ``1/h`` as the test asserts.
    """
    centres = np.asarray(centres, dtype=float)
    edges = np.linspace(lo, hi, n_bins + 1)
    lo_e, hi_e = centres[:, None] - eps / 2, centres[:, None] + eps / 2
    # mass of the spike falling in each fixed bin
    overlap = np.clip(
        np.minimum(hi_e, edges[None, 1:]) - np.maximum(lo_e, edges[None, :-1]), 0.0, None
    )
    probas = overlap / overlap.sum(axis=1, keepdims=True)
    return DistributionPrediction(
        probas=probas,
        bin_edges=np.repeat(edges[None, :], len(centres), axis=0),
        bin_midpoints=np.repeat((0.5 * (edges[:-1] + edges[1:]))[None, :], len(centres), axis=0),
        mean=centres,
        num_equally_sized_bins=n_bins,
        train_range=(lo, hi),
    )


def _discrete_targets(n, seed=0):
    """Targets WITH atoms: exactly on the lattice (the real-dataset case)."""
    return np.random.default_rng(seed).choice(LATTICE, size=n)


def _snap(ys):
    """Nearest lattice point -- the model's best guess, computed from y's
    *support* only, never from the individual realisation's exact value."""
    return LATTICE[np.abs(np.asarray(ys)[:, None] - LATTICE[None, :]).argmin(axis=1)]


@pytest.fixture(scope="module")
def reported_keys():
    ys = np.linspace(0.1, 0.9, 64)
    return set(compute_scoring_rules(_spike(_snap(ys), 0.2), ys).keys())


# ---------------------------------------------------------------------------
# 1. the classification must describe the metrics we actually report
# ---------------------------------------------------------------------------


def test_classification_covers_reported_metrics(reported_keys):
    """Every reported metric lands in exactly one branch and nothing is orphaned.

    This is the guard that makes the two-branch benchmark safe: a metric added
    later without being classified will fail here rather than silently be
    reported on the wrong grid.
    """
    density = set(DENSITY_BASED_KEYS)
    robust = set(GRID_ROBUST_KEYS)
    diag = set(BOUNDED_DIAGNOSTIC_KEYS)

    for key in density | robust | diag:
        assert key in reported_keys, f"{key} is classified but never reported"

    # mutually exclusive
    assert not density & robust, "a rule cannot be both density and grid-robust"
    assert not density & diag, "a rule cannot be both density and diagnostic"
    assert not robust & diag, "a rule cannot be both grid-robust and diagnostic"

    # exhaustive: mae/rmse are point metrics computed elsewhere, everything the
    # distribution path emits must be classified
    unclassified = reported_keys - density - robust - diag - {"mae", "rmse"}
    assert not unclassified, f"unclassified reported metrics: {sorted(unclassified)}"


# ---------------------------------------------------------------------------
# 2. CDF-based rules are immune
# ---------------------------------------------------------------------------


def _collapse_rate(key):
    """Exponent r with which a rule approaches 0 as the support width eps -> 0.

    The energy score with kernel |x - y|**beta inherits that beta, so for small
    beta it decays very slowly (beta=0.1 needs eps=1e-20 to reach 0.01) -- that
    is correct behaviour, not a leak.  Everything else in the CRPS/CRTS/interval
    family is first order in eps.
    """
    if key.startswith("energy_score_beta_"):
        return float(key.rsplit("_", 1)[1])
    return 1.0


def test_grid_robust_rules_cannot_be_gamed():
    """Collapsing the support drives F-based rules to their perfect-forecast
    value, not to -inf, so they stay comparable across native PMF grids."""
    ys = _discrete_targets(2000)
    centres = _snap(ys)
    eps = EPSS[-1]
    wide = compute_scoring_rules(_spike(centres, EPSS[0]), ys)
    tight = compute_scoring_rules(_spike(centres, eps), ys)
    for key in GRID_ROBUST_KEYS:
        assert tight[key] >= -1e-9, f"{key} went negative ({tight[key]:.3e}); it is a loss >= 0"
        assert tight[key] <= wide[key] + 1e-9, f"{key} not monotone in support width"
        # bounded by its own analytic collapse rate, with room for the constant
        bound = max(1e-3, 4.0 * eps ** _collapse_rate(key))
        assert tight[key] < bound, (
            f"{key} = {tight[key]:.3e} exceeds its collapse rate "
            f"eps**{_collapse_rate(key)} (bound {bound:.3e})"
        )


# ---------------------------------------------------------------------------
# 3. density rules do diverge, at the analytic rate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("beta", DPD_BETAS)
def test_dpd_matches_closed_form_spike_limit(beta):
    """DPD_beta on an exact hit equals -eps**-beta / beta."""
    y = np.array([0.5])
    for eps in [1e-2, 1e-3, 1e-4]:
        got = compute_scoring_rules(_spike(y, eps), y)[f"dpd_beta_{beta}"]
        want = -(eps ** -beta) / beta
        assert got == pytest.approx(want, rel=1e-6), f"beta={beta} eps={eps}: {got} != {want}"


@pytest.mark.parametrize("alpha", PSEUDOS_ALPHAS)
def test_pseudospherical_matches_closed_form_spike_limit(alpha):
    """PseudoS_alpha on an exact hit equals -(eps**-((a-1)/a) - 1)/(a-1)."""
    y = np.array([0.5])
    for eps in [1e-2, 1e-3, 1e-4]:
        got = compute_scoring_rules(_spike(y, eps), y)[f"pseudospherical_alpha_{alpha}"]
        want = -(eps ** (-(alpha - 1) / alpha) - 1.0) / (alpha - 1.0)
        assert got == pytest.approx(want, rel=1e-6), f"alpha={alpha} eps={eps}: {got} != {want}"


def test_density_rules_diverge_on_free_grid():
    """With atoms AND a model-chosen grid the raw-density reward is unbounded.

    This is the *risk*, measured on the native PMF grid via ``_density_on_native``.
    ``compute_scoring_rules`` no longer exposes it -- it always regrids density
    rules onto the shared common grid, which is the remedy tested in
    ``test_fixed_shared_grid_caps_density_rules``.  Here we deliberately bypass
    that remedy to confirm the danger it defends against is real and diverges at
    the analytic rate, not hypothetical.
    """
    ys = _discrete_targets(2000)
    centres = _snap(ys)
    wide = _density_on_native(_spike(centres, EPSS[0]), ys)
    tight = _density_on_native(_spike(centres, EPSS[-1]), ys)
    for key in DENSITY_BASED_KEYS:
        assert tight[key] < wide[key], f"{key} did not improve; attack model changed?"
    assert tight["cde_loss"] < -1e3, "cde_loss should be hugely negative"
    logger.info("free grid, discrete y: cde_loss %.1f -> %.1f", wide["cde_loss"], tight["cde_loss"])


def test_default_path_neutralises_the_free_grid_attack():
    """The benchmark path (``compute_scoring_rules``) is immune to the attack.

    Same spike, same collapsing eps as ``test_density_rules_diverge_on_free_grid``,
    but scored through the production ``auto`` path: because density rules are
    read off the shared common grid, tightening eps below the grid scale can no
    longer buy an unbounded reward.  cde_loss stays finite and does not plunge.
    """
    ys = _discrete_targets(2000)
    centres = _snap(ys)
    wide = compute_scoring_rules(_spike(centres, EPSS[0]), ys)
    tight = compute_scoring_rules(_spike(centres, EPSS[-1]), ys)
    assert np.isfinite(tight["cde_loss"]), "common-grid cde_loss must stay finite"
    assert tight["cde_loss"] > -1e3, (
        f"common grid failed to cap the attack: cde_loss {tight['cde_loss']:.1f}"
    )
    logger.info(
        "shared grid, discrete y: cde_loss %.1f -> %.1f (capped)",
        wide["cde_loss"], tight["cde_loss"],
    )


def test_slow_diverging_dpd_betas_are_still_density_based():
    """dpd_beta_0.01 and dpd_beta_0.2 grow only ~1x / ~4x by eps=1e-4, so a naive
    empirical magnitude threshold would mis-file them as grid-robust.  They are
    NOT: they diverge as eps**-beta, just slowly.  This pins them to the density
    branch by their *rate*, not their value at any single eps, so they are only
    ever reported on the fixed shared grid.
    """
    y = np.array([0.5])
    for beta in (0.01, 0.2):
        assert f"dpd_beta_{beta}" in DENSITY_BASED_KEYS
        s = {e: compute_scoring_rules(_spike(y, e), y)[f"dpd_beta_{beta}"] for e in (1e-2, 1e-6)}
        # magnitude must grow by the analytic factor (1e-6/1e-2)**-beta = 1e4**beta
        grew = abs(s[1e-6]) / abs(s[1e-2])
        assert grew == pytest.approx(1e4 ** beta, rel=1e-3), (
            f"dpd_beta_{beta} grew {grew:.3f}x, expected {1e4 ** beta:.3f}x"
        )


def test_bounded_diagnostics_stay_bounded():
    """The non-scoring diagnostics cannot be gamed either: under the collapsing
    support they sit at a fixed value, so they are safe to report on any grid.
    """
    ys = _discrete_targets(2000)
    centres = _snap(ys)
    wide = compute_scoring_rules(_spike(centres, EPSS[0]), ys)
    tight = compute_scoring_rules(_spike(centres, EPSS[-1]), ys)
    for key in BOUNDED_DIAGNOSTIC_KEYS:
        assert np.isfinite(tight[key]), f"{key} became non-finite"
        # never grows without bound; either shrinks toward 0 or holds a constant
        assert abs(tight[key]) <= abs(wide[key]) + 1.0 + 1e-9, (
            f"{key} = {tight[key]:.3e} grew relative to {wide[key]:.3e}"
        )


# ---------------------------------------------------------------------------
# 4. propriety is intact -- atoms are the precondition, not finiteness
# ---------------------------------------------------------------------------


def test_continuous_target_is_not_exploitable():
    """The SAME attack against a continuous target makes every rule worse.

    This refutes "a finite test set is just a collection of Dirac atoms, so the
    blow-up survives anyway".  Finiteness alone is not enough: what matters is
    whether the atoms are *repeated*, i.e. whether the tie rate is bounded away
    from zero as eps -> 0.
    """
    ys = np.random.default_rng(3).uniform(0.0, 1.0, 20_000)
    centres = _snap(ys)  # still no oracle: nearest lattice point
    wide = compute_scoring_rules(_spike(centres, EPSS[0]), ys)
    tight = compute_scoring_rules(_spike(centres, EPSS[-1]), ys)
    for key in DENSITY_BASED_KEYS:
        assert tight[key] > wide[key], (
            f"{key} improved on continuous y ({wide[key]:.3e} -> {tight[key]:.3e}); "
            "that would be a genuine impropriety"
        )
    assert tight["cde_loss"] > 0, "continuous y: shrinking support must be punished"


@pytest.mark.parametrize(
    "q, exploitable",
    [(0.0, False), (0.10, False), (0.30, False), (0.40, True), (1.0, True)],
)
def test_tie_rate_threshold(q, exploitable):
    """Sign of the divergence flips exactly at q* = beta/(1+beta) = 1/3."""
    beta, n = 0.5, 40_000
    rng = np.random.default_rng(7)
    ys = np.where(rng.random(n) < q, 0.5, rng.uniform(0.0, 1.0, n))
    centres = np.full(n, 0.5)
    lo = compute_scoring_rules(_spike(centres, 1e-1), ys)[f"dpd_beta_{beta}"]
    hi = compute_scoring_rules(_spike(centres, 1e-4), ys)[f"dpd_beta_{beta}"]
    assert (hi < lo) is exploitable, (
        f"q={q} (threshold {beta / (1 + beta):.3f}): {lo:.2f} -> {hi:.2f}, "
        f"expected {'downward' if exploitable else 'upward'} divergence"
    )


# ---------------------------------------------------------------------------
# 5. the remedy
# ---------------------------------------------------------------------------


# The lattice points are all multiples of 0.25 = 25h, so with lo = -1.0 every
# atom sits exactly on a bin EDGE (its mass splits over the two neighbouring
# bins, giving f = 1/(2h)), while offsetting the grid by h/2 puts every atom at
# a bin CENTRE (all mass in one bin, f = 1/h -- the true worst case).
_ALIGNMENTS = [
    pytest.param(-1.0, 0.5, id="atom-on-bin-edge"),
    pytest.param(-1.005, 1.0, id="atom-on-bin-centre"),
]


@pytest.mark.parametrize("lo, mass_frac", _ALIGNMENTS)
def test_fixed_shared_grid_caps_density_rules(lo, mass_frac):
    """On one shared grid of width h, every density rule plateaus for eps <= h.

    f is capped at 1/h, so the unbounded reward disappears and scores from
    models with different native resolutions become comparable.  The plateau is
    the closed form evaluated at the grid scale rather than at the model's eps.
    """
    n_bins = 300
    hi = lo + 3.0
    h = 3.0 / n_bins
    # Every row's spike is snapped onto its own lattice atom, so the per-row
    # density score depends only on eps/grid alignment -- not on how many rows
    # nor which atom.  A small sample gives the identical average as a large one
    # at a fraction of the regrid+scoring cost.
    ys = _discrete_targets(200)
    centres = _snap(ys)

    ref = compute_scoring_rules(_on_fixed_grid(centres, h, lo, hi, n_bins), ys)
    for eps in [h / 10, h / 100, h / 1000]:
        got = compute_scoring_rules(_on_fixed_grid(centres, eps, lo, hi, n_bins), ys)
        for key in DENSITY_BASED_KEYS:
            assert got[key] == pytest.approx(ref[key], rel=1e-6, abs=1e-9), (
                f"{key} still moves below the grid scale: {ref[key]:.6f} -> {got[key]:.6f}"
            )

    # DPD_1 = \int f^2 - 2 f(y) with f = mass_frac/h spread over 1/mass_frac bins
    f = mass_frac / h
    assert ref["dpd_beta_1.0"] == pytest.approx(f - 2.0 * f, rel=1e-3)
    # the cap itself: nothing can beat the finest representable resolution
    assert ref["dpd_beta_1.0"] >= -1.0 / h - 1e-6
    assert np.isfinite(ref["cde_loss"]) and ref["cde_loss"] >= -1.0 / h - 1e-6
    logger.info(
        "fixed grid h=%.4g (%s): cde_loss=%.3f, cap=%.1f",
        h, "centre" if mass_frac == 1.0 else "edge", ref["cde_loss"], -1.0 / h,
    )
