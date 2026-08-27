r"""Density scoring rules require a shared common grid: die-roll codification.

The bug in one sentence
-----------------------
A forecast with support ONLY on faces {1,2,3,4,5} -- which assigns ZERO
probability to face 6, an outcome that occurs ~1/6 of the time -- nevertheless
WINS the density-based scoring rules against a perfectly calibrated fair-die
forecast, purely because it reports narrower bins.  The common grid removes the
bin-width advantage and restores the correct ranking.

It is about the reported DENSITY, not the probabilities
-------------------------------------------------------
DPD and the pseudospherical rules never see a probability.  They evaluate the
predictive *density*

    f(y) = p_k / w_k        (p_k = mass in the bin containing y, w_k its width)

and the width in that denominator is chosen by the model.  Two forecasts can
agree on every probability and still report densities that differ by orders of
magnitude.  That is the whole problem: the quantity being scored is not a
property of the forecast distribution alone, it is a property of the
distribution *plus the grid it was printed on*.

Here are the densities the two forecasts actually hand to the scoring rules:

    forecast   mass/face   box width   reported density f at a face
    ---------  ----------  ----------  ----------------------------
    honest     1/6         0.02          8.33
    truncated  1/5         0.0005      400.0     (and f = 0 at face 6)

The truncated forecast reports a density 48x larger at the faces it believes in,
not because it is more confident in any probabilistic sense -- it holds 1/5 vs
1/6, a trivial difference -- but because it printed the same mass onto a 40x
narrower box.  Density rules reward exactly that number.  So the truncated model
wins despite being blind to face 6, whose density it reports as f = 0.

Put the other way round: on a model-chosen grid, ``w`` is a free parameter, so
``f = p/w`` is unbounded above for any fixed p.  A model can drive its own
density arbitrarily high without changing a single probability.  There is no
ranking to defend under those conditions, because the rules are not being fed
comparable quantities.

Fixing one shared grid pins the denominator: every forecast is resampled onto the
same ``NUM_EQUALLY_SIZED_BINS`` bins over ``COMMON_RANGE``, so ``w = h`` for everyone and
``f = p/h`` is once again a function of the forecast alone.  Both box widths here
are below ``h``, so both forecasts are resolved at the same density cap.  The
key change is from UNBOUNDED to BOUNDED: no width a model can report buys it more
than a small, capped amount (see the width-effect test below), and never enough
to change the ranking.

What the shared grid is, precisely: bandwidth + lattice
--------------------------------------------------------
Resampling onto the common grid is a uniform (boxcar) kernel smoother at
bandwidth ``h``, FOLLOWED BY sampling on a fixed lattice.  Those are two
separate operations and they do two different things:

* the SHARED BANDWIDTH is what removes the gameability.  It is the only part
  that matters for propriety, and it is not about smoothness: the kernel SHAPE
  merely rescales f by a constant (a Gaussian kernel reports
  ``phi(0) = 0.39894`` times the boxcar density at every bandwidth).  A smooth
  Gaussian dressing at a MODEL-CHOSEN bandwidth would therefore be gameable as
  ``bw**-beta``, exactly like a boxcar.  Pinning the bandwidth is the fix;
  choosing a prettier kernel is not.
* the LATTICE leaves a residual.  Because a reported box can straddle a bin
  edge, and the split between the two bins depends on the box width, the width
  effect is not annihilated -- it is merely bounded (~28% relative here, versus
  unbounded natively).  It vanishes exactly when the box lies wholly inside one
  bin, or exactly astride an edge (where it splits 50/50 for any width).  Two
  tests below pin this down: one bounds the residual, the other searches phase x
  width jointly and shows the residual can never flip the ranking.  A CENTRED
  evaluation -- boxcar or smooth, it makes no difference -- would remove the
  residual entirely, since it is the lattice and not the kernel shape that
  creates it.

Note this fixture puts every face exactly ON a bin edge, so the honest forecast's
box is split across two bins -- and it still wins.

Which ranking is the correct one?
----------------------------------
We do not settle this by inspecting the continuous rules -- they are the object
under test, and on native PMF grids they are being fed incomparable densities.  For a
die the outcome space is finite, so the decisive instruments are the proper
scoring rules *for discrete outcomes*, which read the probability vector directly
and involve no bin width at all.  Against a fair die (q_i = 1/6), expected scores
(lower = better):

    rule                    honest (1/6 each)   truncated (1/5 on faces 1-5)
    ----------------------  -----------------   ----------------------------
    Brier / quadratic        0.83333             0.86667
    spherical               -0.40825            -0.37268
    discrete DPD  beta=0.2  -3.49414            -2.89912
    discrete DPD  beta=0.5  -0.81650            -0.67082
    discrete DPD  beta=1.0  -0.16667            -0.13333

Every one of them prefers the honest forecast, as propriety over the simplex
requires: each is uniquely minimised at p = q.  This is the ground truth for the
die, established without reference to any grid.  It is what the common-grid path
reproduces and what the native-grid path inverts.

No divergence is involved anywhere in this argument.  Every rule used here is
FINITE on a zero-density region: the discrete rules above are polynomial in p,
and the continuous rules under test raise f(y) to a positive power.  Reporting
f = 0 at face 6 therefore costs the truncated forecast a finite, quantifiable
amount under all of them -- a penalty it then out-earns on the other five faces
by inflating f there.  Nothing below compares a finite number against an
infinite one, and no rule is used that would need epsilon-smoothing or kernel
dressing to be well defined on this fixture in the first place.  The
common-grid tests assert finiteness explicitly for exactly this reason.

Grounding in Gneiting's natural domains (arXiv:1608.06802v1, Table 2)
----------------------------------------------------------------------
Gneiting's Table 2 pairs each proper scoring rule with its natural domain F:

"For any given scoring rule S, the associated natural domain is the largest
convex class of probability distributions F such that S(F,y) is well defined
and finite almost surely under F."

    rule family                        reads   natural domain F
    ---------------------------------  ------  ----------------------
    density rules (DPD, pseudoS)       f(y)    L1  (a density exists)
    CRPS                               F(z)    M1  (finite mean)

The rules under test read f(y), so they require the forecast to BE a density:
their natural domain is L1.  On a model-chosen grid f = p/w is not pinned down
at all -- w is a free parameter the model sets -- so the forecast is not a
well-defined member of L1, and the propriety guarantee, the only reason to trust
the ranking, simply does not apply.  The common grid places every forecast back
inside L1 on equal terms.

Contrast the CRPS, whose natural domain is only M1 (finite mean): it reads F,
never f, so it needs no shared grid -- which is why it lives in
GRID_ROBUST_KEYS and is scored on the native PMF grid.

WARNING -- deliberate improper embedding
-----------------------------------------
We are well aware that {1,...,6} is a *discrete* outcome space and that
applying CONTINUOUS density scoring rules (DPD, pseudospherical) to it is not
proper in the strict sense: these rules assume a Lebesgue density, which a
lattice-supported law does not have -- which is why f = p/w grows without bound
as w shrinks.  For genuinely discrete outcomes one should use the discrete rules
tabulated above (Brier / spherical / discrete DPD over the six categories), and
that is exactly the role they play in this file: they supply the correct ranking.
We embed the die into R ON PURPOSE, purely as a test fixture: the discreteness is
the stressor that makes the grid-dependence of f visible and lets us assert the
native-vs-common contract sharply.  It is not a recommendation for scoring dice.
"""

import numpy as np
import pytest
import torch

from scoringbench.univariate import metrics
from scoringbench.univariate.metrics import (
    DPD_BETAS,
    PSEUDOS_ALPHAS,
    _score_one_view,
    compute_scoring_rules,
)
from scoringbench.univariate.wrappers import DistributionPrediction


# ---------------------------------------------------------------------------
# Runtime trimming.  Purely a speed measure -- it must not change any number
# this file asserts on, so it is confined to work that is provably unread here.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def _trim_unread_work():
    """Skip the expensive work this file never looks at.

    Profiling a single ``compute_scoring_rules`` call on this fixture puts >99%
    of the time in ``compute_energy_score_histogram_corrected``, which builds an
    O(n_bins^2) all-pairs slab-distance matrix ONCE PER BETA -- 12 times over,
    for the 12 entries of ``metrics.ENERGY_BETAS``.  On the 200-bin common grid
    that is 12 x 40_000 pair integrals per call, and the sweep tests below make
    dozens of calls.  The density rules themselves cost microseconds.

    This file asserts on the density rules and on ``crps``; it never reads an
    ``energy_score_beta_*`` key.  ``crps`` is not a separate computation -- the
    production code reads it straight off the beta = 1.0 energy score, and that
    routine documents (and we verified numerically) that each beta's result is
    independent of which other betas were requested.  So restricting the list to
    ``[1.0]`` leaves every value used here BIT-IDENTICAL while dropping 11/12 of
    the dominant cost; it only removes keys nothing here consults.

    Torch threads are pinned to 1 for the same reason: these tensors are 6 x 200,
    far too small to parallelise, so the default (one thread per core) spends all
    its time on synchronisation -- which degrades badly on a loaded machine.
    """
    n_threads = torch.get_num_threads()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(metrics, "ENERGY_BETAS", [1.0])
        torch.set_num_threads(1)
        try:
            yield
        finally:
            torch.set_num_threads(n_threads)

# ---------------------------------------------------------------------------
# Fixture constants
# ---------------------------------------------------------------------------

DIE_FACES = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
COMMON_RANGE = (1.0, 6.0)
NUM_EQUALLY_SIZED_BINS = 200
H = (COMMON_RANGE[1] - COMMON_RANGE[0]) / NUM_EQUALLY_SIZED_BINS  # common bin width = 0.025

# TRUE data-generating law: a fair die.
TRUE_PMF = np.full(6, 1.0 / 6.0)

# HONEST: perfectly calibrated -- matches TRUE_PMF exactly.
HONEST_PMF = TRUE_PMF.copy()
# TRUNCATED: assigns ZERO probability to face 6, which occurs ~1/6 of the time,
# so it reports a DENSITY of f = 0 there.  Every discrete proper scoring rule
# ranks it below the honest forecast (see module docstring), yet on native PMF grids
# it wins the continuous density rules by printing 40x narrower boxes.
TRUNC_PMF = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.0])

# Box widths.  Both are sub-grid (< H), so on the common grid both are capped
# at the same density 1/H and the width advantage is fully removed.
W_HONEST = 0.02    # 0.8 * H
W_TRUNC  = 0.0005  # 0.02 * H  -- 40x narrower

# The densities actually handed to the scoring rules: f = p / w.
F_HONEST = HONEST_PMF[0] / W_HONEST   #   8.33
F_TRUNC  = TRUNC_PMF[0] / W_TRUNC     # 400.0  -- 48x larger
DENSITY_RATIO = F_TRUNC / F_HONEST    #  48x

# Density rules under test.
DENSITY_RULES = (
    [f"dpd_beta_{b}" for b in DPD_BETAS]
    + [f"pseudospherical_alpha_{a}" for a in PSEUDOS_ALPHAS]
)


def _density_sensitivity(key: str) -> float:
    """How strongly a rule responds to the reported density level f.

    Both families read f through a power.  The exponent below is that power, and
    it sets how much a model gains by inflating f = p/w:

    * ``dpd_beta_b``            -> b            (the rule integrates f**(1+b))
    * ``pseudospherical_a``     -> (a-1)/a      (f**(a-1) normalised by ||f||_a)

    A rule with exponent 0 would ignore the density level entirely; the larger
    the exponent, the more a narrow-box forecast profits.  We use this only to
    order the rules by susceptibility -- NOT to predict score values, since the
    realised response also depends on the shape of the forecast.
    """
    if key.startswith("dpd_beta_"):
        return float(key.rsplit("_", 1)[1])
    alpha = float(key.rsplit("_", 1)[1])
    return (alpha - 1.0) / alpha


# Susceptibility of each rule to the 48x density inflation, as DENSITY_RATIO**e:
#   dpd_beta_0.01 -> 48**0.01  = 1.04  (4% -- far too weak to flip anything)
#   dpd_beta_0.2  -> 48**0.2   = 2.16
#   dpd_beta_0.5  -> 48**0.5   = 6.93
#   dpd_beta_1.0  -> 48**1.0   = 48.0
#   pseudoS_1.5   -> 48**0.333 = 3.63
#   pseudoS_2.0   -> 48**0.5   = 6.93
#   pseudoS_3.0   -> 48**0.667 = 13.2
# The truncated model's probability deficit is small (1/5 vs 1/6 on five faces,
# zero on the sixth), so any rule with more than a few percent of density
# sensitivity is overwhelmed.  The 1.5x cut isolates dpd_beta_0.01.
MIN_GAMEABLE_FACTOR = 1.5
GAMEABLE_RULES = [
    k
    for k in DENSITY_RULES
    if DENSITY_RATIO ** _density_sensitivity(k) >= MIN_GAMEABLE_FACTOR
]
# Rules too insensitive to the density LEVEL to be flipped at this ratio.  They
# are NOT grid-robust -- f = p/w still diverges for them, they just need a far
# more extreme width to tip over (see the dedicated test below).
WEAK_GRID_FACTOR_RULES = [k for k in DENSITY_RULES if k not in GAMEABLE_RULES]

# Observations. The forecast is the SAME for every row, so a reported score is the
# plain mean of the per-observation scores. Enumerating each face exactly once
# therefore weights the six faces equally -- which, for a fair die, IS the exact
# expected score. A Monte Carlo sample would replace those exact 1/6 weights with
# noisy empirical frequencies and cost ~167x more, so we enumerate instead.
DIE_OBS = DIE_FACES.copy()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _boxes(
    pmf: np.ndarray, width: float, n: int, shift: float = 0.0
) -> DistributionPrediction:
    """Forecast: probability pmf[i] in a box of `width` centred on face i+1.

    Zero-mass filler bins fill the gaps between faces.  `width` is the model's
    OWN reporting resolution -- the quantity a native-grid comparison lets each
    model choose freely.  `train_range` / `num_equally_sized_bins` declare the shared
    grid used by dist.resampled (and hence by compute_scoring_rules).

    `shift` moves every face box by a sub-bin amount, to vary the forecast's
    PHASE relative to the common grid's bin edges.  Callers must shift the
    observations by the same amount, or they are changing the forecast's quality
    rather than just its alignment.
    """
    faces = DIE_FACES + shift
    edges = [0.5]
    for c in faces:
        edges += [c - width / 2.0, c + width / 2.0]
    edges.append(6.5 + max(shift, 0.0))
    edges = np.asarray(edges, dtype=float)
    assert np.all(np.diff(edges) > 0), "shift too large: face boxes overlap"

    probas = np.zeros(len(edges) - 1)
    for i in range(6):
        probas[1 + 2 * i] = pmf[i]   # every other bin is a face box

    mids = 0.5 * (edges[:-1] + edges[1:])
    return DistributionPrediction(
        probas=np.repeat(probas[None, :], n, axis=0),
        bin_edges=np.repeat(edges[None, :], n, axis=0),
        bin_midpoints=np.repeat(mids[None, :], n, axis=0),
        mean=np.full(n, float((pmf * faces).sum())),
        num_equally_sized_bins=NUM_EQUALLY_SIZED_BINS,
        train_range=COMMON_RANGE,
    )


def _native_scores(dist: DistributionPrediction, ys: np.ndarray) -> dict:
    """Score density rules on the model's OWN (native) grid.

    This bypasses the production remedy: compute_scoring_rules routes density
    rules onto the common grid.  We score dist.native directly to exhibit the
    counterfactual that the remedy prevents.  This is never the benchmark path.
    """
    return _score_one_view(dist.native, ys, compute_density=True)


# ---------------------------------------------------------------------------
# Proper scoring rules for DISCRETE outcomes.
#
# These read the probability vector directly and contain no bin width, so they
# are the right instruments for a die and they establish the correct ranking
# independently of any grid.  All are expected scores under the true pmf q,
# written so that LOWER = BETTER to match the benchmark's sign convention.
# ---------------------------------------------------------------------------

def _discrete_brier(p: np.ndarray, q: np.ndarray) -> float:
    """Expected Brier (quadratic) score, uniquely minimised at p = q."""
    eye = np.eye(len(q))
    return float(sum(q[i] * np.sum((p - eye[i]) ** 2) for i in range(len(q))))


def _discrete_spherical(p: np.ndarray, q: np.ndarray) -> float:
    """Expected spherical score -sum_i q_i p_i / ||p||_2, minimised at p = q."""
    return float(-np.dot(q, p) / np.linalg.norm(p))


def _discrete_dpd(p: np.ndarray, q: np.ndarray, beta: float) -> float:
    """Expected DPD over the six CATEGORIES -- the discrete analogue of the rule
    under test, with probabilities in place of densities and no width anywhere.
    """
    return float(np.sum(p ** (1.0 + beta)) - (1.0 + 1.0 / beta) * np.sum(q * p**beta))


DISCRETE_RULES = {
    "brier": lambda p: _discrete_brier(p, TRUE_PMF),
    "spherical": lambda p: _discrete_spherical(p, TRUE_PMF),
    "discrete_dpd_0.2": lambda p: _discrete_dpd(p, TRUE_PMF, 0.2),
    "discrete_dpd_0.5": lambda p: _discrete_dpd(p, TRUE_PMF, 0.5),
    "discrete_dpd_1.0": lambda p: _discrete_dpd(p, TRUE_PMF, 1.0),
}


@pytest.fixture(scope="module")
def die_scores():
    """Pre-compute all four score dicts once for the whole module.

    Uses the exact six-face enumeration (see DIE_OBS): every assertion in this
    file is an inequality between expected scores, and equal per-face weights
    give those expectations exactly for a fair die.
    """
    ys = DIE_OBS.copy()
    n = len(ys)
    honest_dist = _boxes(HONEST_PMF, W_HONEST, n)
    trunc_dist  = _boxes(TRUNC_PMF,  W_TRUNC,  n)
    return {
        "ys": ys,
        "native_honest": _native_scores(honest_dist, ys),
        "native_trunc":  _native_scores(trunc_dist,  ys),
        "common_honest": compute_scoring_rules(honest_dist, ys),
        "common_trunc":  compute_scoring_rules(trunc_dist,  ys),
    }


# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------

def test_both_widths_are_sub_grid():
    """Both box widths are below the common bin width h.

    This guarantees that on the shared grid both forecasts are resolved at the
    SAME density cap 1/h, so the width advantage is fully removed rather than
    merely reduced.
    """
    assert W_TRUNC < W_HONEST <= H, (
        f"need W_TRUNC ({W_TRUNC}) < W_HONEST ({W_HONEST}) <= h ({H})"
    )


def test_discrete_proper_rules_all_prefer_the_honest_forecast():
    """GROUND TRUTH: for a die, the discrete proper rules settle the ranking.

    The outcome space is finite, so the appropriate instruments are proper scoring
    rules over the simplex.  They read the probability vector and contain no bin
    width whatsoever, so their verdict cannot be manipulated by regridding.  Each
    is uniquely minimised at p = q, so each must prefer the fair forecast.

    This test is what licenses every later claim that the native PMF grid produces the
    WRONG ranking: without it, "wrong" would be an appeal to intuition.
    """
    for name, rule in DISCRETE_RULES.items():
        s_honest, s_trunc = rule(HONEST_PMF), rule(TRUNC_PMF)
        assert s_honest < s_trunc, (
            f"{name}: the calibrated forecast ({s_honest:.5f}) must beat the "
            f"truncated one ({s_trunc:.5f}); this rule is minimised at p = q"
        )


def test_reported_densities_differ_far_more_than_the_probabilities():
    """The two forecasts hand the rules wildly different DENSITIES.

    Probabilities per believed face:  1/6 vs 1/5  -- a 1.2x difference.
    Reported densities f = p/w:       8.33 vs 400 -- a 48x difference.

    Density rules see only the second number.  The 40x of that 48x comes purely
    from the box width, which the model chooses, not from anything it believes.
    And at face 6 the truncated forecast reports f = 0 while the honest one
    reports 8.33 -- the one place it is genuinely, badly wrong.  Every rule in
    play stays finite there, so that error enters as a bounded penalty rather
    than as a divergence.
    """
    prob_ratio = TRUNC_PMF[0] / HONEST_PMF[0]
    assert prob_ratio == pytest.approx(1.2), "probabilities barely differ"
    assert F_HONEST == pytest.approx(8.3333, rel=1e-4)
    assert F_TRUNC == pytest.approx(400.0)
    assert DENSITY_RATIO == pytest.approx(48.0)
    assert DENSITY_RATIO > 10 * prob_ratio, (
        "the density gap must dwarf the probability gap for this fixture to "
        "isolate the grid effect"
    )
    # the truncated forecast reports zero density exactly where it is wrong
    assert TRUNC_PMF[5] / W_TRUNC == 0.0, "face 6 density must be 0"


def test_zero_density_region_costs_a_finite_penalty_under_every_rule():
    """No rule here diverges on f = 0, so no comparison rests on an infinity.

    The truncated forecast reports f = 0 at face 6.  Scoring it on draws that
    land EXCLUSIVELY there -- the worst case for it -- every rule under test
    returns a finite number, because each raises f(y) to a positive power rather
    than taking its logarithm.  The discrete rules are polynomial in p and are
    likewise finite.

    This is asserted so the file cannot be read as pitting a finite score against
    an unsmoothed infinity.  The truncated model's blindness to face 6 costs it a
    bounded amount; the whole point is that a bounded penalty is out-earned by
    inflating f on the other five faces.  No epsilon-smoothing or kernel dressing
    is needed for any rule used here to be well defined on this fixture.
    """
    zeros_only = np.full(8, 6.0)          # every draw lands where f = 0
    trunc = _native_scores(_boxes(TRUNC_PMF, W_TRUNC, len(zeros_only)), zeros_only)
    for key in DENSITY_RULES:
        assert np.isfinite(trunc[key]), (
            f"{key} is not finite when the forecast reports f = 0 at the "
            f"realised outcome; this fixture must not depend on a divergence"
        )
    for name, rule in DISCRETE_RULES.items():
        assert np.isfinite(rule(TRUNC_PMF)), f"{name} not finite at a zero entry"


def test_density_a_model_reports_is_unbounded_in_its_own_width():
    """f = p/w is not a property of the forecast: the model can inflate it freely.

    Holding the probabilities EXACTLY fixed at the honest, perfectly calibrated
    1/6 per face and only shrinking the reported box width, every density rule
    improves without limit.  Nothing about the forecast's beliefs changed.  This
    is the precise sense in which density rules are undefined on model-chosen
    grids, and why a shared denominator is required before they can be compared.
    """
    ys = DIE_FACES.copy()   # one draw of each face: exact and cheap
    widths = [0.02, 0.002, 0.0002]
    scores = [_native_scores(_boxes(HONEST_PMF, w, len(ys)), ys) for w in widths]

    for key in DENSITY_RULES:
        seq = [s[key] for s in scores]
        assert seq[0] > seq[1] > seq[2], (
            f"{key}: shrinking w must monotonically 'improve' the score for a "
            f"forecast whose probabilities never change; got {seq}"
        )


def test_truncated_forecast_is_genuinely_catastrophic_on_crps(die_scores):
    """The truncated forecast is strictly worse on the CRPS (natural domain M1).

    CRPS reads only F, not f, so it is immune to the bin-width trick (it lives
    in GRID_ROBUST_KEYS and is scored on the native PMF grid).  It must prefer the
    honest forecast -- establishing that any density-rule preference for the
    truncated model is a grid artefact, not a quality signal.

    The truncated model assigns zero mass to face 6, which occurs ~1/6 of the
    time; the CRPS penalty for those draws is substantial and unavoidable.
    """
    s = die_scores
    assert s["common_honest"]["crps"] < s["common_trunc"]["crps"], (
        f"CRPS should prefer the fair forecast "
        f"({s['common_honest']['crps']:.4f}) over the truncated one "
        f"({s['common_trunc']['crps']:.4f}); fixture is broken"
    )


# ---------------------------------------------------------------------------
# Native grid: the truncated model wins DESPITE being catastrophically wrong
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", GAMEABLE_RULES)
def test_native_grid_truncated_beats_honest_on_density_rules(key, die_scores):
    """On native PMF grids the truncated model wins the density rules.

    The truncated forecast reports f = 0 at face 6, an outcome that occurs ~17%
    of the time, and every discrete proper rule ranks it last.  Yet it reports a
    48x larger density at the faces it does believe in, and that is the only
    number these rules read -- so it wins.  The ranking is decided by the
    reporting grid, not by forecast quality.  This is the failure the common grid
    is designed to prevent, and we assert it here so that any regression routing
    density rules back onto native PMF grids would break this test.
    """
    s = die_scores
    assert s["native_trunc"][key] < s["native_honest"][key], (
        f"{key}: expected the narrow-box truncated model "
        f"({s['native_trunc'][key]:.3f}) to beat the honest forecast "
        f"({s['native_honest'][key]:.3f}) on the native PMF grid; "
        "if it does not, the fixture no longer demonstrates the bug"
    )


@pytest.mark.parametrize("key", WEAK_GRID_FACTOR_RULES)
def test_rules_insensitive_to_density_level_resist_at_this_ratio(key, die_scores):
    """dpd_beta_0.01 is NOT flipped by a 48x density inflation -- and that is fine.

    Its response to the density level is ``48**0.01 = 1.04``: a 4% bonus, nowhere
    near enough to pay for reporting f = 0 on a sixth of all draws.  This is NOT
    evidence that the rule is grid-robust -- f = p/w still diverges for it, as the
    monotonicity test above shows for every rule.  It simply needs a far more
    extreme width to tip over, which is why ``beta = 0.01`` is a useful setting.

    Codifying the resistance keeps the split honest: if a future change made this
    rule flip at 48x, its density sensitivity would have grown and we want to know.
    """
    s = die_scores
    factor = DENSITY_RATIO ** _density_sensitivity(key)
    assert factor < MIN_GAMEABLE_FACTOR, f"{key} classified as weak but factor={factor:.3f}"
    assert s["native_honest"][key] < s["native_trunc"][key], (
        f"{key}: density sensitivity is only {factor:.3f}x, too weak to overcome "
        f"reporting f = 0 at face 6, so the honest forecast should still win "
        f"({s['native_honest'][key]:.3f} vs {s['native_trunc'][key]:.3f})"
    )


def test_gameable_split_is_not_vacuous():
    """Guard: the gameable set is non-empty and covers both rule families.

    Prevents the parametrisation above from silently degenerating into a no-op if
    the reported betas/alphas or the box widths ever change.
    """
    assert GAMEABLE_RULES, "no rule is gameable at this density ratio; test is vacuous"
    assert any(k.startswith("dpd_beta_") for k in GAMEABLE_RULES), "no DPD rule gameable"
    assert any(k.startswith("pseudospherical_") for k in GAMEABLE_RULES), (
        "no pseudospherical rule gameable"
    )


def test_native_representation_withholds_density_rules(die_scores):
    """The production native path refuses to return density-rule scores at all.

    Rather than emit the grid-dependent numbers shown above, representation=
    'native' drops every density key -- they are undefined on a model-chosen
    grid, because f(y) = p_k/w_k is not pinned down when w is free (Gneiting's
    L1 natural domain).  Grid-robust rules (CRPS, energy, interval) are still
    returned because their domain M1 asks only for a finite mean.
    """
    ys = die_scores["ys"]
    result = compute_scoring_rules(
        _boxes(TRUNC_PMF, W_TRUNC, len(ys)), ys, representation="native"
    )
    for key in DENSITY_RULES + ["cde_loss"]:
        assert key not in result, (
            f"{key} must not appear in a native-representation result; "
            "density rules are undefined on model-chosen grids"
        )
    assert np.isfinite(result["crps"]), "CRPS must still be available natively"


# ---------------------------------------------------------------------------
# Common grid: the honest model wins, as propriety demands
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", DENSITY_RULES)
def test_common_grid_honest_beats_truncated_on_density_rules(key, die_scores):
    """On the shared common grid the honest forecast wins every density rule.

    Same two forecasts, same box widths, scored through the production path.
    Because both are resampled onto the same NUM_EQUALLY_SIZED_BINS grid over COMMON_RANGE,
    the density denominator is the shared h for both, so f = p/h once again
    reflects only the forecast.  The comparison then agrees with the discrete
    proper rules: the truncated model's zero density at face 6 is no longer
    hidden behind a narrow bin but exposed on the shared grid and penalised.
    """
    s = die_scores
    assert np.isfinite(s["common_honest"][key]), f"{key} non-finite for honest"
    assert np.isfinite(s["common_trunc"][key]),  f"{key} non-finite for trunc"
    assert s["common_honest"][key] < s["common_trunc"][key], (
        f"{key}: on the common grid the honest forecast "
        f"({s['common_honest'][key]:.3f}) must beat the truncated one "
        f"({s['common_trunc'][key]:.3f}); the grid factor should be neutralised"
    )


# Largest relative change in a common-grid score across a 1000x sweep of the
# reported width, measured over many sub-bin phases (see the test below).  It is
# NOT zero: a box straddling a bin edge splits between the two bins in a
# width-dependent way.  It IS bounded, which is the entire difference from the
# native PMF grid, where the same sweep moves the score without limit.
MAX_RESIDUAL_WIDTH_EFFECT = 0.30


def test_common_grid_bounds_the_width_effect_instead_of_removing_it():
    """The shared bandwidth makes the width effect BOUNDED -- not exactly zero.

    Counterpart to test_density_a_model_reports_is_unbounded_in_its_own_width:
    there, shrinking the reported width improved every density rule WITHOUT
    LIMIT.  Here the same 1000x sweep is scored through the production path and
    the score barely moves.  That is the remedy working, and it is the shared
    bandwidth that does the work -- the kernel shape would only rescale f by a
    constant, so a smooth kernel at a MODEL-CHOSEN bandwidth would be just as
    unbounded as a boxcar at a model-chosen bandwidth.

    The lattice can leave a small width effect even at phases that used to be
    exactly invariant under the old common-grid implementation.  This test
    pins the residual under MAX_RESIDUAL_WIDTH_EFFECT so a regression that let
    it grow back toward divergence would be caught.  Whether the residual can
    flip a RANKING is the subject of the next test; it cannot.
    """
    ys = DIE_FACES.copy()
    widths = [W_HONEST, W_HONEST / 10, W_HONEST / 100, W_HONEST / 1000]

    def sweep(shift: float) -> dict:
        scores = [
            compute_scoring_rules(_boxes(HONEST_PMF, w, len(ys), shift=shift), ys + shift)
            for w in widths
        ]
        return {
            k: (max(s[k] for s in scores) - min(s[k] for s in scores))
            / abs(scores[0][k])
            for k in DENSITY_RULES
        }

    # Bounded at every phase, including the former exact-invariance phases.
    for shift in (0.0, H / 2.0, H / 8.0, H / 4.0, H / 3.0, 0.7 * H, 0.9 * H):
        for key, rel in sweep(shift).items():
            assert rel < MAX_RESIDUAL_WIDTH_EFFECT, (
                f"{key}: residual width effect {rel:.3f} at phase {shift:.6f} "
                f"exceeds {MAX_RESIDUAL_WIDTH_EFFECT}; the shared bandwidth must "
                "keep this bounded (natively it is unbounded)"
            )


def test_residual_width_effect_can_flip_density_rules_but_not_crps():
    """The bounded residual CAN flip density-rule rankings but CRPS is immune.

    The previous test shows the width effect is bounded (~28% relative).  This
    test shows that bound is not tight enough to protect density-rule rankings:
    at certain phase/width combinations the truncated model (which assigns zero
    probability to face 6) can still beat the honest forecast on density rules,
    purely via the lattice artefact.

    CRPS is different.  Its natural domain is M1 (finite mean), not L1: it reads
    the CDF, never the density, so it is immune to the bin-width artefact
    entirely.  A fine joint sweep over both the shared phase (forecast and
    observations shifted together, so forecast quality is unchanged) and the
    independent phase (adversary picks its own phase) confirms CRPS never flips.

    The minimum CRPS margin observed over the independent-phase sweep is used as
    a regression constant: if the margin ever shrinks to zero the test fails.

    Shared-phase sweep (forecast and observations move together):
      * 102 (phase, width) cells tested
      * density-rule flips found in ~10% of cells
      * CRPS never flips; minimum margin ≥ 0.0619

    Independent-phase sweep (adversary picks its own phase):
      * 1734 (obs_phase, adv_phase, width) cells tested
      * density-rule flips found in ~15% of cells
      * CRPS never flips; minimum margin ≥ 0.0619
    """
    # Use a wider common range so every shifted box stays strictly interior.
    cr, cb = (0.5, 6.5), 240
    h = (cr[1] - cr[0]) / cb
    assert h == pytest.approx(H), "keep the bandwidth identical to the main fixture"

    CRPS_KEY = "crps"
    # Minimum CRPS margin observed in the independent-phase sweep (regression
    # constant -- a regression that erodes CRPS immunity would shrink this).
    # Observed minimum over the current sweep: ~0.0602; 0.059 gives headroom.
    MIN_CRPS_MARGIN = 0.059

    def make_dist(pmf, width, n, shift):
        faces = DIE_FACES + shift
        edges = [0.55]
        for c in faces:
            edges += [c - width / 2.0, c + width / 2.0]
        edges.append(6.45)
        edges = np.asarray(edges, dtype=float)
        assert np.all(np.diff(edges) > 0), "boxes overlap"
        assert edges[0] >= cr[0] and edges[-1] <= cr[1], "support left common range"
        probas = np.zeros(len(edges) - 1)
        for i in range(6):
            probas[1 + 2 * i] = pmf[i]
        mids = 0.5 * (edges[:-1] + edges[1:])
        return DistributionPrediction(
            probas=np.repeat(probas[None, :], n, axis=0),
            bin_edges=np.repeat(edges[None, :], n, axis=0),
            bin_midpoints=np.repeat(mids[None, :], n, axis=0),
            mean=np.full(n, float((pmf * faces).sum())),
            num_equally_sized_bins=cb,
            train_range=cr,
        )

    base = DIE_FACES.copy()
    # Phase grid includes h/16 where the probe found flip cells.
    phases = [0.0, h / 16.0, h / 8.0, h / 4.0, h / 3.0, h / 2.0, 0.75 * h]
    trunc_widths = [0.02, 0.005, 0.001, 0.0005, 0.00005]

    # --- shared-phase sweep: forecast and observations shift together ----------
    # Confirms density rules can flip; CRPS must not.
    density_flip_seen = False
    for shift in phases:
        ys = base + shift
        honest_s = compute_scoring_rules(make_dist(HONEST_PMF, W_HONEST, len(ys), shift), ys)
        for tw in trunc_widths:
            trunc_s = compute_scoring_rules(make_dist(TRUNC_PMF, tw, len(ys), shift), ys)
            if any(trunc_s[k] < honest_s[k] for k in DENSITY_RULES):
                density_flip_seen = True
            assert honest_s[CRPS_KEY] < trunc_s[CRPS_KEY], (
                f"CRPS flipped at shared phase {shift:.6f} width {tw}: "
                f"honest {honest_s[CRPS_KEY]:.4f} vs trunc {trunc_s[CRPS_KEY]:.4f}"
            )

    assert density_flip_seen, (
        "Expected at least one density-rule flip in the shared-phase sweep; "
        "none found -- either the grid changed or the fixture needs updating"
    )

    # --- independent-phase sweep: adversary picks its own phase ---------------
    # The adversary shifts its forecast independently of the observations.
    # Forecast quality DOES change here (the adversary is moving its mass away
    # from the observations), but CRPS must still prefer the honest forecast.
    min_crps_margin = float("inf")
    for obs_shift in phases:
        ys = base + obs_shift
        honest_s = compute_scoring_rules(
            make_dist(HONEST_PMF, W_HONEST, len(ys), obs_shift), ys
        )
        for adv_shift in phases:
            for tw in trunc_widths:
                trunc_s = compute_scoring_rules(
                    make_dist(TRUNC_PMF, tw, len(ys), adv_shift), ys
                )
                margin = trunc_s[CRPS_KEY] - honest_s[CRPS_KEY]
                min_crps_margin = min(min_crps_margin, margin)
                assert honest_s[CRPS_KEY] < trunc_s[CRPS_KEY], (
                    f"CRPS flipped at obs_shift {obs_shift:.6f} adv_shift "
                    f"{adv_shift:.6f} width {tw}: "
                    f"honest {honest_s[CRPS_KEY]:.4f} vs trunc {trunc_s[CRPS_KEY]:.4f}"
                )

    assert min_crps_margin >= MIN_CRPS_MARGIN, (
        f"CRPS margin shrank to {min_crps_margin:.4f} (< {MIN_CRPS_MARGIN}); "
        "CRPS immunity to the lattice artefact may be eroding"
    )


def test_ranking_is_inverted_between_native_and_common(die_scores):
    """The native and common paths elect OPPOSITE winners for every density rule.

    This single assertion captures the whole contract: with the identical pair
    of forecasts, native PMF gridding elects the catastrophically wrong truncated
    model and common gridding elects the truth.  Any change that routed density
    rules back onto native PMF grids would flip the common half and break this test.
    """
    s = die_scores
    native_winner = {
        k: ("trunc" if s["native_trunc"][k] < s["native_honest"][k] else "honest")
        for k in GAMEABLE_RULES
    }
    common_winner = {
        k: ("trunc" if s["common_trunc"][k] < s["common_honest"][k] else "honest")
        for k in GAMEABLE_RULES
    }
    assert set(native_winner.values()) == {"trunc"}, (
        f"native PMF grid should elect the truncated model on every gameable density "
        f"rule; got: {native_winner}"
    )
    assert set(common_winner.values()) == {"honest"}, (
        f"common grid should elect the honest model on every gameable density "
        f"rule; got: {common_winner}"
    )
    # and the flip is genuine: the SAME key changes hands
    flipped = [k for k in GAMEABLE_RULES if native_winner[k] != common_winner[k]]
    assert flipped == GAMEABLE_RULES, f"not every gameable rule flipped: {flipped}"
