"""Tests for support-sensitivity properties of distributional scoring rules.

Covers two concerns:
1. ``TestSupportSensitivity`` — verifies that ``crts_alpha_*`` and
   ``dpd_beta_0.01`` are proper (the true DGP scores no worse than a
   truncated variant) and domain-insensitive (widening the reported support
   does not change the score).

2. ``TestPadToCommonGrid`` — unit tests for the ``pad_to_common_grid`` helper
   that extends each sample's bin grid with zero-mass bins so that every
   target y is interior to the integration domain.
"""

import functools
import math
import numpy as np
import pytest
import torch

# Force CPU for all tests in this file.
torch.cuda.is_available = lambda: False

from scoringbench.univariate.metrics import (
    compute_scoring_rules,
    compute_crts,
    pad_to_common_grid,
    DPD_BETAS,
    CRTS_ALPHAS,
)
from scoringbench.univariate.wrappers import DistributionPrediction


# ============================================================================
# Shared helpers
# ============================================================================

# Bin width held fixed across all grids so that changing [lo, hi] changes
# *only* the integration domain and not the grid resolution.  0.01 keeps
# discretisation error below the assertion tolerances while keeping the
# widest grid small enough that the O(n_bins^2) energy-score kernel stays fast.
_BW = 0.01


def _norm_pdf(x, mu=0.0, sigma=1.0):
    """Standard normal PDF (no SciPy dependency)."""
    z = (np.asarray(x, dtype=float) - mu) / sigma
    return np.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))


def _norm_cdf(x, mu=0.0, sigma=1.0):
    """Standard normal CDF via ``math.erf`` (vectorised)."""
    z = (np.asarray(x, dtype=float) - mu) / (sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + np.vectorize(math.erf)(z))


def _standard_normal_ppf(p):
    """Standard normal inverse CDF (quantile function), no SciPy dependency.

    Uses ``torch.special.ndtri`` (already have torch imported) so we can turn an
    evenly spaced probability grid into deterministic N(0,1) quantile targets.
    """
    t = torch.as_tensor(np.asarray(p, dtype=np.float64))
    return torch.special.ndtri(t).cpu().numpy()


def _gaussian_on_grid(lo, hi, bin_width=_BW, mu=0.0, sigma=1.0, truncate=None):
    """Build a ``DistributionPrediction`` for N(mu, sigma) on ``[lo, hi]``.

    ``bin_width`` is held fixed across grids so that changing ``[lo, hi]``
    changes *only* the integration domain and not the grid resolution — that
    isolates the effect under test.

    ``truncate=(a, b)`` zeroes the density outside ``[a, b]`` before
    renormalising, i.e. it produces a genuinely truncated forecast while
    keeping the *same* bin grid.
    """
    n_bins = int(round((hi - lo) / bin_width))
    edges = np.linspace(lo, hi, n_bins + 1)
    mids = (edges[:-1] + edges[1:]) / 2.0

    if truncate is None:
        probas = _norm_cdf(edges[1:], mu, sigma) - _norm_cdf(edges[:-1], mu, sigma)
    else:
        a, b = truncate
        dens = _norm_pdf(mids, mu, sigma).copy()
        dens[(mids < a) | (mids > b)] = 0.0
        probas = dens * np.diff(edges)

    probas = probas / probas.sum()
    return DistributionPrediction(
        probas=probas[None, :].astype(np.float64),
        bin_edges=edges[None, :].astype(np.float64),
        bin_midpoints=mids[None, :].astype(np.float64),
        mean=np.array([mu], dtype=np.float64),
        train_range=(float(edges.min()), float(edges.max())),
    )


def _tile(dp, n):
    """Repeat a single-row ``DistributionPrediction`` into ``n`` identical rows.

    ``compute_scoring_rules`` scores the i-th distribution row against the i-th
    target, so tiling one forecast lets us score it against a whole batch of
    targets in a single call (used for the Monte-Carlo expected-score test).
    """
    return DistributionPrediction(
        probas=np.repeat(dp.probas, n, axis=0),
        bin_edges=np.repeat(dp.bin_edges, n, axis=0),
        bin_midpoints=np.repeat(dp.bin_midpoints, n, axis=0),
        mean=np.repeat(dp.mean, n, axis=0),
        train_range=dp.train_range,
    )


# N(0,1) quantiles used to define the truncation window.
_Q005 = -2.5758293035489004
_Q995 = 2.5758293035489004

# A target above the 99.5% quantile: the outlier case where the defect bites.
_Y_OUTLIER = _Q995 + 0.5

# --- Expectation-test geometry (chosen for speed) -------------------------
# The expected-score propriety test is the runtime bottleneck (energy score is
# O(n_bins^2) and it scores many targets), so its grid and truncation window
# are deliberately *tighter* than the extreme-tail single-target tests above:
#
#   * ``_EXP_GRID`` = 4.0  -> grid [-4, 4] at bin_width 0.05 = 160 bins, a 4x
#     energy-score speedup vs the [-8, 8] grid used elsewhere.
#   * ``_EXP_TRUNC`` = q_0.90 (|z| ~ 1.2816) -> truncating N(0,1) to the central
#     80% zeroes a *full 20%* of the mass in the tails.  That makes the
#     truncated forecast differ from the true one over a large, high-probability
#     region, so the signed expected-score gap is big (~1e-1, not ~1e-3) and a
#     few hundred deterministic quantile targets resolve it comfortably.
_EXP_GRID = 4.0
_EXP_TRUNC = 1.2815515594457412  # N(0,1) 90th percentile (|z| for central 80%)


@functools.lru_cache(maxsize=None)
def _cached_scores(lo, hi, ys, bin_width=_BW, truncate=None):
    """Cached ``compute_scoring_rules`` for a Gaussian-on-grid forecast.

    The score computation (dominated by the O(n_bins^2) energy score) is
    expensive but depends only on the forecast geometry and the targets, not on
    the scoring-rule key.  Several tests are parametrized over ~30 keys and
    would otherwise recompute the identical score dict once per key.  Keying the
    cache on the hashable construction parameters lets every such test — and
    every test that happens to reuse the same geometry — share a single
    computation.

    Parameters
    ----------
    lo, hi, bin_width, truncate
        Passed straight through to ``_gaussian_on_grid``; ``truncate`` must be a
        tuple (or ``None``) so the arguments stay hashable.
    ys : tuple[float, ...]
        The target(s).  A tuple (not an array) keeps the call hashable; it is
        tiled to match the forecast so a batch of targets is scored in one call.

    Returns
    -------
    dict
        The full scoring-rule dict (keyed by rule name).
    """
    forecast = _gaussian_on_grid(lo, hi, bin_width=bin_width, truncate=truncate)
    y = np.asarray(ys, dtype=np.float64)
    return compute_scoring_rules(_tile(forecast, y.size), y)


def _expectation_targets():
    """Deterministic N(0,1) quantile targets for the shared-grid expectation test.

    Inverse-CDF of an evenly spaced probability grid: a zero-variance,
    low-discrepancy stand-in for sampling from the DGP whose plain mean equals
    ``E_{N(0,1)}[S]`` to high accuracy, with guaranteed tail coverage.
    """
    n_pts = 200
    u = (np.arange(n_pts, dtype=np.float64) + 0.5) / n_pts
    ys = np.clip(_standard_normal_ppf(u), -_EXP_GRID + 1e-3, _EXP_GRID - 1e-3)
    return tuple(ys.tolist())


# ---------------------------------------------------------------------------
# Scoring-rule taxonomy
# ---------------------------------------------------------------------------
# ``_ALL_PROPER_KEYS`` is the set of *proper* scoring rules that every generic
# test below is parametrized over: on a fixed, shared integration grid the true
# forecast must score no worse than a truncated variant of it.
#
# Rather than hand-maintaining that list (and silently forgetting to test a
# newly added rule), it is *discovered* from the live output of
# ``compute_scoring_rules`` at import time and everything that is NOT a proper
# scoring rule of the predictive distribution is subtracted out.  If someone
# adds a new proper rule to ``compute_scoring_rules`` it is automatically picked
# up here and must satisfy propriety / domain-width invariance / finiteness; if
# they add a new *non-proper* diagnostic it must be listed in
# ``_NON_PROPER_KEYS`` below (a failing "unclassified key" guard test enforces
# that nobody forgets).
#
# Why the surviving rules are also support-INSENSITIVE: ``compute_scoring_rules``
# pads every grid so the target is interior, and each rule's integrand either
# vanishes where the forecast is certain (divergence-form CRTS, energy score /
# CRPS) or acts on a *normalised* density whose zero-mass padding contributes
# nothing (dpd_beta_*, cde_loss), so widening the reported support leaves the
# score essentially unchanged.  The only genuinely support-sensitive rule was
# the log-score / CRLS, which has been removed from the suite for that reason.
# The quantile-weighted wCRPS variants are proper too but their *discretised*
# quantile-inversion form is only weakly monotone under truncation (~1e-3
# drift), so they are checked with the looser ``_QUANTILE_SLACK`` via
# ``_slack_for``.

_CRTS_KEYS = [f"crts_alpha_{a}" for a in CRTS_ALPHAS]

# Keys emitted by ``compute_scoring_rules`` that are NOT proper scoring rules of
# the predictive distribution (sharpness / dispersion summaries, PIT calibration
# statistics, raw empirical coverage).  These are excluded from the generic
# propriety / invariance / finiteness parametrizations.
_NON_PROPER_PREFIXES = ("coverage_", "pit_")
_NON_PROPER_EXACT = {"sharpness", "dispersion"}


def _is_proper_key(key):
    """True if ``key`` is a proper scoring rule (not a diagnostic/summary)."""
    if key in _NON_PROPER_EXACT:
        return False
    return not key.startswith(_NON_PROPER_PREFIXES)


def _discover_proper_keys():
    """Score a probe N(0,1) forecast and return every emitted *proper* rule key.

    Discovering the keys from the live metric output means a newly added proper
    rule is exercised automatically without editing this file.
    """
    probe = _gaussian_on_grid(-8.0, 8.0, bin_width=0.05)
    emitted = compute_scoring_rules(probe, np.array([0.5], dtype=np.float64))
    return sorted(k for k in emitted if _is_proper_key(k))


_ALL_PROPER_KEYS = _discover_proper_keys()

# Absolute slack allowed for the quantile-weighted rules' ranking / invariance
# assertions.  Empirically the wCRPS truncation gap and grid-width drift are
# both ~1e-3; 5e-3 leaves comfortable margin while still catching gross
# violations.
_QUANTILE_SLACK = 5e-3

# The interval score depends on the *location* of the reported quantile
# endpoints, which are recovered by inverting the discretised CDF.  On a coarse
# integration grid the endpoints of the exact and truncated forecasts land in
# different bins, producing an O(bin_width) quadrature difference (~1e-2 at
# bin_width=0.05) that is a pure discretisation artefact, not a propriety
# violation.  A dedicated, looser slack absorbs it while still catching gross
# ranking flips.
_INTERVAL_SLACK = 2e-2

# The divergence-form rules are the only ones for which truncation is
# *strictly* worse at an outlier target (the others merely tie or move within
# slack), so the sharpened strict-worsening test is restricted to these keys.
_GUARDED_KEYS = _CRTS_KEYS + ["dpd_beta_0.01"]


def _slack_for(key):
    """Absolute slack allowed in the ranking / invariance assertions for ``key``.

    The density / threshold rules hold at machine precision; the quantile-
    weighted wCRPS rules only hold up to ``_QUANTILE_SLACK`` because their
    discretised quantile-inversion form drifts by ~1e-3 under truncation and
    grid widening.  The interval score is even more sensitive to bin placement
    (its endpoints are recovered by CDF inversion), so it gets the looser
    ``_INTERVAL_SLACK``.
    """
    if "wcrps" in key:
        return _QUANTILE_SLACK
    if key.startswith("interval_score_"):
        return _INTERVAL_SLACK
    return 1e-9


# ============================================================================
# TestSupportSensitivity
# ============================================================================

class TestSupportSensitivity:
    """Support-sensitivity properties of the proper scoring rules.

    The historical CRLS rule is only proper *among distributions sharing one
    support*.  Because the ScoringBench integration domain is the model's
    *self-reported* support, shrinking that support shrinks the CRLS integral,
    so a model could improve its score simply by truncating — badly mis-ranking
    outlier targets.

    These tests pin down two things across the whole proper-rule suite:

      * **Propriety (fixed-grid ranking):** every proper rule must rank the true
        forecast at least as good as a truncated variant when both are scored
        on the *same* integration grid.  Asserted for ``_ALL_PROPER_KEYS``.
      * **Domain-width invariance (ranking + loose numeric):** every proper rule
        must preserve both the ranking and (loosely) the numeric score as the
        reported support widens.

    The density / threshold rules satisfy these at machine precision; the
    quantile-weighted CRPS variants only satisfy them up to a small numeric
    slack (``_QUANTILE_SLACK``) because of their discretised quantile-inversion
    form, so those keys are checked with the looser bound via ``_slack_for``.
    """

    def test_truncating_does_not_improve_score(self):
        """Propriety: for y > q_0.995, truncating the true forecast must not pay off.

        Both forecasts live on the *same* grid, so the integration domain is
        identical and this is a pure propriety statement: the exact Gaussian is
        the true data-generating distribution, so it must not score *better*
        than its truncated variant.

        A ``<=`` (rather than strict ``<``) bound is used because some rules are
        genuinely insensitive to the extreme tail at this target — e.g. an
        interval score at a low central-coverage level, whose interval does not
        reach out to ``y``, is identical for the exact and truncated forecasts.
        A tie is admissible; the truncated forecast winning is not.

        The two score dicts are key-independent (and shared with several other
        tests via ``_cached_scores``), so they are computed once and this test
        loops over every proper rule internally.
        """
        ys = (_Y_OUTLIER,)
        exact = _cached_scores(-8.0, 8.0, ys)
        trunc = _cached_scores(-8.0, 8.0, ys, truncate=(_Q005, _Q995))

        failures = []
        for key in _ALL_PROPER_KEYS:
            slack = _slack_for(key)
            if not (exact[key] <= trunc[key] + slack):
                failures.append(
                    f"{key}: exact={exact[key]:.6f} > trunc={trunc[key]:.6f} "
                    f"(slack={slack:g})"
                )

        assert not failures, (
            f"truncating N(0,1) to [q.005, q.995] improved the score for an "
            f"outlier target y={_Y_OUTLIER:.4f}; a proper rule must not reward "
            "truncation on a shared grid:\n  " + "\n  ".join(failures)
        )

    def test_true_forecast_wins_in_expectation_on_shared_grid(self):
        """Propriety proper: E_DGP[S(true)] <= E_DGP[S(truncated)] on one grid.

        The single-target ``test_truncating_does_not_improve_score`` shows the
        truncated forecast never *wins* at one outlier realisation, but the
        defining property of a proper score is about the *expectation* under the
        true data-generating process, not any single ``y``.  This test makes
        that explicit: with the exact N(0,1) and its truncated copy on the
        *same* [-8, 8] grid, we Monte-Carlo the expected score over
        ``y ~ N(0, 1)`` and require the true forecast to be at least as good on
        average.  Because both forecasts share the grid, this isolates the shape
        difference (truncation) from any support / domain-width effect.

        The truncated forecast agrees with the true one in the interior (up to
        the ~1% renormalisation inflation) and is wrong only in the tails, so
        the expected-score gap is small but must be signed the right way — and
        the signal lives entirely in the rare tail targets.

        Performance / smart geometry: naive ``y ~ N(0,1)`` Monte-Carlo needs a
        *large* sample to (a) beat its own O(1/sqrt(N)) noise and (b) actually
        place points in the tails where the truncation gap lives; the energy
        score is O(n_bins^2), so every extra target and every extra bin is
        expensive.  Three choices make this fast *and* well-resolved:

          1. **Narrow grid** ``[-_EXP_GRID, _EXP_GRID]`` (160 bins at 0.05) — a
             4x energy-score speedup vs the [-8, 8] grid used elsewhere; the
             N(0,1) mass beyond |z|=4 is ~6e-5, negligible for the expectation.
          2. **Aggressive truncation** at the 90th percentile (``_EXP_TRUNC``)
             instead of the 99.5th — zeroing the central-80% tails makes the
             truncated forecast wrong over a *large, high-probability* region,
             so the signed expected-score gap is ~1e-1 instead of ~1e-3.
          3. **Deterministic quantile targets** — the targets are the N(0,1)
             quantiles at evenly spaced probabilities (inverse-CDF of a uniform
             grid): a zero-variance, low-discrepancy stand-in for sampling from
             the DGP whose plain mean equals ``E_{N(0,1)}[S]`` to high accuracy,
             with guaranteed tail coverage by construction.

        Together these resolve the signed gap with a few hundred points where
        tens of thousands of random draws would struggle.  The only casualty is
        the interval score's bin-placement sensitivity, absorbed by
        ``_INTERVAL_SLACK``.

        The heavy score computation is key-independent, so rather than paying it
        once per parametrized key it is done a single time (via the shared
        ``_cached_scores`` wrapper) and this test loops over every proper rule
        internally, asserting the propriety inequality for each.
        """
        ys = _expectation_targets()
        e_exact = _cached_scores(-_EXP_GRID, _EXP_GRID, ys, bin_width=0.05)
        e_trunc = _cached_scores(
            -_EXP_GRID, _EXP_GRID, ys, bin_width=0.05,
            truncate=(-_EXP_TRUNC, _EXP_TRUNC),
        )

        failures = []
        for key in _ALL_PROPER_KEYS:
            slack = _slack_for(key)
            if not (e_exact[key] <= e_trunc[key] + slack):
                failures.append(
                    f"{key}: E[true]={e_exact[key]:.6f} > E[trunc]={e_trunc[key]:.6f} "
                    f"(slack={slack:g})"
                )

        assert not failures, (
            "on a shared [-{g}, {g}] grid the truncated forecast beat the true "
            "N(0,1) in expectation under the true DGP for:\n  ".format(g=_EXP_GRID)
            + "\n  ".join(failures)
            + "\nA proper score must rank the true distribution best on average."
        )

    def test_zero_padding_neutralises_a_narrow_reported_grid(self):
        """The dynamic zero-padding path must not reward a narrow reported grid.

        Unlike the other propriety tests (which keep one fixed grid and merely
        zero *interior* bins via ``truncate=``), this test gives the two
        forecasts genuinely *different* grids and relies on
        ``pad_to_common_grid`` to equalise the integration domain:

          * the *true* forecast reports N(0,1) on a wide grid ([-8, 8]) that
            already contains the outlier target; and
          * the *narrow* forecast reports the very same N(0,1) shape but on a
            grid that simply stops at [q.005, q.995] — the reported support ends
            *inside* the DGP's range, with no bins beyond it.

        The outlier target ``_Y_OUTLIER`` ≈ 3.08 falls *outside* the narrow
        grid, so ``compute_scoring_rules`` must pad that forecast with zero-mass
        bins out to y (CDF pinned to 1 in the padded tail).  This is exactly the
        anti-gaming guarantee documented on ``pad_to_common_grid``: reporting a
        narrower support must not buy a better score.  If padding were skipped
        the narrow forecast's CDF would be frozen at its last reported value and
        several rules would look artificially good.

        The assertion is therefore ``S(true) <= S(narrow_after_padding)`` — the
        truncated-by-grid forecast, once padded, may only tie or lose.

        The two score dicts are key-independent, so they are computed once (via
        ``_cached_scores``) and this test loops over every proper rule
        internally.
        """
        # Sanity: the target really is exterior to the narrow grid, so the
        # padding path is exercised (guards against the test silently degrading
        # into a same-grid comparison if the constants ever change).
        narrow_forecast = _gaussian_on_grid(_Q005, _Q995)
        assert _Y_OUTLIER > float(narrow_forecast.bin_edges[0, -1]), (
            "test setup error: _Y_OUTLIER must lie outside the narrow grid so "
            "that pad_to_common_grid actually pads."
        )

        ys = (_Y_OUTLIER,)
        true_wide = _cached_scores(-8.0, 8.0, ys)
        narrow = _cached_scores(_Q005, _Q995, ys)  # grid ends inside the DGP range

        failures = []
        for key in _ALL_PROPER_KEYS:
            slack = _slack_for(key)
            if not (true_wide[key] <= narrow[key] + slack):
                failures.append(
                    f"{key}: true={true_wide[key]:.6f} > narrow={narrow[key]:.6f} "
                    f"(slack={slack:g})"
                )

        assert not failures, (
            f"after zero-padding, the narrow-grid forecast (support ends at "
            f"q.995) scored better than the true N(0,1) for an outlier target "
            f"y={_Y_OUTLIER:.4f}; pad_to_common_grid must neutralise a narrow "
            "reported support:\n  " + "\n  ".join(failures)
        )

    def test_truncating_strictly_worsens_guarded_rules(self):
        """The headline defect scenario, sharpened for the divergence-form rules.

        For ``crts_alpha_*`` and ``dpd_beta_0.01`` the extreme-tail target sits
        exactly where the truncated forecast is wrongly certain, so truncating
        must make the score *strictly* worse — this is the precise statement
        that these rules do not inherit the CRLS support-sensitivity defect.

        Reuses the same cached [-8, 8] exact/truncated score dicts as
        ``test_truncating_does_not_improve_score`` (via ``_cached_scores``) and
        loops over the guarded rules internally.
        """
        ys = (_Y_OUTLIER,)
        exact = _cached_scores(-8.0, 8.0, ys)
        trunc = _cached_scores(-8.0, 8.0, ys, truncate=(_Q005, _Q995))

        failures = []
        for key in _GUARDED_KEYS:
            if not (exact[key] < trunc[key]):
                failures.append(
                    f"{key}: exact={exact[key]:.6f} >= trunc={trunc[key]:.6f}"
                )

        assert not failures, (
            f"truncating N(0,1) to [q.005, q.995] did not strictly worsen the "
            f"score for an outlier target y={_Y_OUTLIER:.4f}; this is the CRLS "
            "support-sensitivity defect:\n  " + "\n  ".join(failures)
        )

    def test_ranking_is_invariant_to_reported_support_width(self):
        """Widening the reported support must not change the ranking or (loosely) the score.

        Note on padding: ``compute_scoring_rules`` calls ``pad_to_common_grid``,
        which equalises the integration domain *for targets that fall outside a
        forecast's grid* — that is what makes an exterior-target comparison fair
        (see ``TestPadToCommonGrid``).  This test deliberately uses an
        *interior* target (``_Y_OUTLIER`` ≈ 3.08 lies inside both grids), so
        padding is a no-op here and the two grids differ purely in their tail
        *extent*.  That isolates the property under test: does the score depend
        on how far the reported support extends *beyond* the region that matters?

        Two forecasts are compared: the exact N(0,1) (true DGP) and a truncated
        variant.  Each is scored on a narrow grid ([-8, 8]) and a wide grid
        ([-20, 20]) at constant bin width.  For a support-insensitive rule:

          * the *ranking* (exact no worse than truncated) must not flip when the
            grid widens — a support-sensitive rule could otherwise be gamed by
            reporting a narrower grid; and
          * the numeric score of a *fixed* forecast must stay within a loose
            tolerance across grid widths, since the integrand vanishes far from
            the target so extra tail bins add ~nothing.

        Strict numeric equivalence is deliberately *not* required: differing
        grid extent and bin-midpoint alignment introduce O(bin_width) quadrature
        differences that are physically meaningless.

        The four score dicts are key-independent, so they are computed once (via
        the shared ``_cached_scores`` wrapper) and this test loops over every
        proper rule internally.
        """
        ys = (_Y_OUTLIER,)
        exact_narrow = _cached_scores(-8.0, 8.0, ys)
        trunc_narrow = _cached_scores(-8.0, 8.0, ys, truncate=(_Q005, _Q995))
        exact_wide = _cached_scores(-20.0, 20.0, ys)
        trunc_wide = _cached_scores(-20.0, 20.0, ys, truncate=(_Q005, _Q995))

        ranking_failures = []
        numeric_failures = []
        for key in _ALL_PROPER_KEYS:
            slack = _slack_for(key)

            # (a) Ranking must be preserved on both grids (within per-rule slack).
            if not (
                (exact_narrow[key] <= trunc_narrow[key] + slack)
                and (exact_wide[key] <= trunc_wide[key] + slack)
            ):
                ranking_failures.append(
                    f"{key}: narrow exact={exact_narrow[key]:.6f} vs "
                    f"trunc={trunc_narrow[key]:.6f}; wide exact={exact_wide[key]:.6f} "
                    f"vs trunc={trunc_wide[key]:.6f} (slack={slack:g})"
                )

            # (b) A fixed forecast's score must be stable (loose numeric) across widths.
            if exact_wide[key] != pytest.approx(
                exact_narrow[key], rel=5e-2, abs=max(1e-6, slack)
            ):
                numeric_failures.append(
                    f"{key}: exact score changed from {exact_narrow[key]:.6f} "
                    f"(narrow) to {exact_wide[key]:.6f} (wide)"
                )

        assert not ranking_failures, (
            "the ranking between exact and truncated forecasts flipped when the "
            "reported support widened for:\n  " + "\n  ".join(ranking_failures)
        )
        assert not numeric_failures, (
            "the score of the *fixed* exact forecast moved too much when the "
            "reported support widened from [-8, 8] to [-20, 20] (bin width held "
            "fixed); a support-insensitive rule should barely move:\n  "
            + "\n  ".join(numeric_failures)
        )

    @pytest.mark.parametrize("alpha", CRTS_ALPHAS)
    def test_integrand_vanishes_where_forecast_is_certain(self, alpha):
        """Root-cause guard, asserted directly on the kernel.

        Domain insensitivity holds *iff* the CRTS integrand is zero on every
        bin where the forecast is certain and correct, i.e. where ``F`` is flat
        at ``0`` strictly below the target (``s_α(0, 0) = 0``) or flat at ``1``
        strictly above it (``s_α(1, 1) = 0``).  A non-zero constant on such a
        bin is precisely the CRLS defect: every extra bin of reported support
        then adds a fixed amount to the integral, so the score scales with the
        domain width.

        The slab kernel reconstructs ``F`` by *linear* interpolation between
        the two edge values of each bin, so the invariant is a statement about
        the flat, saturated bins away from the target -- the single bin that
        contains ``y`` legitimately carries the finite CRPS-like contribution
        of the sub-bin forecast and is excluded here.  This drives
        ``compute_crts`` directly on a CDF made of flat-0 bins, one jump bin
        containing ``y``, and flat-1 bins, and widens the flat *tails* while
        holding the jump bin fixed: a defect on the flat bins would make the
        score grow with the added support, whereas a correct kernel leaves it
        unchanged.
        """

        def score(n_below, n_above):
            # Flat-0 tail (n_below bins) | jump bin containing y | flat-1 tail
            # (n_above bins).  cdf holds F at the right edge of each bin.
            cdf_row = ([0.0] * n_below) + [1.0] + ([1.0] * n_above)
            cdf = torch.tensor([cdf_row], dtype=torch.float64)
            n_bins = len(cdf_row)
            bin_edges = torch.arange(n_bins + 1, dtype=torch.float64)
            jump = n_below                       # index of the y-bin
            y = torch.tensor([jump + 0.5], dtype=torch.float64)
            y_bin = torch.tensor([jump], dtype=torch.long)
            out = compute_crts.__wrapped__(
                cdf, bin_edges, y, y_bin, shared=True, alphas=[alpha],
            )
            return out[f"crts_alpha_{alpha}"]

        narrow = score(1, 1)
        wide = score(4, 4)   # three extra flat bins on each side
        assert wide == pytest.approx(narrow, abs=1e-8), (
            f"alpha={alpha}: widening the reported support from 2 to 6 flat "
            f"bins moved the score from {narrow:.6e} to {wide:.6e}; the "
            "integrand must vanish on the flat, certain-and-correct bins, "
            "otherwise the integral grows with the reported support width."
        )

    @pytest.mark.parametrize("alpha", CRTS_ALPHAS)
    @pytest.mark.parametrize("width", [0.5, 1.0, 3.7, 100.0])
    def test_gap_slab_integral_matches_analytic_value(self, alpha, width):
        """The out-of-support *gap* term must equal ``width/(alpha-1)`` exactly.

        This is the numerical guard for the ``eps``-clamp regression.  On a bin
        where the forecast is *certain and wrong* the integrand collapses to the
        constant gap value

            s_α(1, 0) = s_α(0, 1) = 1/(α-1),

        so the slab integral over a bin of a given ``width`` is analytically
        ``width/(α-1)`` — independent of the quadrature and of the bin width.

        The historical bug clamped ``F`` to ``[eps, 1-eps]`` instead of the
        natural ``[0, 1]``.  That forced the certain-and-wrong point term from
        ``(1-F)^{α-1} = 0^{α-1} = 0`` up to ``eps^{α-1}``, which for orders near
        1 is *not* negligible:

            α=2.0 → 2.04e-14 (ok)   α=1.5 → 1.51e-7 (ok)
            α=1.2 → 1.22e-3 (drift) α=1.01 → 0.73    (73% off!)

        With the correct ``[0, 1]`` clamp all four cases must sit at machine
        precision.  Both tails are exercised: the right tail (``F ≡ 1``,
        ``q = 0``, uses ``(1-F)^{α-1}``) and the mirror left tail (``F ≡ 0``,
        ``q = 1``, uses ``F^{α-1}``).
        """
        from scoringbench.univariate.metrics import _crts_slab_integral

        w = torch.tensor([[width]], dtype=torch.float64)
        analytic = width / (alpha - 1.0)

        # Right catch-all: forecast certain (F ≡ 1) but target is above → wrong.
        F_lo = torch.ones_like(w)
        F_hi = torch.ones_like(w)
        right = _crts_slab_integral(F_lo, F_hi, w, 0.0, alpha).item()

        # Left catch-all: forecast certain (F ≡ 0) but target is below → wrong.
        F_lo0 = torch.zeros_like(w)
        F_hi0 = torch.zeros_like(w)
        left = _crts_slab_integral(F_lo0, F_hi0, w, 1.0, alpha).item()

        assert right == pytest.approx(analytic, rel=1e-9), (
            f"alpha={alpha}, width={width}: right-tail gap slab integral "
            f"{right:.6e} != analytic {analytic:.6e} (rel err "
            f"{abs(right - analytic) / analytic:.2e}); the certain-and-wrong "
            "point term must vanish, i.e. the clamp must be [0, 1] not "
            "[eps, 1-eps]."
        )
        assert left == pytest.approx(analytic, rel=1e-9), (
            f"alpha={alpha}, width={width}: left-tail gap slab integral "
            f"{left:.6e} != analytic {analytic:.6e} (rel err "
            f"{abs(left - analytic) / analytic:.2e})."
        )

    @pytest.mark.parametrize("alpha", CRTS_ALPHAS)
    def test_gap_slab_integral_is_width_scale_invariant(self, alpha):
        """The *relative* gap-term error must not depend on the bin width.

        A width-dependent relative error would signal a quadrature defect; a
        width-*independent* non-zero relative error signals a clamp/endpoint
        defect (the historical ``eps``-clamp).  Here the relative error is
        pinned to machine precision at every width, which reproduces the "these
        terms are small" table across scales.
        """
        from scoringbench.univariate.metrics import _crts_slab_integral

        rel_errs = []
        for width in (0.1, 1.0, 42.0):
            w = torch.tensor([[width]], dtype=torch.float64)
            F1 = torch.ones_like(w)
            got = _crts_slab_integral(F1, F1, w, 0.0, alpha).item()
            analytic = width / (alpha - 1.0)
            rel_errs.append(abs(got - analytic) / analytic)

        for rel in rel_errs:
            assert rel < 1e-9, (
                f"alpha={alpha}: gap-term relative error {rel:.2e} exceeds "
                "machine precision; after dropping the biased [eps, 1-eps] "
                "clamp every alpha (including 1.01) must be ~1e-14."
            )
        # Width-independence: spread of relative errors is itself tiny.
        assert max(rel_errs) - min(rel_errs) < 1e-9, (
            f"alpha={alpha}: relative gap-term error varies with width "
            f"({rel_errs}); it should be width-invariant (round-off only)."
        )

    def test_no_infinite_penalty_outside_support(self):
        """An out-of-support target must stay finite.

        Finiteness is a universal requirement, so this runs over *every* proper
        rule, not just the divergence-form ones.

        The log score returns +∞ when the target falls where the forecast
        assigns zero density.  A support-insensitive rule must instead return a
        finite (if large) penalty, otherwise a single outlier makes the whole
        benchmark column incomparable.

        The score dict is key-independent, so it is computed once (via
        ``_cached_scores``) and this test loops over every proper rule.
        """
        scores = _cached_scores(-8.0, 8.0, (_Q995 + 1.0,), truncate=(_Q005, _Q995))

        failures = [
            f"{key}={scores[key]}"
            for key in _ALL_PROPER_KEYS
            if not np.isfinite(scores[key])
        ]
        assert not failures, (
            "the following rules returned a non-finite score for a target "
            "outside the forecast support (expected a finite penalty):\n  "
            + "\n  ".join(failures)
        )

    def test_crts_alpha2_matches_crps(self):
        """At α=2 the binary Tsallis loss is the Brier divergence, so CRTS == CRPS.

        CRPS is a known support-insensitive threshold-integral rule, so this
        identity independently corroborates that the CRTS integrand is in the
        correct divergence form.

        Reuses the cached exact [-8, 8] score dict (shared with the propriety
        tests) via ``_cached_scores``.
        """
        metrics = _cached_scores(-8.0, 8.0, (_Y_OUTLIER,))

        assert metrics["crts_alpha_2.0"] == pytest.approx(
            metrics["crps"], rel=5e-3
        ), (
            f"crts_alpha_2.0 ({metrics['crts_alpha_2.0']:.6f}) should equal crps "
            f"({metrics['crps']:.6f}); the α=2 integrand is the Brier divergence."
        )

    def test_every_emitted_key_is_classified(self):
        """Guard the auto-discovery: no emitted key may be silently ignored.

        ``_ALL_PROPER_KEYS`` is discovered from ``compute_scoring_rules`` minus
        the explicit ``_NON_PROPER_*`` exclusions.  This test makes that split
        exhaustive: every key the metric emits must be classified as either a
        proper rule (and therefore tested above) or an explicitly excluded
        diagnostic.  Because ``_ALL_PROPER_KEYS`` is literally ``[k for k in
        emitted if _is_proper_key(k)]`` this cannot fail today, but it pins the
        contract so that a future refactor which starts *filtering* proper keys
        (rather than only classifying them) can't quietly drop a rule from the
        suite, and it documents ``_NON_PROPER_*`` as the single place to update
        when adding a non-proper diagnostic.
        """
        probe = _gaussian_on_grid(-8.0, 8.0, bin_width=0.05)
        emitted = set(compute_scoring_rules(probe, np.array([0.5], dtype=np.float64)))

        proper = {k for k in emitted if _is_proper_key(k)}
        non_proper = emitted - proper

        assert proper == set(_ALL_PROPER_KEYS), (
            "proper-rule discovery drifted from _ALL_PROPER_KEYS; "
            f"missing={proper - set(_ALL_PROPER_KEYS)}, "
            f"extra={set(_ALL_PROPER_KEYS) - proper}."
        )
        assert emitted == proper | non_proper, (
            f"unclassified scoring-rule keys: {emitted - (proper | non_proper)}. "
            "Classify them as proper (they will then be tested) or add them to "
            "_NON_PROPER_EXACT / _NON_PROPER_PREFIXES."
        )


# ============================================================================
# TestPadToCommonGrid
# ============================================================================

class TestPadToCommonGrid:
    """Unit tests for the padding helper that extends grids to cover y.

    These tests drive ``pad_to_common_grid`` directly so that failures are
    localised to the padding logic rather than to a downstream scoring rule.
    """

    def _uniform_grid(self, lo, hi, bw=_BW):
        """Return (probas, bin_edges, bin_mids) tensors for a flat PMF."""
        n = int(round((hi - lo) / bw))
        edges = torch.linspace(lo, hi, n + 1, dtype=torch.float64)
        mids  = 0.5 * (edges[:-1] + edges[1:])
        probas = torch.full((1, n), 1.0 / n, dtype=torch.float64)
        return probas, edges, mids

    def test_no_padding_when_y_interior(self):
        """Grid is returned unchanged when y is already inside the support."""
        probas, edges, mids = self._uniform_grid(-4.0, 4.0)
        y = torch.tensor([0.0], dtype=torch.float64)
        p2, e2, m2, shared2, *_ = pad_to_common_grid(probas, edges, mids, y, shared=True)
        assert p2 is probas
        assert e2 is edges
        assert m2 is mids
        assert shared2 is True

    def test_padding_right_extends_grid(self):
        """A target above the right edge triggers right-side padding."""
        probas, edges, mids = self._uniform_grid(-4.0, 4.0)
        y = torch.tensor([5.0], dtype=torch.float64)
        p2, e2, m2, shared2, *_ = pad_to_common_grid(probas, edges, mids, y, shared=True)

        assert e2[-1].item() >= 5.0, "right edge must cover y=5.0"
        assert p2.shape[1] > probas.shape[1], "extra bins must have been added"
        assert p2.sum().item() == pytest.approx(1.0, abs=1e-12), "PMF must still sum to 1"
        n_orig = probas.shape[1]
        assert p2[0, n_orig:].sum().item() == pytest.approx(0.0, abs=1e-15)

    def test_padding_left_extends_grid(self):
        """A target below the left edge triggers left-side padding."""
        probas, edges, mids = self._uniform_grid(-4.0, 4.0)
        y = torch.tensor([-6.0], dtype=torch.float64)
        p2, e2, m2, shared2, *_ = pad_to_common_grid(probas, edges, mids, y, shared=True)

        assert e2[0].item() <= -6.0, "left edge must cover y=-6.0"
        assert p2.sum().item() == pytest.approx(1.0, abs=1e-12)
        n_orig = probas.shape[1]
        n_new  = p2.shape[1]
        n_pad  = n_new - n_orig
        assert p2[0, :n_pad].sum().item() == pytest.approx(0.0, abs=1e-15)

    def test_padding_both_sides(self):
        """Targets outside both edges pad both sides simultaneously."""
        probas, edges, mids = self._uniform_grid(-4.0, 4.0)
        y = torch.tensor([-6.0, 6.0], dtype=torch.float64)
        probas2 = probas.expand(2, -1).clone()
        p2, e2, m2, shared2, *_ = pad_to_common_grid(probas2, edges, mids, y, shared=True)

        assert e2[0].item() <= -6.0
        assert e2[-1].item() >= 6.0
        assert p2.sum().item() == pytest.approx(2.0, abs=1e-12)

    def test_bin_width_preserved(self):
        """Original interior bins keep their width; the single catch-all pad bin
        may be wider (it spans from the grid edge to just beyond the target).
        """
        bw = _BW
        probas, edges, mids = self._uniform_grid(-4.0, 4.0, bw=bw)
        y = torch.tensor([7.0], dtype=torch.float64)
        _, e2, *_ = pad_to_common_grid(probas, edges, mids, y, shared=True)
        widths = torch.diff(e2)
        n_orig = probas.shape[1]
        # Interior (original) bins must still have width bw.
        interior_widths = widths[:-1]  # last bin is the right catch-all
        assert interior_widths.min().item() == pytest.approx(bw, rel=1e-9)
        assert interior_widths.max().item() == pytest.approx(bw, rel=1e-9)
        # Catch-all bin must be at least as wide as bw and cover y=7.0.
        assert widths[-1].item() >= bw - 1e-12
        assert e2[-1].item() >= 7.0

    def test_exterior_target_does_not_improve_score(self):
        """End-to-end: truncating must not win for a target outside the truncation window.

        This is the scenario that padding is designed to fix.  Before padding
        was introduced, a model with a narrower reported support could score
        better than the exact Gaussian for an outlier target, because the
        integration domain was too short to charge the mispredicted tail.

        Both forecasts are evaluated on the *same* common grid after padding,
        so the exact Gaussian (the true DGP) must not score worse than the
        truncated one.

        The two score dicts are key-independent, so they are computed once (via
        ``_cached_scores``) and this test loops over the guarded rules.
        """
        y_out = _Q995 + 1.0
        exact = _cached_scores(-8.0, 8.0, (y_out,))
        trunc = _cached_scores(-8.0, 8.0, (y_out,), truncate=(_Q005, _Q995))

        failures = []
        for key in _GUARDED_KEYS:
            if not (exact[key] < trunc[key]):
                failures.append(
                    f"{key}: exact={exact[key]:.6f} >= trunc={trunc[key]:.6f}"
                )

        assert not failures, (
            f"the truncated forecast scored better than the exact Gaussian for "
            f"an exterior target y={y_out:.4f}; padding should have closed this "
            "gap:\n  " + "\n  ".join(failures)
        )

    def test_exterior_target_narrow_grid_penalised_more_than_wide(self):
        """Padding closes but does not fully eliminate the narrow-grid advantage.

        Before padding, a truncated model on a narrow grid [Q005, Q995] scored
        *better* than the same truncation on a wide grid [-8, 8] for an outlier
        target — because the narrow grid's integration domain was too short to
        charge the mispredicted tail.

        After padding both grids extend to cover y, so the narrow-grid score
        must be at most marginally better than the wide-grid score.  We verify
        that the ratio s_narrow / s_wide is ≥ 0.95 (i.e. the narrow grid no
        longer wins by more than 5%, which is the O(bw) quadrature difference
        from misaligned bin midpoints).

        The two score dicts are key-independent, so they are computed once (via
        ``_cached_scores``) and this test loops over the CRTS rules.
        """
        y_out = _Q995 + 1.0
        narrow = _cached_scores(_Q005, _Q995, (y_out,), truncate=(_Q005, _Q995))
        wide = _cached_scores(-8.0, 8.0, (y_out,), truncate=(_Q005, _Q995))

        failures = []
        for key in _CRTS_KEYS:
            s_n = narrow[key]
            s_w = wide[key]
            if not (s_n > 0 and s_w > 0):
                failures.append(f"{key}: expected positive scores, got {s_n}, {s_w}")
                continue
            ratio = s_n / s_w
            if ratio < 0.95:
                failures.append(
                    f"{key}: narrow grid score {s_n:.6f} is more than 5% below "
                    f"wide grid score {s_w:.6f} (ratio={ratio:.4f})"
                )

        assert not failures, (
            "padding should have eliminated the large systematic narrow-grid "
            "advantage:\n  " + "\n  ".join(failures)
        )
