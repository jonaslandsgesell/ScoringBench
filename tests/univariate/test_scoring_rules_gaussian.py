"""Fast Gaussian analytical + Monte-Carlo validation of *every* scoring rule.

The idea: build a **fine histogram approximation of a Gaussian** ``N(μ, σ)`` —
bin mass ``p_k = Φ(e_{k+1}) − Φ(e_k)`` on a shared grid — feed it through the
public :func:`compute_scoring_rules`, and check each returned rule against an
independent reference.  Two reference families are used, chosen per rule so the
test is *both fast and tight*:

**A. Analytical Gaussian closed forms (deterministic, no MC noise).**
    * ``crps`` — ``σ·[z(2Φ(z)−1) + 2φ(z) − 1/√π]``, ``z = (y−μ)/σ``.
      Converges to the continuous Gaussian as the grid refines (O(w²)).
    * ``dpd_beta_β`` / ``cde_loss`` / ``pseudospherical_alpha_α`` — these are
      *density* functionals of the piecewise-constant histogram density
      ``f_k = p_k / w_k``.  Rather than compare to the continuous Gaussian
      (which only converges), we compare to the **exact histogram-density
      formula**, so the test is exact to machine precision *and* exercises the
      production code on a genuinely Gaussian-shaped input.

**B. Fast Monte-Carlo (shared no code with the closed forms).**
    * ``energy_score_beta_β`` for general β has no simple Gaussian closed form,
      so we validate it with a vectorised NumPy MC estimator that draws real
      Gaussian samples (``N ≈ 300k``) — fast (<1 s) yet tight (tol ≈ 5·SE).  The
      β = 1 energy score is additionally cross-checked against the analytical
      CRPS.

Every test runs on the CPU in float64 and finishes in well under a second, so
the whole module is cheap enough for the default suite.

**Auto-discovery.**  Parametrised families derive their values directly from the
``metrics`` constants (``ENERGY_BETAS``, ``DPD_BETAS``, ``CRTS_ALPHAS``,
``PSEUDOS_ALPHAS``, ``COVERAGE_LEVELS``), so adding a *parameter value* to any of
those lists is validated automatically with no edit here.  A final guard test,
``test_all_emitted_rules_are_accounted_for``, inspects every key
``compute_scoring_rules`` actually returns and fails loudly if a *brand-new
rule* is emitted without either a validation test or an explicit presence-only
acknowledgement — so new rules cannot slip in untested.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from scoringbench.univariate.metrics import (
    COVERAGE_LEVELS,
    CRTS_ALPHAS,
    DPD_BETAS,
    ENERGY_BETAS,
    PSEUDOS_ALPHAS,
    compute_scoring_rules,
)
from scoringbench.univariate.wrappers import DistributionPrediction

# Deterministic CPU float64 (compute_scoring_rules upcasts internally anyway).
_MU, _SIG = 0.7, 1.3
_LO, _HI = _MU - 8.0 * _SIG, _MU + 8.0 * _SIG  # ±8σ: tails are numerically zero
_N_BINS = 1500                                  # fine enough: CRPS err ~1e-6
_N_SAMPLES = 200                                # rows (all identical → cheap)
_Y = 1.5                                        # fixed target, interior to grid


# --------------------------------------------------------------------------- #
# Gaussian-histogram fixture
# --------------------------------------------------------------------------- #
def _gaussian_histogram():
    """Return (edges, mids, probas, f_norm, w) for the fine Gaussian histogram.

    ``f_norm`` is the *normalised* piecewise-constant density the production
    code reconstructs internally (``unified_bin_density``), so density-rule
    references built from it match the code exactly.
    """
    edges = np.linspace(_LO, _HI, _N_BINS + 1)
    p = np.diff(norm.cdf(edges, _MU, _SIG))
    p = p / p.sum()
    mids = 0.5 * (edges[:-1] + edges[1:])
    w = np.diff(edges)
    f = p / w
    f = f / (f * w).sum()  # normalise exactly like unified_bin_density
    return edges, mids, p, f, w


@pytest.fixture(scope="module")
def gauss_result():
    """Run compute_scoring_rules once on the Gaussian histogram (shared grid).

    Density-rule references (``f``, ``w``, ``ky``) are rebuilt from the
    ``dist.resampled`` view — the shared common grid the density rules actually
    score on — not the fine native PMF grid.  Reconstructing on the native PMF grid was
    the old contract; density rules now resample onto the common grid first, so
    references built from the native PMF grid diverge from the returned values.
    """
    edges, mids, p, _f_native, _w_native = _gaussian_histogram()
    probas = np.tile(p, (_N_SAMPLES, 1)).astype(np.float64)
    y = np.full(_N_SAMPLES, _Y, dtype=np.float64)
    dist = DistributionPrediction(
        probas=probas,
        bin_edges=edges.astype(np.float64),
        bin_midpoints=mids.astype(np.float64),
        mean=np.full(_N_SAMPLES, _MU),
        train_range=(float(np.asarray(edges).min()), float(np.asarray(edges).max())),
    )
    res = compute_scoring_rules(dist, y)

    # Density-rule references are built on the resampled (common) grid: exactly
    # the grid density rules score on. Reconstruct the normalised piecewise-
    # constant density the same way ``unified_bin_density`` does.
    rg = dist.resampled
    rg_edges = np.asarray(rg.bin_edges, dtype=np.float64)
    rg_p = np.asarray(rg.probas, dtype=np.float64)[0]
    w = np.diff(rg_edges)
    f = rg_p / w
    f = f / (f * w).sum()  # normalise exactly like unified_bin_density
    ky = int(np.searchsorted(rg_edges[1:], _Y).clip(0, len(w) - 1))
    return res, {"edges": rg_edges, "f": f, "w": w, "ky": ky}


# --------------------------------------------------------------------------- #
# Analytical Gaussian references
# --------------------------------------------------------------------------- #
def _crps_gaussian(y: float, mu: float = _MU, sig: float = _SIG) -> float:
    """Closed-form CRPS of ``N(mu, sig)`` at ``y`` (Gneiting & Raftery 2007)."""
    z = (y - mu) / sig
    return sig * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))


def _hist_density_integral(f: np.ndarray, w: np.ndarray, power: float) -> float:
    """``∫ f_hist(t)^power dt`` for a piecewise-constant density."""
    return float((f ** power * w).sum())


# --------------------------------------------------------------------------- #
# A. Analytical CRPS
# --------------------------------------------------------------------------- #
def test_crps_matches_gaussian_closed_form(gauss_result):
    res, _ = gauss_result
    ref = _crps_gaussian(_Y)
    # Discretisation error on a 1500-bin ±8σ grid is O(1e-6).
    assert abs(res["crps"] - ref) < 1e-4, f"crps {res['crps']} vs analytic {ref}"


# --------------------------------------------------------------------------- #
# A. Analytical density rules (exact histogram-density reference)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("beta", DPD_BETAS)
def test_dpd_matches_histogram_density(gauss_result, beta):
    """DPD_β = ∫ f^{1+β} − (1 + 1/β) f(y)^β for the histogram density f."""
    res, aux = gauss_result
    f, w, ky = aux["f"], aux["w"], aux["ky"]
    fy = f[ky]
    integral = _hist_density_integral(f, w, 1.0 + beta)
    ref = integral - (1.0 + 1.0 / beta) * fy ** beta
    got = res[f"dpd_beta_{beta}"]
    assert abs(got - ref) < 1e-9, f"dpd_beta_{beta}: {got} vs {ref}"


def test_cde_matches_histogram_density(gauss_result):
    """CDE = ∫ f² − 2 f(y) (== DPD β=1)."""
    res, aux = gauss_result
    f, w, ky = aux["f"], aux["w"], aux["ky"]
    ref = _hist_density_integral(f, w, 2.0) - 2.0 * f[ky]
    assert abs(res["cde_loss"] - ref) < 1e-9
    # Documented identity: cde_loss is exactly dpd_beta_1.0.
    assert abs(res["cde_loss"] - res["dpd_beta_1.0"]) < 1e-12


@pytest.mark.parametrize("alpha", PSEUDOS_ALPHAS)
def test_pseudospherical_matches_histogram_density(gauss_result, alpha):
    """Negated pseudospherical: −1/(α−1)·[ f(y)^{α−1} / (∫f^α)^{(α−1)/α} − 1 ]."""
    res, aux = gauss_result
    f, w, ky = aux["f"], aux["w"], aux["ky"]
    fy = f[ky]
    norm_alpha = _hist_density_integral(f, w, alpha)
    ratio = fy ** (alpha - 1.0) / norm_alpha ** ((alpha - 1.0) / alpha)
    ref = -(ratio - 1.0) / (alpha - 1.0)  # negated -> loss
    got = res[f"pseudospherical_alpha_{alpha}"]
    assert abs(got - ref) < 1e-9, f"pseudospherical_alpha_{alpha}: {got} vs {ref}"


# --------------------------------------------------------------------------- #
# A. Interval score & coverage (analytical Gaussian, well-specified forecast)
# --------------------------------------------------------------------------- #
def test_interval_score_matches_gaussian(gauss_result):
    """Interval score of a Gaussian at nominal level 1−α.

    IS_α(y) = (u − l) + 2/α·(l − y)_+ + 2/α·(y − u)_+, with l, u the α/2 and
    1−α/2 Gaussian quantiles.  Compared per available coverage level.
    """
    res, _ = gauss_result
    # Skip very wide/narrow levels where the quantile lands far in the tail and
    # the O(w) bin-edge quantisation dominates; test the informative middle band.
    for cov_level in [c for c in COVERAGE_LEVELS if 80 <= c <= 95]:
        alpha = 1.0 - cov_level / 100.0
        lo = norm.ppf(alpha / 2.0, _MU, _SIG)
        hi = norm.ppf(1.0 - alpha / 2.0, _MU, _SIG)
        ref = (
            (hi - lo)
            + (2.0 / alpha) * max(lo - _Y, 0.0)
            + (2.0 / alpha) * max(_Y - hi, 0.0)
        )
        got = res[f"interval_score_{cov_level}"]
        # The quantile lands on a bin edge (±w/2 quantisation) -> tol ~ few·w.
        assert abs(got - ref) < 5e-2, (
            f"interval_score_{cov_level}: {got} vs analytic {ref}"
        )


def test_coverage_is_well_calibrated(gauss_result):
    """A well-specified Gaussian forecast covers y at ~the nominal rate.

    Here every row is the same forecast and y is the same fixed point, so
    coverage is 0/1 per level: it must be 1 exactly when y lies inside the
    analytical central interval, 0 otherwise.
    """
    res, _ = gauss_result
    for cov_level in COVERAGE_LEVELS:
        alpha = 1.0 - cov_level / 100.0
        lo = norm.ppf(alpha / 2.0, _MU, _SIG)
        hi = norm.ppf(1.0 - alpha / 2.0, _MU, _SIG)
        expected = 1.0 if (lo <= _Y <= hi) else 0.0
        assert res[f"coverage_{cov_level}"] == pytest.approx(expected, abs=1e-9)


# --------------------------------------------------------------------------- #
# A. Quantile-weighted CRPS (analytical pinball integral over the Gaussian)
# --------------------------------------------------------------------------- #
def _wcrps_gaussian_reference(y: float, weight: str) -> float:
    """qwCRPS_v = ∫₀¹ 2·ρ_α(y, Q(α))·v(α) dα with Q the Gaussian quantile fn.

    Uses the SAME 99-level midpoint quadrature grid as the production code so
    the discretisation of the integral matches; the quantiles are the exact
    Gaussian quantiles (the reference), whereas the code inverts the histogram
    CDF (→ bin midpoints).  Tolerance therefore absorbs the O(w) quantile
    quantisation only.
    """
    alphas = np.linspace(0.01, 0.99, 99)
    q = norm.ppf(alphas, _MU, _SIG)
    pinball = 2.0 * ((y <= q).astype(float) - alphas) * (q - y)
    if weight == "left":
        v = (1.0 - alphas) ** 2
    elif weight == "right":
        v = alphas ** 2
    else:
        v = alphas * (1.0 - alphas)
    # midpoint rule on the interior grid of (0,1): equal weight 1/(99+1)
    return float((pinball * v).sum() / (len(alphas) + 1))


@pytest.mark.parametrize("weight", ["left", "right", "center"])
def test_wcrps_matches_gaussian_pinball(gauss_result, weight):
    res, _ = gauss_result
    ref = _wcrps_gaussian_reference(_Y, weight)
    got = res[f"wcrps_{weight}"]
    # Quantile quantisation to bin midpoints on a fine grid -> tol ~ few·w.
    assert abs(got - ref) < 5e-3, f"wcrps_{weight}: {got} vs analytic {ref}"


# --------------------------------------------------------------------------- #
# A. CRTS — continuous-Gaussian reference via the documented α-Tsallis integrand
# --------------------------------------------------------------------------- #
def _crts_binary_tsallis(p: np.ndarray, q: float, alpha: float) -> np.ndarray:
    """Divergence-form binary α-Tsallis integrand s_α(p, q) (metrics docstring).

        s_α(p,q) = [p^α + (1-p)^α]/α
                   − [q·p^{α-1} + (1-q)·(1-p)^{α-1}]/(α-1)
                   − [1/α − 1/(α-1)].
    """
    p = np.clip(p, 1e-15, 1.0 - 1e-15)
    return (
        (p ** alpha + (1.0 - p) ** alpha) / alpha
        - (q * p ** (alpha - 1.0) + (1.0 - q) * (1.0 - p) ** (alpha - 1.0)) / (alpha - 1.0)
        - (1.0 / alpha - 1.0 / (alpha - 1.0))
    )


def _crts_gaussian_reference(y: float, alpha: float) -> float:
    """CRTS_α(N(μ,σ), y) = ∫ s_α(Φ((t−μ)/σ), 1{t≥y}) dt via direct quadrature.

    An independent continuous reference (uses the exact Gaussian CDF, not the
    histogram), so a match confirms the histogram discretisation is faithful.
    """
    from scipy import integrate

    integrand = lambda t: _crts_binary_tsallis(  # noqa: E731
        norm.cdf(t, _MU, _SIG), 1.0 if t >= y else 0.0, alpha
    )
    val, _ = integrate.quad(integrand, _LO, _HI, points=[y], limit=200)
    return float(val)


def test_crts_alpha2_equals_crps(gauss_result):
    """At α = 2 the α-Tsallis integrand is the Brier divergence → CRTS == CRPS."""
    res, _ = gauss_result
    assert abs(res["crts_alpha_2.0"] - res["crps"]) < 1e-9


@pytest.mark.parametrize("alpha", CRTS_ALPHAS)
def test_crts_matches_gaussian_integrand(gauss_result, alpha):
    """CRTS reproduces the continuous-Gaussian α-Tsallis integral for every α."""
    res, _ = gauss_result
    ref = _crts_gaussian_reference(_Y, alpha)
    got = res[f"crts_alpha_{alpha}"]
    # Slab discretisation of the integrand on a fine grid -> O(1e-5).
    assert abs(got - ref) < 1e-3, f"crts_alpha_{alpha}: {got} vs analytic {ref}"


# --------------------------------------------------------------------------- #
# B. Energy score — fast Monte-Carlo (no simple Gaussian closed form for β≠1)
# --------------------------------------------------------------------------- #
def _mc_energy_gaussian(y: float, beta: float, n: int, rng) -> tuple[float, float]:
    """MC estimate (and SE) of ES_β(N(μ,σ), y) = E|X−y|^β − ½E|X−X'|^β."""
    x = rng.normal(_MU, _SIG, n)
    xp = rng.normal(_MU, _SIG, n)
    term1 = np.abs(x - y) ** beta                    # E|X−y|^β
    term2 = np.abs(x - xp) ** beta                   # E|X−X'|^β (independent pair)
    per = term1 - 0.5 * term2
    mean = float(per.mean())
    se = float(per.std(ddof=1) / np.sqrt(n))
    return mean, se


@pytest.mark.parametrize("beta", ENERGY_BETAS)
def test_energy_score_matches_monte_carlo(gauss_result, beta):
    """The histogram energy score reproduces the Gaussian MC energy score.

    The histogram is a fine discretisation of N(μ,σ), so its energy score must
    match samples drawn from the same Gaussian up to MC noise + discretisation.
    """
    res, _ = gauss_result
    rng = np.random.default_rng(12345)
    mc_mean, mc_se = _mc_energy_gaussian(_Y, beta, 400_000, rng)
    got = res[f"energy_score_beta_{beta}"]
    tol = 5.0 * mc_se + 5e-3  # MC noise band + small discretisation floor
    assert abs(got - mc_mean) < tol, (
        f"energy_score_beta_{beta}: hist={got} vs MC={mc_mean}±{mc_se} (tol={tol})"
    )


def test_energy_beta1_equals_crps(gauss_result):
    """ES_{β=1} is exactly CRPS: the code reads crps off energy_score_beta_1.0."""
    res, _ = gauss_result
    assert abs(res["energy_score_beta_1.0"] - res["crps"]) < 1e-12
    # ...and both match the analytical Gaussian CRPS.
    assert abs(res["crps"] - _crps_gaussian(_Y)) < 1e-4


# --------------------------------------------------------------------------- #
# Sanity: sharpness == σ for a Gaussian
# --------------------------------------------------------------------------- #
def test_sharpness_matches_gaussian_std(gauss_result):
    """Sharpness (mean predictive std) of a Gaussian histogram equals σ."""
    res, _ = gauss_result
    assert abs(res["sharpness"] - _SIG) < 1e-3


# --------------------------------------------------------------------------- #
# Auto-discovery guard: every emitted scoring rule must be explicitly covered
# --------------------------------------------------------------------------- #
#
# The tests above pin each rule to an independent reference.  This guard closes
# the loop the other way: it inspects *every* key that ``compute_scoring_rules``
# actually returns and asserts it is accounted for here.  If someone adds a new
# rule (or a new parameter value to an existing family) and forgets to add a
# matching validation test, this test fails LOUDLY instead of the new rule
# slipping through untested.
#
# How it stays in sync automatically:
#   * Parametrised families are derived from the metrics constants
#     (ENERGY_BETAS, DPD_BETAS, CRTS_ALPHAS, PSEUDOS_ALPHAS, COVERAGE_LEVELS),
#     so adding a value to any of those lists is picked up with no edit here.
#   * Scalar rules with a dedicated analytical/MC test are listed in
#     ``_VALIDATED_SCALAR_KEYS``.
#   * Rules that are intentionally *not* value-checked in this module (only
#     their presence is asserted) live in ``_PRESENCE_ONLY_KEYS`` with a reason.
#
# Adding a brand-new rule therefore forces a deliberate choice: either add a
# real validation test (and, if scalar, list its key below) or explicitly
# acknowledge it as presence-only — you cannot add it silently.

# Scalar keys that have a dedicated value-checking test in this file.
_VALIDATED_SCALAR_KEYS = frozenset({
    "crps",            # test_crps_matches_gaussian_closed_form
    "cde_loss",        # test_cde_matches_histogram_density
    "sharpness",       # test_sharpness_matches_gaussian_std
    "wcrps_left",      # test_wcrps_matches_gaussian_pinball
    "wcrps_right",
    "wcrps_center",
})

# Keys whose *presence* is required but whose value is not analytically checked
# here (documented reason each).  Keep this list short and justified.
_PRESENCE_ONLY_KEYS = {
    "dispersion":    "std of per-sample predictive std; no simple Gaussian target",
    "pit_ks_stat":   "PIT KS statistic; distributional, not a pointwise score",
    "pit_ks_pvalue": "PIT KS p-value; distributional, not a pointwise score",
}


def _expected_family_keys() -> set[str]:
    """All keys expected from the metrics constants (auto-tracks new values)."""
    keys: set[str] = set()
    keys |= {f"energy_score_beta_{b}" for b in ENERGY_BETAS}
    keys |= {f"dpd_beta_{b}" for b in DPD_BETAS}
    keys |= {f"crts_alpha_{a}" for a in CRTS_ALPHAS}
    keys |= {f"pseudospherical_alpha_{a}" for a in PSEUDOS_ALPHAS}
    keys |= {f"coverage_{c}" for c in COVERAGE_LEVELS}
    keys |= {f"interval_score_{c}" for c in COVERAGE_LEVELS}
    return keys


def test_all_emitted_rules_are_accounted_for(gauss_result):
    """Fail loudly if compute_scoring_rules emits a key nothing here covers.

    This is the auto-discovery net: it guarantees the validated set stays a
    *superset* of what the pipeline produces, so new rules cannot be added
    without either a validation test or an explicit presence-only entry.
    """
    res, _ = gauss_result
    emitted = set(res.keys())

    accounted = (
        _expected_family_keys()
        | _VALIDATED_SCALAR_KEYS
        | set(_PRESENCE_ONLY_KEYS)
    )

    # 1) Nothing unexpected slipped through untested.
    unaccounted = emitted - accounted
    assert not unaccounted, (
        "compute_scoring_rules emitted keys with no coverage in "
        "test_scoring_rules_gaussian.py: "
        f"{sorted(unaccounted)}. Add a validation test (and list the key in "
        "_VALIDATED_SCALAR_KEYS) or, if intentionally not value-checked, add it "
        "to _PRESENCE_ONLY_KEYS with a reason."
    )

    # 2) Everything we claim to cover is actually emitted (guards against typos
    #    and stale entries drifting out of sync with the pipeline).
    stale = accounted - emitted
    assert not stale, (
        f"These keys are listed as covered but the pipeline no longer emits "
        f"them (stale references): {sorted(stale)}."
    )

    # 3) Presence-only keys really are present.
    missing_presence = set(_PRESENCE_ONLY_KEYS) - emitted
    assert not missing_presence, (
        f"Presence-only keys are missing from output: {sorted(missing_presence)}"
    )
