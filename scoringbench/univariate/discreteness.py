"""Target discreteness detection for density-based scoring rules.

Why this exists
---------------
The density-based scoring rules in :mod:`scoringbench.univariate.metrics`
(``DENSITY_BASED_KEYS``: ``cde_loss``, ``dpd_beta_*``, ``pseudospherical_alpha_*``)
read the predictive *density* ``f`` pointwise.  They are strictly proper only
with respect to a *continuous* target distribution.  When the target ``y`` is
lattice-valued (ordinal / integer / multiples of a fixed step) or carries point
masses (a value repeated with non-negligible probability), there is no
continuous density for these rules to be proper against: the histogram density
``f = p/w`` and the kernel-smoothed resampling that feeds them produce a finite
but statistically meaningless number.  For such datasets we SKIP the density
rules and keep only the grid-robust family (CRPS, energy, CRTS, interval,
wCRPS), which read the CDF ``F`` and remain proper against atoms.

Detector: lattice OR atom-mass
------------------------------
Computed ONCE on the full ``y`` (never per-fold), so the flag is invariant to
CV folds / repeats -- a dataset-level property, not a subsample statistic.
Three sufficient conditions, any of which flags the target discrete:

1. **Few unique values** -- ``n_unique <= MIN_UNIQUE``: an ordinal / small-
   support target (Likert scales, small counts) is discrete by inspection.

2. **Lattice (Fourier)** -- the empirical characteristic function
   ``phi_n(t) = (1/n) sum_j exp(i t y_j)`` has ``|phi_n| in [0, 1]`` and attains
   ``1`` at ``t = 2*pi/h`` *iff* every ``y_j`` lies on a lattice of span ``h``.
   We PROPOSE a candidate span by a tolerance-gcd of the gaps between sorted
   unique values, then VALIDATE it by checking ``|phi_n(2*pi/h)| >= 1 - ECF_TOL``.
   The gcd alone is brittle to float noise; the ECF magnitude is the robust
   test.  ``phi_n`` is evaluated on the FULL ``y`` (with repetitions), so tied
   values weight the magnitude by their multiplicity -- repetitions *reinforce*
   detection rather than fight it.  We additionally require substantial
   repetition (``frac_unique <= LATTICE_UNIQUE_RATIO``) so that genuinely
   continuous data that merely happens to be rounded to a fine grid (thousands
   of lattice sites, each hit ~once, negligible atom mass) is NOT flagged.

3. **Dominant atom** -- ``max_atom_mass >= ATOM_MASS_MAX``: a single value
   carrying a large share of the mass (e.g. a censored / capped target, a spike
   at zero) is a point mass that breaks density propriety even when the rest of
   the target is continuous and not on any global lattice.

Manual override
---------------
``DISCRETE_OVERRIDES`` maps a dataset ``name`` to a forced boolean, mirroring the
``SKIPPED_OPENML_IDS`` / ``dedup_keys`` patterns in :mod:`datasets`.  It wins over
the auto-detector for audited edge cases.  Empty by default.

All thresholds are module constants so they are tunable in one place.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Tunable thresholds (one place)
# ---------------------------------------------------------------------------
MIN_UNIQUE = 20            # n_unique <= this -> discrete (few-support / ordinal)
ECF_TOL = 1e-2             # lattice confirmed if ecf_peak >= 1 - ECF_TOL
LATTICE_UNIQUE_RATIO = 0.10  # lattice branch also needs frac_unique <= this
ATOM_MASS_MAX = 0.10       # a single value holding >= this share of mass -> discrete

# ---------------------------------------------------------------------------
# Manual overrides: dataset name -> forced is_discrete.  Wins over auto-detect.
# Add audited edge cases here, e.g.:
#   "california": False,   # keep the capped target in density rankings
#   "some_count_set": True,
#
# The auto-detector cannot separate an integer COUNT (no continuous density ->
# discrete) from an integer-valued PRICE on a fine grid (rounded continuous ->
# keep) by statistics alone -- both are unit lattices -- so a genuine count with
# many distinct values (frac_unique > LATTICE_UNIQUE_RATIO) slips through the
# lattice branch.  Such cases are pinned here after an audit of the target's
# meaning rather than by loosening the ratio (which would misflag fine-grid
# prices such as ``house_prices_nominal``).
# ---------------------------------------------------------------------------
DISCRETE_OVERRIDES: dict[str, bool] = {
    # RS = baseball runs scored, a pure integer count (unit lattice, ecf_peak=1);
    # No continuous density -> density / PIT rules are not proper here.
    "Moneyball": True,
    # Target is log(shares): a log-transformed integer count. The log breaks the
    # lattice (ecf_peak~0) and spreads mass over ~1454 atoms, each just under
    # ATOM_MASS_MAX, so neither the lattice nor the atom_mass branch fires
    "OnlineNewsPopularity": True,
    # Integer 0-255: an 8-bit quantized sensor reading (unit lattice, ecf_peak=1);
    "Laser": True,
}


def _lattice_span(u: np.ndarray, scale: float) -> float:
    """Candidate lattice span h via a tolerance-gcd of gaps between sorted unique values.

    The span of a lattice distribution divides every gap between occupied sites,
    so it is the gcd of those gaps.  Floating gcd is done with an absolute
    tolerance scaled to the data range; if the running gcd collapses below the
    tolerance the values are not on a coarse lattice and ``nan`` is returned.

    Parameters
    ----------
    u : np.ndarray
        Sorted unique target values (1-D).
    scale : float
        Data range ``u[-1] - u[0]``, used to set the absolute tolerance.

    Returns
    -------
    float
        Candidate span ``h > 0``, or ``np.nan`` if no coarse lattice is found.
    """
    if u.size < 2:
        return float("nan")
    gaps = np.diff(u)
    gaps = gaps[gaps > 0.0]
    if gaps.size == 0:
        return float("nan")
    atol = 1e-9 + 1e-6 * float(scale)
    h = float(gaps[0])
    for g in gaps[1:]:
        a, b = h, float(g)
        # Euclid with tolerance; ``a - floor(a/b)*b`` is the non-negative remainder.
        while b > atol:
            a, b = b, a - np.floor(a / b) * b
        h = a
        if h <= atol:
            return float("nan")
    return h


def _ecf_peak(y: np.ndarray, h: float) -> float:
    """``|phi_n(2*pi/h)|`` -- empirical characteristic-function magnitude at the candidate span.

    Evaluated on the FULL ``y`` (repetitions included), so the magnitude is the
    multiplicity-weighted lattice signal.  Returns 0 for a degenerate / absent
    span so the lattice branch simply does not fire.
    """
    if not np.isfinite(h) or h <= 0.0:
        return 0.0
    t = 2.0 * np.pi / h
    return float(np.abs(np.mean(np.exp(1j * t * y))))


def target_discreteness(
    y,
    name: str | None = None,
    *,
    min_unique: int = MIN_UNIQUE,
    ecf_tol: float = ECF_TOL,
    lattice_unique_ratio: float = LATTICE_UNIQUE_RATIO,
    atom_mass_max: float = ATOM_MASS_MAX,
) -> dict:
    """Decide whether target ``y`` is discrete (density rules should be skipped).

    Computed on the full ``y`` so the result is invariant to CV folds / repeats.

    Parameters
    ----------
    y : array-like
        The FULL target column (not a fold subsample).  NaNs are dropped.
    name : str, optional
        Dataset name, used only to consult ``DISCRETE_OVERRIDES``.

    Returns
    -------
    dict
        ``is_discrete`` (bool) plus the diagnostics that drove the decision:
        ``n``, ``n_unique``, ``frac_unique``, ``max_atom_mass``, ``span``,
        ``ecf_peak``, ``rule`` (which branch fired, or ``None``), and
        ``overridden`` (whether ``DISCRETE_OVERRIDES`` was applied).
    """
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    n = int(y.size)
    out = {
        "n": n,
        "n_unique": 0,
        "frac_unique": float("nan"),
        "max_atom_mass": float("nan"),
        "span": float("nan"),
        "ecf_peak": 0.0,
        "is_discrete": False,
        "rule": None,
        "overridden": False,
    }

    if n == 0:
        # No usable target values: nothing to score densities against.
        out["is_discrete"] = True
        out["rule"] = "empty"
    else:
        u, counts = np.unique(y, return_counts=True)
        k = int(u.size)
        frac_unique = k / n
        max_atom = float(counts.max()) / n
        scale = float(u[-1] - u[0]) if k > 1 else abs(float(u[0]))
        span = _lattice_span(u, scale) if k > 1 else float("nan")
        ecf_peak = _ecf_peak(y, span)

        out.update(
            n_unique=k,
            frac_unique=frac_unique,
            max_atom_mass=max_atom,
            span=span,
            ecf_peak=ecf_peak,
        )

        if k <= min_unique:
            rule = "few_unique"
        elif ecf_peak >= 1.0 - ecf_tol and frac_unique <= lattice_unique_ratio:
            rule = "lattice"
        elif max_atom >= atom_mass_max:
            rule = "atom_mass"
        else:
            rule = None
        out["is_discrete"] = rule is not None
        out["rule"] = rule

    # Manual override wins over the auto-detector.
    if name is not None and name in DISCRETE_OVERRIDES:
        forced = bool(DISCRETE_OVERRIDES[name])
        out["is_discrete"] = forced
        out["overridden"] = True
        out["rule"] = f"override={forced}"

    return out
