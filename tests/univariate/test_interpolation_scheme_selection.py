"""Selection test for the quantile -> density interpolation scheme.

Why this module exists
----------------------
``quantiles_to_distribution`` (and every wrapper that goes through
``base.regrid_to_uniform``) turns a per-sample quantile grid ``q(alpha)`` into a
regularly-binned PMF by reading ``(q, alpha)`` as a CDF and resampling it onto a
regular grid.  *How* that CDF is interpolated between the predicted levels is a
free choice, and it is the choice this file pins down.

The candidates, each mapping ``(q, alphas)`` to a density on a common grid:

* **A  (diff / linear)** -- the historical scheme: linear CDF interpolation at
  the bin edges, differenced into per-bin mass.  Mass-exact but the implied
  density is piecewise constant, so a sharp mode is flattened.
* **B  (gradient)** -- linear CDF on the grid nodes, ``np.gradient`` (a centred
  finite difference), clip, renormalise.  The centred difference smears a peak
  across its neighbours.
* **C1 (edge-diff)** -- B's tight support but a mass-exact forward difference.
  Numerically almost identical to B on a dense grid: the difference stencil is
  not the real lever.
* **C2 (pchip)** -- fit a *monotone* shape-preserving cubic (PCHIP) to the CDF
  nodes and read the density as its analytic derivative.  Monotone, so the CDF
  stays valid and CRPS/KS stay as good as the linear schemes, but the derivative
  is a smooth C1 curve that does not average a peak with its neighbours.
* **E  (cdf-exact)** -- the *same* PCHIP CDF as C2, but each bin's mass is the
  exact CDF increment ``C(e_{k+1}) - C(e_k)`` rather than a sampled derivative.
  C2's node-sampled derivative is a midpoint approximation of that increment; E
  is the exact integral (FTC), so it is mass-exact by construction and best on
  the density-shape metrics.  E is a strict refinement of C2, not a rival CDF.

The schemes are scored against known ground-truth distributions with the five
metrics the benchmark cares about (IAE/ISE of the density, KS of the CDF, NLL of
the log-score, CRPS of the CDF-score).  The test *asserts* the outcome that
selects E: sharing C2's monotone CDF, E wins the most per-(distribution, metric)
contests, is best-in-class on the density-shape metrics (IAE/ISE) on average,
stays level with C2 on the CDF metrics (KS/CRPS), and never loses NLL by more
than a hair.

Run standalone (prints the full comparison table)::

    python tests/test_interpolation_scheme_selection.py

or as part of the suite::

    pytest tests/test_interpolation_scheme_selection.py -q
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats
from scipy.interpolate import PchipInterpolator


# ---------------------------------------------------------------------------
# Candidate reconstruction schemes: (q_row, alphas, z_eval) -> (z, pdf)
# Each is evaluated on the SAME dense z grid so the metrics are comparable.
# ---------------------------------------------------------------------------

def _support(q, frac=0.1):
    """Padded support, a 10%-of-ptp extension on each side."""
    ptp = np.ptp(q)
    if ptp == 0.0:
        ptp = max(abs(q[0]) * 1e-6, 1e-7)
    return q.min() - frac * ptp, q.max() + frac * ptp


def scheme_A_diff(q, alphas, z_eval):
    """Linear CDF at bin edges, differenced into a piecewise-constant density."""
    q = np.sort(q)
    left = max(q[1] - q[0], 0.0) if q.size > 1 else 0.0
    right = max(q[-1] - q[-2], 0.0) if q.size > 1 else 0.0
    z_min, z_max = q[0] - left, q[-1] + right
    xa = np.concatenate([[z_min], q, [z_max]])
    ya = np.concatenate([[0.0], alphas, [1.0]])

    n_bins = z_eval.size
    edges = np.linspace(z_min, z_max, n_bins + 1)
    C = np.interp(edges, xa, ya)
    mass = np.maximum(np.diff(C), 0.0)
    s = mass.sum()
    mass = mass / s if s > 0 else np.full(n_bins, 1.0 / n_bins)
    w = np.diff(edges)
    density = mass / np.maximum(w, 1e-300)
    return 0.5 * (edges[:-1] + edges[1:]), density


def scheme_B_gradient(q, alphas, z_eval):
    """Linear CDF at grid nodes, np.gradient -> clip -> renormalise."""
    q = np.sort(q)
    dz = z_eval[1] - z_eval[0]
    cdf = np.interp(z_eval, q, alphas, left=0.0, right=1.0)
    pdf = np.maximum(np.gradient(cdf, dz), 0.0)
    total = pdf.sum() * dz
    if total > 0:
        pdf = pdf / total
    return z_eval, pdf


def scheme_C1_edge_diff(q, alphas, z_eval):
    """B's grid/support, but a mass-exact forward difference of C at the edges."""
    q = np.sort(q)
    z0, z1 = z_eval[0], z_eval[-1]
    n_bins = z_eval.size - 1
    edges = np.linspace(z0, z1, n_bins + 1)
    C = np.interp(edges, q, alphas, left=0.0, right=1.0)
    mass = np.maximum(np.diff(C), 0.0)
    s = mass.sum()
    mass = mass / s if s > 0 else np.full(n_bins, 1.0 / n_bins)
    density = mass / (edges[1] - edges[0])
    return 0.5 * (edges[:-1] + edges[1:]), density


def scheme_C2_pchip(q, alphas, z_eval):
    """Monotone cubic (PCHIP) CDF, analytic derivative as the density."""
    q = np.sort(q)
    z0, z1 = z_eval[0], z_eval[-1]
    xs, idx = np.unique(q, return_index=True)
    ys = np.maximum.accumulate(alphas[idx])
    xs = np.concatenate([[z0], xs, [z1]])
    ys = np.concatenate([[0.0], ys, [1.0]])
    xs, ui = np.unique(xs, return_index=True)
    ys = np.maximum.accumulate(ys[ui])

    pc = PchipInterpolator(xs, ys, extrapolate=True)
    pdf = np.clip(pc.derivative()(z_eval), 0.0, None)
    dz = z_eval[1] - z_eval[0]
    total = pdf.sum() * dz
    if total > 0:
        pdf = pdf / total
    return z_eval, pdf


def scheme_E_cdf_exact(q, alphas, z_eval):
    """Same PCHIP CDF as C2, but exact per-bin mass ``C(e_{k+1}) - C(e_k)``.

    C2 samples the analytic derivative at grid nodes -- a midpoint approximation
    of the bin mass.  E integrates the CDF exactly (FTC), so each bin's mass is
    the true CDF increment: mass-exact by construction, no renormalisation.  The
    density is that mass divided by the bin width.
    """
    q = np.sort(q)
    z0, z1 = z_eval[0], z_eval[-1]
    xs, idx = np.unique(q, return_index=True)
    ys = np.maximum.accumulate(alphas[idx])
    xs = np.concatenate([[z0], xs, [z1]])
    ys = np.concatenate([[0.0], ys, [1.0]])
    xs, ui = np.unique(xs, return_index=True)
    ys = np.maximum.accumulate(ys[ui])

    pc = PchipInterpolator(xs, ys, extrapolate=True)
    n_bins = z_eval.size - 1
    edges = np.linspace(z0, z1, n_bins + 1)
    C = np.clip(pc(edges), 0.0, 1.0)
    mass = np.maximum(np.diff(C), 0.0)
    s = mass.sum()
    mass = mass / s if s > 0 else np.full(n_bins, 1.0 / n_bins)
    density = mass / np.diff(edges)
    return 0.5 * (edges[:-1] + edges[1:]), density


SCHEMES = {
    "A diff": scheme_A_diff,
    "B grad": scheme_B_gradient,
    "C1 edge-diff": scheme_C1_edge_diff,
    "C2 pchip": scheme_C2_pchip,
    "E cdf-exact": scheme_E_cdf_exact,
}
METRICS = ("IAE", "ISE", "KS", "NLL", "CRPS")

# The scheme this test exists to select: E (exact per-bin CDF increment on the
# same monotone PCHIP CDF as C2).  C2 remains the reference it refines.
WINNER = "E cdf-exact"
REFERENCE = "C2 pchip"


# ---------------------------------------------------------------------------
# Scoring a reconstructed density against a known truth.
# ---------------------------------------------------------------------------

def _crps_from_cdf(z, C_hat, ys):
    """Mean CRPS of predictive CDF ``C_hat`` (grid ``z``) at observations ``ys``."""
    dz = z[1] - z[0]
    step = (z[None, :] >= ys[:, None]).astype(np.float64)
    integrand = (C_hat[None, :] - step) ** 2
    return np.trapezoid(integrand, dx=dz, axis=1).mean()


def score_row(z, pdf_hat, dist, ys):
    """IAE, ISE, KS, NLL, CRPS of ``pdf_hat`` (grid ``z``) against ``dist``.

    ``ys`` are shared across schemes so NLL and CRPS see identical draws.
    """
    dz = z[1] - z[0]
    f_true = dist.pdf(z)

    iae = np.sum(np.abs(pdf_hat - f_true)) * dz
    ise = np.sum((pdf_hat - f_true) ** 2) * dz

    C_hat = np.cumsum(pdf_hat) * dz
    C_hat = C_hat - C_hat[0]
    ks = np.max(np.abs(C_hat - dist.cdf(z)))

    ysc = np.clip(ys, z[0], z[-1])
    f_at_y = np.maximum(np.interp(ysc, z, pdf_hat), 1e-12)
    nll = -np.mean(np.log(f_at_y))

    crps = _crps_from_cdf(z, C_hat, ysc)
    return iae, ise, ks, nll, crps


# ---------------------------------------------------------------------------
# Ground-truth distribution zoo.
# ---------------------------------------------------------------------------

class _Mixture:
    """Minimal Gaussian-mixture with pdf/cdf/ppf/rvs for the test zoo."""

    def __init__(self, comps, weights):
        self.comps = comps
        self.w = np.asarray(weights, float)
        self.w /= self.w.sum()

    def pdf(self, x):
        return sum(w * c.pdf(x) for w, c in zip(self.w, self.comps))

    def cdf(self, x):
        return sum(w * c.cdf(x) for w, c in zip(self.w, self.comps))

    def ppf(self, a):
        grid = np.linspace(-20, 20, 200001)
        return np.interp(a, self.cdf(grid), grid)

    def rvs(self, size, random_state=None):
        rng = np.random.default_rng(random_state)
        k = rng.choice(len(self.comps), size=size, p=self.w)
        out = np.empty(size)
        for j, c in enumerate(self.comps):
            m = k == j
            out[m] = c.rvs(size=int(m.sum()), random_state=rng.integers(1 << 31))
        return out


def make_dists():
    return {
        "normal": stats.norm(loc=0.0, scale=1.0),
        "skew-normal": stats.skewnorm(a=6.0, loc=-1.0, scale=1.5),
        "heavy-tail (t3)": stats.t(df=3, loc=0.0, scale=1.0),
        "bimodal": _Mixture([stats.norm(-2, 0.6), stats.norm(2.5, 0.9)], [0.4, 0.6]),
        "lognormal": stats.lognorm(s=0.6, scale=np.exp(0.5)),
        "sharp-peak": stats.norm(loc=0.0, scale=0.15),
        "uniform": stats.uniform(loc=-1.0, scale=3.0),
    }


# ---------------------------------------------------------------------------
# Driver: reconstruct + score every scheme on every distribution.
# ---------------------------------------------------------------------------

def _evaluate(n_alphas=199, n_grid=400, seed=0):
    """Return (totals, wins, table) for all schemes over the zoo.

    ``totals[metric][scheme]`` is the summed metric, ``wins[scheme]`` the number
    of per-(dist, metric) contests won, ``table`` a list of
    ``(dist, metric, {scheme: value}, best)`` rows for optional printing.
    """
    alphas = np.linspace(0.005, 0.995, n_alphas)
    dists = make_dists()
    rng = np.random.default_rng(seed)
    names = list(SCHEMES)

    totals = {m: {n: 0.0 for n in names} for m in METRICS}
    wins = {n: 0 for n in names}
    table = []

    for dname, dist in dists.items():
        q = dist.ppf(alphas)
        z_min, z_max = _support(q, frac=0.1)
        z_eval = np.linspace(z_min, z_max, n_grid)
        ys = dist.rvs(size=2000, random_state=rng)

        pdfs = {}
        for n, fn in SCHEMES.items():
            zz, pdf = fn(q, alphas, z_eval)
            pdfs[n] = np.interp(z_eval, zz, pdf)

        scores = {n: score_row(z_eval, pdfs[n], dist, ys) for n in names}
        for mi, metric in enumerate(METRICS):
            vals = {n: scores[n][mi] for n in names}
            best = min(vals, key=vals.get)
            wins[best] += 1
            for n in names:
                totals[metric][n] += vals[n]
            table.append((dname, metric, vals, best))

    ndist = len(dists)
    means = {m: {n: totals[m][n] / ndist for n in names} for m in METRICS}
    return means, wins, table


def _print_table(means, wins, table):
    names = list(SCHEMES)
    col = "".join(f"{n:>14}" for n in names)
    header = f"{'distribution':<16}{'metric':<6}{col}{'best':>14}"
    print("\n" + header)
    print("-" * len(header))
    last = None
    for dname, metric, vals, best in table:
        if last is not None and dname != last:
            print()
        last = dname
        row = "".join(f"{vals[n]:>14.5f}" for n in names)
        print(f"{dname:<16}{metric:<6}{row}{best:>14}")

    print("\n" + "=" * len(header))
    print(f"{'MEAN over zoo':<16}{'metric':<6}{col}{'best':>14}")
    print("-" * len(header))
    for metric in METRICS:
        vals = means[metric]
        best = min(vals, key=vals.get)
        row = "".join(f"{vals[n]:>14.5f}" for n in names)
        print(f"{'':<16}{metric:<6}{row}{best:>14}")
    print("\nper-(dist,metric) wins:  " + "  ".join(f"{n}={wins[n]}" for n in names))


# ---------------------------------------------------------------------------
# The assertions that pin the scheme choice.
# ---------------------------------------------------------------------------

def test_cdf_exact_wins_the_most_contests():
    """E (cdf-exact) wins more per-(dist, metric) contests than any other scheme.

    No single scheme takes a strict majority of the 35 contests -- NLL and some
    CDF ties go elsewhere -- but E is the clear plurality winner, and in
    particular it out-wins the scheme it refines, C2.
    """
    _, wins, _ = _evaluate()
    top = max(wins, key=wins.get)
    assert top == WINNER, f"expected {WINNER} to win the most contests: {wins}"
    assert wins[WINNER] > wins[REFERENCE], (
        f"{WINNER} won {wins[WINNER]} vs {REFERENCE} {wins[REFERENCE]}: {wins}"
    )


def test_cdf_exact_is_best_on_density_shape_metrics():
    """E is best-in-class on IAE and ISE averaged over the zoo.

    Exact per-bin mass (the CDF increment) removes the midpoint quadrature error
    in C2's node-sampled derivative, so the reconstructed density matches the
    truth more closely in both L1 (IAE) and L2 (ISE).
    """
    means, _, _ = _evaluate()
    for metric in ("IAE", "ISE"):
        best = min(means[metric], key=means[metric].get)
        assert best == WINNER, (
            f"expected {WINNER} best on {metric}, got {best}: {means[metric]}"
        )


def test_cdf_exact_stays_level_with_pchip_on_cdf_metrics():
    """E matches C2 on the CDF metrics (KS, CRPS) -- they share the same CDF.

    E only reassigns per-bin mass; the underlying monotone PCHIP CDF is C2's, so
    KS and CRPS must be effectively identical (within 1% / a hair).
    """
    means, _, _ = _evaluate()
    assert means["KS"][WINNER] <= means["KS"][REFERENCE] * 1.01, (
        f"E KS {means['KS'][WINNER]:.5f} vs C2 {means['KS'][REFERENCE]:.5f}"
    )
    assert means["CRPS"][WINNER] <= means["CRPS"][REFERENCE] * 1.01, (
        f"E CRPS {means['CRPS'][WINNER]:.5f} vs C2 {means['CRPS'][REFERENCE]:.5f}"
    )


def test_cdf_exact_never_far_behind_on_nll():
    """E's NLL is within 2% of the best scheme's NLL, and beats C2.

    A (piecewise-constant density) can edge out the PCHIP schemes on mean NLL --
    a density-sampling artifact of scoring the log-score at grid nodes.  E does
    not chase that: it keeps a far better density *shape* while landing a hair
    ahead of C2 on NLL and comfortably within 2% of the best scheme.
    """
    means, _, _ = _evaluate()
    best_nll = min(means["NLL"].values())
    assert means["NLL"][WINNER] <= best_nll * 1.02, (
        f"{WINNER} NLL {means['NLL'][WINNER]:.5f} vs best {best_nll:.5f} "
        f"(> 2% behind): {means['NLL']}"
    )
    assert means["NLL"][WINNER] <= means["NLL"][REFERENCE], (
        f"{WINNER} NLL {means['NLL'][WINNER]:.5f} should not exceed "
        f"{REFERENCE} {means['NLL'][REFERENCE]:.5f}: {means['NLL']}"
    )


def test_gradient_scheme_loses_as_expected():
    """B (np.gradient) wins nothing: it is dominated everywhere.

    Documents the negative result that ruled the gradient snippet out -- the
    centred finite difference smears every peak, so it is never best on any
    contest.
    """
    _, wins, _ = _evaluate()
    assert wins["B grad"] == 0, f"gradient scheme unexpectedly won: {wins}"


if __name__ == "__main__":
    means, wins, table = _evaluate()
    _print_table(means, wins, table)
