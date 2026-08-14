"""Monte-Carlo validation of the exact point↔slab (Dirac) energy-score maths.

The energy score's two expectations

    ES_β(F, y) = E|X − y|^β − ½ E|X − X'|^β ,   X, X' ~ F  (independent),

are computed in closed form by :func:`uniform_slab_pairwise_distance`
(the pairwise term ``E|X − X'|^β``) and
:func:`compute_energy_score_histogram_corrected` (the full score).  For a
histogram whose bins are uniform slabs the closed forms are *exact*; the
subtle part is a **zero-width (Dirac) bin**, i.e. a bin with a point mass at a
single location.  There the general 4-corner slab↔slab formula degenerates to
``0/0`` and is dispatched to the exact point↔slab / point↔point limits.

This module pins those exact formulas against an *independent* Monte-Carlo
estimator that shares no code with them: it draws real samples from the mixed
point/slab PMF (point bins → a fixed value, slab bins → a uniform draw) and
averages ``|X − X'|^β`` / the full energy score over many pairs.  A zero-width
bin in the sampler is a genuine Dirac atom (the draw is deterministic), so the
MC mean is the ground truth the closed form must reproduce (up to MC noise).

The tolerances are set to ``≈ 5·SE`` of the MC estimator so the tests are
sensitive to a wrong formula yet robust to sampling noise.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scoringbench._integration import uniform_slab_pairwise_distance
from scoringbench.metrics import compute_energy_score_histogram_corrected

# Deterministic CPU float64 throughout: the closed forms upcast to float64
# internally, and the MC reference is built in float64 too.
torch.cuda.is_available = lambda: False
EPS = 100 * torch.finfo(torch.float64).eps


# ---------------------------------------------------------------------------
# Monte-Carlo sampler for a mixed point/slab predictive distribution
# ---------------------------------------------------------------------------

def _sample_mixture(edges, probas, n, rng):
    """Draw ``n`` samples from the histogram mixture described by ``edges``.

    Bin ``k`` spans ``[edges[k], edges[k+1]]`` with mass ``probas[k]``:

    * a **positive-width** bin is a uniform slab → ``U[edges[k], edges[k+1]]``;
    * a **zero-width** bin (``edges[k] == edges[k+1]``) is a Dirac atom → the
      constant value ``edges[k]``.

    Returns a ``(n,)`` float64 numpy array.  This shares no code with the
    closed-form primitives under test.
    """
    edges = np.asarray(edges, dtype=np.float64)
    probas = np.asarray(probas, dtype=np.float64)
    probas = probas / probas.sum()
    a, b = edges[:-1], edges[1:]
    # Which bin each sample falls in.
    bin_idx = rng.choice(len(probas), size=n, p=probas)
    lo = a[bin_idx]
    hi = b[bin_idx]
    # Uniform inside the (possibly zero-width) slab: lo == hi ⇒ the atom.
    u = rng.random(n)
    return lo + u * (hi - lo)


def _mc_pairwise(edges, probas, beta, n, rng):
    """Monte-Carlo estimate of ``E|X − X'|^β`` and its standard error."""
    x = _sample_mixture(edges, probas, n, rng)
    xp = _sample_mixture(edges, probas, n, rng)
    d = np.abs(x - xp) ** beta
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(n))


def _mc_energy_score(edges, probas, y, beta, n, rng):
    """Monte-Carlo estimate of ``E|X − y|^β − ½ E|X − X'|^β`` and its SE."""
    x = _sample_mixture(edges, probas, n, rng)
    xp = _sample_mixture(edges, probas, n, rng)
    term1_samp = np.abs(x - y) ** beta                 # E|X − y|^β
    term2_samp = np.abs(x - xp) ** beta                # E|X − X'|^β
    es_samp = term1_samp - 0.5 * term2_samp
    return float(es_samp.mean()), float(es_samp.std(ddof=1) / np.sqrt(n))


def _closed_form_pairwise_expectation(edges, probas, beta):
    """``E|X − X'|^β = Σ_ij p_i p_j D_ij`` from the exact primitive."""
    edges_t = torch.as_tensor(edges, dtype=torch.float64)
    p_t = torch.as_tensor(probas, dtype=torch.float64)
    p_t = p_t / p_t.sum()
    D = uniform_slab_pairwise_distance(edges_t, beta, eps=EPS)
    return float(torch.einsum("i,ij,j->", p_t, D, p_t))


# ---------------------------------------------------------------------------
# Test grids: every combination of point / slab bins we care about
# ---------------------------------------------------------------------------

# edges, probas  (a zero-width run [x, x] is a Dirac atom at x)
_MIXTURES = {
    # one Dirac atom next to two slabs
    "atom+slabs": (
        [0.0, 1.0, 2.0, 2.0, 4.0],
        [0.25, 0.30, 0.20, 0.25],
    ),
    # two Dirac atoms + one slab (point↔point + point↔slab quadrants)
    "two_atoms+slab": (
        [-1.0, -1.0, 0.0, 3.0, 3.0],
        [0.35, 0.15, 0.30, 0.20],
    ),
    # a wide slab, then a Dirac atom, then another wide slab (asymmetric masses)
    "slab_atom_slab": (
        [0.0, 5.0, 6.0, 6.0, 12.0],
        [0.30, 0.10, 0.35, 0.25],
    ),
    # all slabs (control: no Dirac dispatch, pure 4-corner form)
    "all_slabs": (
        [0.0, 1.0, 2.0, 3.5, 6.0],
        [0.20, 0.30, 0.30, 0.20],
    ),
}

_BETAS = [0.5, 1.0, 1.5, 2.0]


# ---------------------------------------------------------------------------
# 1) Pairwise term  E|X − X'|^β   (the primitive)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(_MIXTURES))
@pytest.mark.parametrize("beta", _BETAS)
def test_pairwise_expectation_matches_monte_carlo(name, beta):
    """``Σ p_i p_j D_ij`` (exact primitive) == MC ``E|X − X'|^β``.

    The mixture contains genuine zero-width Dirac bins, so this exercises the
    point↔slab and point↔point dispatch paths of
    :func:`uniform_slab_pairwise_distance` against real samples.
    """
    edges, probas = _MIXTURES[name]
    rng = np.random.default_rng(20240607 + int(beta * 100))

    closed = _closed_form_pairwise_expectation(edges, probas, beta)
    mc, se = _mc_pairwise(edges, probas, beta, n=2_000_000, rng=rng)

    # 5·SE tolerance (plus a tiny absolute floor for the near-degenerate cases).
    tol = 5.0 * se + 1e-3
    assert abs(closed - mc) < tol, (
        f"[{name}] beta={beta}: closed={closed:.6f} MC={mc:.6f}±{se:.2e} "
        f"(|diff|={abs(closed - mc):.2e} > tol={tol:.2e})"
    )


# ---------------------------------------------------------------------------
# 2) Full energy score  E|X − y|^β − ½ E|X − X'|^β
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(_MIXTURES))
@pytest.mark.parametrize("beta", _BETAS)
def test_energy_score_matches_monte_carlo(name, beta):
    """``compute_energy_score_histogram_corrected`` == MC energy score.

    Validates the *whole* score (both terms) on a PMF with Dirac atoms.  ``y``
    is placed away from all atoms so term-1's own Dirac handling and term-2's
    point↔slab dispatch are both exercised.
    """
    edges, probas = _MIXTURES[name]
    y = 1.234  # generic target, not coincident with any atom or edge
    rng = np.random.default_rng(555 + int(beta * 100))

    edges_t = torch.as_tensor(edges, dtype=torch.float64)
    p_t = torch.as_tensor(probas, dtype=torch.float64)
    p_t = (p_t / p_t.sum())[None, :]                    # (1, n_bins)
    y_t = torch.as_tensor([y], dtype=torch.float64)

    out = compute_energy_score_histogram_corrected(
        p_t, edges_t, y_t, betas=[beta]
    )
    closed = out[f"energy_score_beta_{beta}"]

    mc, se = _mc_energy_score(edges, probas, y, beta, n=2_000_000, rng=rng)

    # The production score clamps at 0; only compare when the true score is
    # comfortably positive (all these mixtures are, for y well inside support).
    tol = 5.0 * se + 1e-3
    assert abs(closed - mc) < tol, (
        f"[{name}] beta={beta}: closed={closed:.6f} MC={mc:.6f}±{se:.2e} "
        f"(|diff|={abs(closed - mc):.2e} > tol={tol:.2e})"
    )


# ---------------------------------------------------------------------------
# 3) A point-mass bin exactly at y (term-1 Dirac + zero-distance term-2)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("beta", _BETAS)
def test_energy_score_atom_at_target(beta):
    """Mixture with a Dirac atom placed exactly at ``y``.

    A sample landing on that atom contributes ``|y − y|^β = 0`` to term-1, so
    the analytic score must still equal the MC estimate — a stress test for the
    ``torch.where`` zero-width branch of term-1 lining up with the sampler.
    """
    edges = [0.0, 1.0, 2.5, 2.5, 5.0]      # Dirac atom at 2.5
    probas = [0.25, 0.25, 0.30, 0.20]
    y = 2.5                                 # target sits on the atom
    rng = np.random.default_rng(9001 + int(beta * 100))

    edges_t = torch.as_tensor(edges, dtype=torch.float64)
    p_t = (torch.as_tensor(probas, dtype=torch.float64)
           / sum(probas))[None, :]
    y_t = torch.as_tensor([y], dtype=torch.float64)

    closed = compute_energy_score_histogram_corrected(
        p_t, edges_t, y_t, betas=[beta]
    )[f"energy_score_beta_{beta}"]

    mc, se = _mc_energy_score(edges, probas, y, beta, n=2_000_000, rng=rng)
    tol = 5.0 * se + 1e-3
    assert abs(closed - mc) < tol, (
        f"atom-at-y beta={beta}: closed={closed:.6f} MC={mc:.6f}±{se:.2e} "
        f"(|diff|={abs(closed - mc):.2e} > tol={tol:.2e})"
    )
