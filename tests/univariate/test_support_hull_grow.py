"""Grow-only density support: train range as a floor, grown to the model's hull.

The density (resampled) view no longer truncates a forecast to the train range.
Its support is ``[min(train_lo, hull_lo_i), max(train_hi, hull_hi_i)]`` per row --
the train range widened outward, per-instance and per-model, to whatever mass the
model itself reports (see ``DistributionPrediction.resampled`` and
``resample_cdf_nodes_to_support_outer_hull_y_train_set_y_instance_prediction_grid``).
These tests pin:

* the support grows to the hull beyond the train range, and floors at the train
  range when the hull is inside it (never shrinks below train);
* no predicted mass is dropped (unit mass), fixing the old truncation;
* heteroscedastic rows (different hulls) get per-row 2-D grids; a shared hull
  collapses to one 1-D grid;
* the anti-gaming property that motivated the change: a forecast that honestly
  keeps its tail mass beats a strictly-worse clipped one on the density rules,
  matching the CRPS ranking -- the reverse of the old truncated grid, which was
  blind to the honest tail.
"""

import numpy as np

from scoringbench.univariate.metrics import compute_scoring_rules
from scoringbench.univariate.wrappers import DistributionPrediction

NUM_EQUALLY_SIZED_BINS = 200


def _make(bin_edges, probas, train_range, num_equally_sized_bins=NUM_EQUALLY_SIZED_BINS):
    edges = np.asarray(bin_edges, float)
    probas = np.atleast_2d(np.asarray(probas, float))
    mids = 0.5 * (edges[..., :-1] + edges[..., 1:])
    if edges.ndim == 1:
        mean = (probas * mids[None, :]).sum(axis=1)
    else:
        mean = (probas * mids).sum(axis=1)
    return DistributionPrediction(
        probas=probas, bin_edges=edges, bin_midpoints=mids, mean=mean,
        train_range=train_range, num_equally_sized_bins=num_equally_sized_bins,
    )


# ---------------------------------------------------------------------------
# Support geometry
# ---------------------------------------------------------------------------

def test_grows_to_hull_beyond_train_range():
    edges = np.array([-8.0, -2.0, 0.0, 2.0, 8.0])       # hull [-8, 8]
    dp = _make(edges, [0.1, 0.4, 0.4, 0.1], (-2.0, 2.0))  # train floor [-2, 2]
    rg = dp.resampled
    assert rg.bin_edges.ndim == 1
    assert np.isclose(rg.bin_edges[0], -8.0)             # grew to the hull, both sides
    assert np.isclose(rg.bin_edges[-1], 8.0)
    assert np.isclose(rg.probas.sum(), 1.0)             # no tail mass dropped


def test_floors_at_train_range_when_hull_inside():
    edges = np.array([-0.5, 0.0, 0.5])                   # hull [-0.5, 0.5]
    dp = _make(edges, [0.5, 0.5], (-2.0, 3.0))          # train wider than hull
    rg = dp.resampled
    assert np.isclose(rg.bin_edges[0], -2.0)             # floored at train, never shrinks
    assert np.isclose(rg.bin_edges[-1], 3.0)


def test_asymmetric_grow_only():
    edges = np.array([-0.5, 0.0, 6.0])                   # hull high side 6 > train hi
    dp = _make(edges, [0.5, 0.5], (-2.0, 2.0))
    rg = dp.resampled
    assert np.isclose(rg.bin_edges[0], -2.0)             # low side floors at train
    assert np.isclose(rg.bin_edges[-1], 6.0)             # high side grows to hull


def test_per_instance_grid_when_hulls_differ():
    # Row 0 hull [-5, 5] (exceeds train), row 1 hull [-1, 1] (inside train).
    edges = np.array([[-5.0, 0.0, 5.0], [-1.0, 0.0, 1.0]])
    probas = np.array([[0.5, 0.5], [0.5, 0.5]])
    dp = _make(edges, probas, (-2.0, 2.0), num_equally_sized_bins=10)
    rg = dp.resampled
    assert rg.bin_edges.ndim == 2                        # genuinely per-instance
    assert np.isclose(rg.bin_edges[0, 0], -5.0) and np.isclose(rg.bin_edges[0, -1], 5.0)
    assert np.isclose(rg.bin_edges[1, 0], -2.0) and np.isclose(rg.bin_edges[1, -1], 2.0)


def test_shared_hull_collapses_to_1d_even_from_2d_edges():
    # Same hull on every row (2-D input) -> one shared 1-D grow-only grid.
    edges = np.array([[-4.0, 0.0, 4.0], [-4.0, 1.0, 4.0]])
    probas = np.array([[0.5, 0.5], [0.3, 0.7]])
    dp = _make(edges, probas, (-1.0, 1.0), num_equally_sized_bins=10)
    rg = dp.resampled
    assert rg.bin_edges.ndim == 1
    assert np.isclose(rg.bin_edges[0], -4.0) and np.isclose(rg.bin_edges[-1], 4.0)


# ---------------------------------------------------------------------------
# Anti-gaming: honest tail mass now beats a clipped forecast on density rules
# ---------------------------------------------------------------------------

def _gauss_hist(mu, sd, lo, hi, n, clip=None):
    edges = np.linspace(lo, hi, n + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    pdf = np.exp(-0.5 * ((mids - mu) / sd) ** 2)
    if clip is not None:
        pdf = pdf * ((mids >= clip[0]) & (mids <= clip[1]))
    p = pdf / pdf.sum()
    return edges, p


def test_honest_tail_beats_clipped_on_density_and_crps():
    """Grow-only makes the density rules reward honest tail coverage (like CRPS).

    HONEST keeps its true Normal tails; NARROW clips to the train range.  Targets
    include out-of-range draws.  Under the old truncated grid both scored alike
    (the grid was blind to the tail); under grow-only HONEST wins the density
    rules AND CRPS.
    """
    mu, sd = 0.0, 1.0
    train = (-1.5, 1.5)                                   # ~1.5 sd: real tail mass outside
    lo, hi, n = -10.0, 10.0, 4000
    n_test = 400
    rng = np.random.default_rng(0)
    y = rng.normal(mu, sd, size=n_test)

    he, hp = _gauss_hist(mu, sd, lo, hi, n)               # honest: full tails
    ne, npmf = _gauss_hist(mu, sd, lo, hi, n, clip=train)  # narrow: clipped to train

    honest = _make(he, np.tile(hp, (n_test, 1)), train)
    narrow = _make(ne, np.tile(npmf, (n_test, 1)), train)

    sh = compute_scoring_rules(honest, y, representation="auto")
    sn = compute_scoring_rules(narrow, y, representation="auto")

    assert sh["crps"] < sn["crps"]                       # reference truth
    for k in ("cde_loss", "dpd_beta_1.0", "pseudospherical_alpha_2.0"):
        assert sh[k] < sn[k], f"{k}: honest {sh[k]} should beat narrow {sn[k]}"


# ---------------------------------------------------------------------------
# CDF is extended EXACTLY 0 below the hull and 1 above (no eps)
# ---------------------------------------------------------------------------

def test_extension_is_exactly_zero_below_and_one_above_no_eps():
    """When the train floor reaches past the model's hull, the extension bins
    carry EXACTLY zero mass and the CDF is clamped to [0, 1] with no eps slack.

    Hull [-2, 2], train floor [-10, 10] -> grow-only grid spans [-10, 10] with
    the model's mass confined to [-2, 2].  The bins in [-10, -2] and [2, 10] must
    be exactly 0, the CDF must never dip below 0 or exceed 1, and the total mass
    must be exactly 1.
    """
    edges = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])        # hull [-2, 2]
    dp = _make(edges, [0.25, 0.25, 0.25, 0.25], (-10.0, 10.0), num_equally_sized_bins=200)
    rg = dp.resampled
    e = np.asarray(rg.bin_edges)
    p = np.asarray(rg.probas)[0]
    assert np.isclose(e[0], -10.0) and np.isclose(e[-1], 10.0)

    # Bins entirely outside the hull carry exactly zero mass (no eps leak).
    left_of_hull = e[1:] <= -2.0                          # bin fully left of hull
    right_of_hull = e[:-1] >= 2.0                         # bin fully right of hull
    assert np.all(p[left_of_hull] == 0.0)
    assert np.all(p[right_of_hull] == 0.0)

    # CDF clamped to [0, 1] exactly; mass is a valid PMF summing to 1.
    cdf = np.concatenate([[0.0], np.cumsum(p)])
    assert cdf.min() >= 0.0 and cdf.max() <= 1.0
    assert np.all(p >= 0.0)
    assert np.isclose(p.sum(), 1.0, atol=0.0, rtol=1e-12)


def test_grow_beyond_hull_keeps_unit_mass_no_eps():
    """When the hull exceeds the train range the grid is exactly the hull and the
    CDF runs 0 -> 1 across it with no truncation and no eps renormalisation gap.
    """
    edges = np.array([-8.0, -3.0, 0.0, 3.0, 8.0])        # hull exceeds train
    dp = _make(edges, [0.1, 0.4, 0.4, 0.1], (-1.0, 1.0), num_equally_sized_bins=200)
    rg = dp.resampled
    p = np.asarray(rg.probas)[0]
    cdf = np.concatenate([[0.0], np.cumsum(p)])
    assert cdf[0] == 0.0 and np.isclose(cdf[-1], 1.0, atol=0.0, rtol=1e-12)
    assert cdf.min() >= 0.0 and cdf.max() <= 1.0

