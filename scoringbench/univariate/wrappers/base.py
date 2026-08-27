"""Base classes for ScoringBench probabilistic model wrappers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .resampling_grid import (
    DistributionPredictionView,
    interpolate_cdf_to_grid_with_equally_sized_bins,
    cdf_nodes_to_native_PMF_grid,
    quantiles_to_cdf_nodes,
    resample_cdf_nodes_to_support_outer_hull_y_train_set_y_instance_prediction_grid,
    samples_to_cdf_nodes,
)


# ---------------------------------------------------------------------------
# Container: one prediction, two scoring-ready views
# ---------------------------------------------------------------------------
#
# Metrics split into two families that need incompatible grids
# (see ``tests/univariate/test_grid_robustness.py``):
#
# * GRID-ROBUST rules (CRPS, CRTS, energy, interval/coverage) are exact on the
#   model's own grid, atoms and all -- the NATIVE view preserves sharpness.
# * DENSITY-BASED rules (CDE loss, DPD, pseudospherical) read f = p_k / w_k,
#   which diverges as w_k -> 0.  The RESAMPLED view puts every model on a
#   shared-count grow-only grid so densities are comparable and atoms neutralised.
#
# Every prediction reduces to ONE lossless intermediate -- monotone CDF nodes
# ``(cdf_support_point, cdf_levels)`` -- from which both views are derived DIRECTLY, with no
# round-trip through an intermediate histogram.  The three sources
# (multi-quantile, samples, grid-native histogram) differ only in how they
# produce that intermediate and their native PMF grid; the classmethod constructors
# below dispatch each to the shortest correct path.


@dataclass
class DistributionPrediction:
    """Mother container: one prediction, two scoring-ready views.

    Build it through a source-specific constructor -- :meth:`from_multi_quantile`,
    :meth:`from_samples` or :meth:`from_histogram` -- so the lossless CDF-node
    intermediate is captured and both views are derived directly from it.  The
    dataclass fields are the NATIVE grid (a plain histogram); constructing the
    class directly is still supported (tests do) and then the resampled view is
    reconstructed from that histogram.

    * :attr:`native`    -- raw grid, atoms preserved; for grid-robust rules.
    * :attr:`resampled` -- PCHIP-resampled onto the grow-only density grid;
      for density rules.

    ``train_range = (y_lo, y_hi)`` FLOORS the density grid; the support grows
    outward per instance to the model's own hull so no tail mass is truncated.

    ``is_grid_native`` does ONE mechanical thing: the :attr:`resampled` view is the
    native PMF grid VERBATIM -- no PCHIP resample, no train-range widening.  Two
    different kinds of head want that, for opposite reasons:

    1. DISCRETIZED heads -- output already IS a PMF over borders fixed at fit time producing a prediction in 
       class L1/L2 in https://arxiv.org/pdf/1608.06802v2: TabPFN's bar
       distribution and ``XGBVectorWrapper``'s softmax over ``n_bins`` uniform
       bins.  Here the grid is AUTHORITATIVE: it is the resolution the head
       actually predicts at, so resampling would blur away real information.
    2. CONTINUOUS-density heads -- ``CDEWrapper``, ``FlexCodeWrapper``,
       ``SurjectorsWrapper``, via :func:`grid_density_to_distribution`.  These own
       a predicted ``p(y|x)`` that can be evaluated ANYWHERE; the grid is merely where
       the wrapper chose to sample it (``n_grid=200`` equally spaced points spanning
       the padded train range). 

    Quantile heads (``XGBQuantileVector``, CatBoost, NGBoost, TabICL, pytabkit,
    XGBLSS, crepes, Exaone, Nori, ...) and sample heads are NOT grid-native: they
    live in forecast family M1/M2 and do not guarantee to have forecasts in  L1/L2, so their implied density
    may carry Dirac atoms (tied quantiles, repeated draws) that would blow up the
    density based scoring rules (DPD, pseudospherical, CDE loss).  
    We adapt the resampling approach by Izbicki 2026 to smooth potential dirac delta distribution atoms in the density forcasts that neutralises those atoms.
    """
    probas: np.ndarray
    bin_edges: np.ndarray
    bin_midpoints: np.ndarray
    mean: np.ndarray
    train_range: tuple = field(kw_only=True)
    is_sample_based: bool = False
    # Number of equally-sized bins in the density (resampled) grid: the shared
    # RESOLUTION (bin COUNT), common across every model so densities compare at
    # one granularity; only the count is shared -- each grow-only SUPPORT is
    # per-instance.
    num_equally_sized_bins: int = 200
    is_grid_native: bool = False
    # The lossless CDF-node intermediate, when the prediction was built through a
    # source constructor.  Present -> the resampled view PCHIPs these nodes
    # directly; absent -> it is reconstructed from the native histogram.
    cdf_nodes: object = field(default=None, kw_only=True)

    def __post_init__(self):
        self._native: DistributionPredictionView | None = None
        self._resampled: DistributionPredictionView | None = None
        if self.train_range is None:
            raise ValueError(
                "DistributionPrediction.train_range is required and must be the "
                "train-target range (y_lo, y_hi); got None. It is the FLOOR the "
                "density grid grows outward from, per instance."
            )
        y_lo, y_hi = self.train_range
        if not (np.isfinite(y_lo) and np.isfinite(y_hi) and y_hi > y_lo):
            raise ValueError(
                f"train_range must be a finite (y_lo, y_hi) with y_hi > y_lo; "
                f"got {self.train_range!r}."
            )

    # -- source constructors ------------------------------------------------
    @classmethod
    def from_multi_quantile(cls, q, alphas, mean=None, *, train_range, **kw):
        """From a per-sample quantile matrix ``q`` (n, K) at levels ``alphas``.

        The quantile function is read as a CDF; its nodes are used DIRECTLY as
        native bin edges (atoms preserved), and the SAME nodes feed the resampled
        view.  No intermediate histogram.
        """
        cdf_support_point, cdf_levels = quantiles_to_cdf_nodes(q, alphas)
        edges, probas = cdf_nodes_to_native_PMF_grid(cdf_support_point, cdf_levels)
        mids = 0.5 * (edges[..., :-1] + edges[..., 1:])
        out_mean = (probas * mids).sum(axis=-1) if mean is None else np.asarray(mean, float).reshape(-1)
        return cls._from_native(edges, probas, out_mean, train_range,
                                cdf_nodes=(cdf_support_point, cdf_levels), **kw)

    @classmethod
    def from_samples(cls, samples, n_bins=100, mean=None, *, train_range, **kw):
        """From conditional draws ``samples`` (n_test, n_draws).

        Each row's empirical CDF (strictly increasing, de-tied) is the lossless
        intermediate: the native PMF grid bins it onto ``n_bins`` uniform bins (a
        bounded grid -- the native energy score is O(n_bins^2)), and the resampled
        view PCHIPs the SAME eCDF nodes.  No histogram round-trip.
        """
        samples = np.atleast_2d(np.asarray(samples, dtype=np.float64))
        n_bins = max(int(n_bins), 1)
        # Per-row eCDF nodes have different lengths, so keep them as a list and
        # bin each to the shared native bin count.
        node_rows = [samples_to_cdf_nodes(row) for row in samples]
        edges = np.empty((samples.shape[0], n_bins + 1), dtype=np.float64)
        probas = np.empty((samples.shape[0], n_bins), dtype=np.float64)
        for i, (x, c) in enumerate(node_rows):
            e, p = interpolate_cdf_to_grid_with_equally_sized_bins(x, c, n_bins)
            edges[i], probas[i] = e[0], p[0]
        out_mean = samples.mean(axis=1) if mean is None else np.asarray(mean, float).reshape(-1)
        return cls._from_native(edges, probas, out_mean, train_range,
                                cdf_nodes=node_rows, is_sample_based=True, **kw)

    @classmethod
    def from_histogram(cls, bin_edges, probas, mean=None, *, train_range,
                       is_grid_native=False, **kw):
        """From a model's OWN histogram ``(bin_edges, probas)``.

        For grid-native heads (TabPFN bar distribution): the native PMF grid is the
        histogram verbatim and the resampled view keeps it too.  ``mean`` defaults
        to the PMF mean.
        """
        edges = np.asarray(bin_edges, dtype=np.float64)
        probas = np.asarray(probas, dtype=np.float64)
        mids = 0.5 * (edges[..., :-1] + edges[..., 1:])
        out_mean = (probas * mids).sum(axis=-1) if mean is None else np.asarray(mean, float).reshape(-1)
        return cls._from_native(edges, probas, out_mean, train_range,
                                is_grid_native=is_grid_native, **kw)

    @classmethod
    def _from_native(cls, edges, probas, mean, train_range, **kw):
        mids = 0.5 * (np.asarray(edges)[..., :-1] + np.asarray(edges)[..., 1:])
        return cls(probas=probas, bin_edges=edges, bin_midpoints=mids,
                   mean=mean, train_range=train_range, **kw)

    # -- native view (grid-robust rules) ------------------------------------
    @property
    def native(self) -> DistributionPredictionView:
        """The raw grid, VERBATIM -- atoms and all.

        No repair is applied: a zero-width bin is a Dirac and the grid-robust
        rules score it exactly.  Degenerate grids are neutralised only where it
        matters, in the resampled view read by the width-dividing density rules.
        """
        if self._native is None:
            edges = self.bin_edges
            mids = self.bin_midpoints
            self._native = DistributionPredictionView(
                probas=self.probas, bin_edges=edges, bin_midpoints=mids,
                mean=self.mean,
                is_sample_based=self.is_sample_based,
                is_grid_native=self.is_grid_native,
            )
        return self._native

    # -- resampled view (density rules, grow-only grid) ---------------------
    @property
    def resampled(self) -> DistributionPredictionView:
        """The density-rule view on the per-instance grow-only grid.

        ``support_i = (min(train_lo, hull_lo_i), max(train_hi, hull_hi_i))`` -- the
        prediction's own hull can only widen the train range, never shrink it.
        Built from the stored CDF nodes when present (no histogram round-trip),
        else from the native histogram; a grid-native PMF grid is kept verbatim
        (scoring pads it to the observed target on its own).
        """
        if self._resampled is None:
            y_lo, y_hi = self.train_range
            n = self.num_equally_sized_bins
            if self.is_grid_native:
                edges, probas = self.bin_edges, self.probas
            elif self.cdf_nodes is not None:
                edges, probas = self._resample_stored_nodes(y_lo, y_hi, n)
            else:
                x, c = _histogram_to_cdf_nodes(self.bin_edges, self.probas)
                edges, probas = resample_cdf_nodes_to_support_outer_hull_y_train_set_y_instance_prediction_grid(x, c, y_lo, y_hi, n)
            edges = np.asarray(edges)
            mids = 0.5 * (edges[..., :-1] + edges[..., 1:])
            self._resampled = DistributionPredictionView(
                probas=probas, bin_edges=edges, bin_midpoints=mids, mean=self.mean,
                is_sample_based=self.is_sample_based, is_grid_native=self.is_grid_native,
            )
        return self._resampled

    def _resample_stored_nodes(self, y_lo, y_hi, n):
        """Resample the stored CDF nodes onto the grow-only grid.

        Quantile nodes share one 2-D array (a tuple); sample nodes are a per-row
        list of ragged eCDFs, resampled row by row (each onto the shared count).
        """
        nodes = self.cdf_nodes
        if isinstance(nodes, tuple):                        # shared 2-D nodes
            return resample_cdf_nodes_to_support_outer_hull_y_train_set_y_instance_prediction_grid(nodes[0], nodes[1], y_lo, y_hi, n)
        edges = np.empty((len(nodes), n + 1), dtype=np.float64)
        probas = np.empty((len(nodes), n), dtype=np.float64)
        for i, (x, c) in enumerate(nodes):
            e, p = resample_cdf_nodes_to_support_outer_hull_y_train_set_y_instance_prediction_grid(x, c, y_lo, y_hi, n)
            edges[i] = e if np.ndim(e) == 1 else e[0]
            probas[i] = p[0]
        return edges, probas


def _histogram_to_cdf_nodes(bin_edges, probas):
    """Monotone CDF nodes ``(x, c)`` from a raw histogram -- the direct-construction
    fallback (tests build ``DistributionPrediction`` without source nodes).

    ``x`` are the sorted edges, ``c`` the cumulative mass (``c[0]=0``, ``c[-1]=1``).
    Shape is preserved (shared 1-D grid -> 1-D ``x``).
    """
    e = np.asarray(bin_edges, dtype=np.float64)
    p = np.asarray(probas, dtype=np.float64)
    p2 = p[None, :] if p.ndim == 1 else p
    rows, n_bins = p2.shape
    p2 = np.clip(np.nan_to_num(p2, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    tot = p2.sum(axis=-1, keepdims=True)
    p2 = np.where(tot > 0.0, p2 / np.where(tot > 0.0, tot, 1.0), 1.0 / n_bins)
    c = np.concatenate([np.zeros((rows, 1)), np.cumsum(p2, axis=-1)], axis=-1)
    c[:, -1] = 1.0
    x = np.sort(e, axis=-1)
    return (x[0] if (e.ndim == 2 and e.shape[0] == 1) else x), c


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class ProbabilisticWrapper:
    """Base class for ScoringBench model wrappers.

    Subclass and implement fit(), predict(), predict_distribution().
    If predict_distribution() is not yet supported, leave it raising
    NotImplementedError — cv.py will skip distributional metrics gracefully.

    Every subclass's ``fit`` calls ``self._set_train_range(y)``.  Distribution
    predictions then read ``self._y_train_range`` instead of receiving it from
    the caller.
    """

    _y_train_range: tuple[float, float] | None = None

    def _set_train_range(self, y) -> tuple[float, float]:
        """Capture the finite training-target range."""
        arr = np.asarray(y.values if hasattr(y, "values") else y, dtype=float).reshape(-1)
        arr = np.where(np.isfinite(arr), arr, np.nan)
        self._y_train_range = (float(np.nanmin(arr)), float(np.nanmax(arr)))
        return self._y_train_range

    def fit(self, X, y) -> "ProbabilisticWrapper":
        raise NotImplementedError

    def predict(self, X) -> np.ndarray:
        raise NotImplementedError

    def predict_distribution(self, X) -> DistributionPrediction:
        """Return the predictive distribution over the test rows.

        The train-target range that seeds the grow-only density grid is read
        from ``self._y_train_range`` (captured by ``fit`` via
        :meth:`_set_train_range`); it is NOT threaded through the call.
        """
        raise NotImplementedError
