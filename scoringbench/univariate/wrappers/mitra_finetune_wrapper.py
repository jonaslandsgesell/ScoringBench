"""Mitra-finetune regressor wrapper for ScoringBench (v0.2.0 interface).

``mitra-finetune`` v0.2.0 exposes ``model.predict_distribution(X)`` returning a
``RegressionDistribution`` with per-bag-child histograms (``bin_edges``
``(n_members, n_bins+1)``, ``probabilities`` ``(n_members, n_test, n_bins)``);
each child bins the target on its own grid, so the grids differ across children.

This wrapper collapses those per-child histograms to ONE shared union-edge grid
— an *exact* re-expression of the mixture (the mixture CDF is piecewise-linear
between the union of all members' edges) — and returns a grid-native
``DistributionPrediction`` comparable to TabPFN's bar distribution and
``XGBVectorWrapper``.  No monkey-patching / capture hooks needed.

**GPU-only.**  Requires ``autogluon.tabular[mitra]>=1.6`` + ``tabarena`` + CUDA
(``flash-attn`` optional).  Imports are lazy.  ``checkpoint_dir`` points at a
downloaded ``autogluon/mitra-regressor-2`` checkpoint.  License: Apache-2.0.
"""

from __future__ import annotations

import contextlib
import os
import sys

import numpy as np

from .base import DistributionPrediction, ProbabilisticWrapper

# The recipe is vendored under ``additional_models/mitra-finetune``; only the
# thin ``mitra_finetune`` package is imported from source (deps from the env).
_MITRA_SRC = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..",
        "additional_models", "mitra-finetune", "src",
    )
)


def resolve_mitra2_checkpoint() -> str:
    """Locate the ``autogluon/mitra-regressor-2`` checkpoint directory.

    Order: ``$MITRA_CKPT``/``$CKPT`` (if a dir), else newest snapshot under the
    HF hub cache.  Raises ``FileNotFoundError`` if none is found.
    """
    import glob

    for env in ("MITRA_CKPT", "CKPT"):
        val = os.environ.get(env)
        if val and os.path.isdir(val):
            return val

    cache = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    pattern = os.path.join(
        cache, "hub", "models--autogluon--mitra-regressor-2", "snapshots", "*"
    )
    snaps = sorted(
        (p for p in glob.glob(pattern) if os.path.isdir(p)), key=os.path.getmtime
    )
    if snaps:
        return snaps[-1]
    raise FileNotFoundError(
        "Could not resolve a mitra-regressor-2 checkpoint. Set $MITRA_CKPT to a "
        "checkpoint directory, or download it into the HF cache "
        "(models--autogluon--mitra-regressor-2)."
    )


def _import_mitra_finetune():
    """Import ``MitraFinetune`` from the vendored source tree (lazy)."""
    if os.path.isdir(_MITRA_SRC) and _MITRA_SRC not in sys.path:
        sys.path.insert(0, _MITRA_SRC)
    try:
        from mitra_finetune import MitraFinetune
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(
            f"Could not import the vendored 'mitra_finetune' recipe from "
            f"{_MITRA_SRC!r}. Ensure the sources are present and the backend "
            "('autogluon.tabular[mitra]>=1.6', 'tabarena', torch) is installed."
        ) from exc
    return MitraFinetune


def _mixture_to_shared_grid(bin_edges, probabilities):
    """Collapse per-child histograms to ONE shared union-edge PMF grid (exact).

    ``bin_edges`` (n_members, n_bins+1) and ``probabilities`` (n_members,
    n_test, n_bins) -> ``(knots, probs)``: ``knots`` (K+1,) is the sorted union
    of all members' edges; ``probs`` (n_test, K) is the exact mixture mass per
    union bin (differencing the piecewise-linear mixture CDF at the knots).
    """
    edges = np.asarray(bin_edges, dtype=np.float64)
    probs = np.asarray(probabilities, dtype=np.float64)
    if edges.ndim == 1:
        edges = edges[None, :]
    if probs.ndim == 2:
        probs = probs[None, :, :]
    n_members, n_test, n_bins = probs.shape

    knots = np.unique(edges.ravel())
    F = np.zeros((n_test, knots.shape[0]), dtype=np.float64)
    for m in range(n_members):
        e = edges[m]
        pm = probs[m].copy()
        row_sum = pm.sum(axis=1, keepdims=True)
        pm = np.where(row_sum > 0, pm / row_sum, 1.0 / n_bins)
        cum0 = np.concatenate([np.zeros((n_test, 1)), np.cumsum(pm, axis=1)], axis=1)
        j = np.searchsorted(e, knots, side="right") - 1
        below = j < 0
        jc = np.clip(j, 0, n_bins - 1)
        inside = (knots >= e[0]) & (knots <= e[-1])
        frac = np.clip((knots - e[jc]) / (e[jc + 1] - e[jc]), 0.0, 1.0)
        fm = cum0[:, jc] + pm[:, jc] * frac[None, :]
        F += np.where(inside[None, :], fm, np.where(below[None, :], 0.0, 1.0))
    F /= n_members
    np.clip(F, 0.0, 1.0, out=F)
    F[:, 0], F[:, -1] = 0.0, 1.0

    out = np.diff(F, axis=1)
    np.clip(out, 0.0, None, out=out)
    row_sum = out.sum(axis=1, keepdims=True)
    return knots, np.where(row_sum > 0, out / row_sum, 1.0 / out.shape[1])


@contextlib.contextmanager
def _apply_recipe_env(overrides):
    """Temporarily set Mitra recipe env-var levers (``None`` skipped); restore.

    Levers: ``MITRA_FT_STEPS`` (fine-tune epochs; 0 = pure in-context),
    ``MITRA_SUPPORT_CAP`` / ``MITRA_PREDICT_SUPPORT_CAP`` (in-context support row
    caps — dominant VRAM levers), ``MITRA_FAST_PREDICT_QCHUNK`` (predict chunk).
    """
    sentinel = object()
    saved: dict[str, object] = {}
    try:
        for key, val in overrides.items():
            if val is None:
                continue
            saved[key] = os.environ.get(key, sentinel)
            os.environ[key] = str(int(val))
        yield
    finally:
        for key, prev in saved.items():
            if prev is sentinel:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prev  # type: ignore[assignment]


@contextlib.contextmanager
def _patch_ag_mem_ratio(ratio):
    """Temporarily raise AutoGluon's ``max_memory_usage_ratio`` guard.

    AutoGluon raises ``NotEnoughMemoryError`` when a model's estimated host-RAM
    footprint exceeds the ratio; the recipe never sets it, so tight machines
    abort the bagged fit.  ``None`` is a no-op; ``float("inf")`` disables it.
    """
    if ratio is None:
        yield
        return
    try:
        from autogluon.core.models.abstract import _auxiliary_params as _aux
    except Exception:  # pragma: no cover - AutoGluon optional at import time
        yield
        return
    field = _aux.AuxiliaryParams.__dataclass_fields__["max_memory_usage_ratio"]
    prev = field.default
    try:
        field.default = ratio
        yield
    finally:
        field.default = prev


class MitraFinetuneWrapper(ProbabilisticWrapper):
    """Wrap ``MitraFinetune`` (regression) via its ``predict_distribution`` API.

    Collapses the per-bag-child mixture to one shared union-edge PMF grid
    (exact) and returns a grid-native ``DistributionPrediction``.

    Key params: ``checkpoint_dir`` (``autogluon/mitra-regressor-2`` dir/``.pt``);
    ``num_bag_folds`` (child models mixed in); ``fine_tune`` / ``ft_steps`` (0 =
    pure in-context; ``ft_steps`` wins); ``support_cap`` /
    ``predict_support_cap`` / ``predict_qchunk`` (VRAM levers, ``None`` = recipe
    default); ``empty_cache`` (free CUDA after predict); ``mem_usage_ratio``
    (override AutoGluon's host-RAM guard for the fit).
    """

    def __init__(
        self,
        checkpoint_dir: str,
        *,
        time_limit: int = 3600,
        device: str = "cuda",
        random_state: int = 0,
        num_bag_folds: int = 8,
        eval_metric: str | None = None,
        fine_tune: bool = True,
        ft_steps: int | None = None,
        support_cap: int | None = None,
        predict_support_cap: int | None = None,
        predict_qchunk: int | None = None,
        empty_cache: bool = True,
        mem_usage_ratio: float | None = None,
    ) -> None:
        _oint = lambda v: None if v is None else int(v)  # noqa: E731
        self.checkpoint_dir = checkpoint_dir
        self.time_limit = int(time_limit)
        self.device = device
        self.random_state = int(random_state)
        self.num_bag_folds = int(num_bag_folds)
        self.eval_metric = eval_metric
        self.fine_tune = bool(fine_tune)
        self.ft_steps = _oint(ft_steps)
        self.support_cap = _oint(support_cap)
        self.predict_support_cap = _oint(predict_support_cap)
        self.predict_qchunk = _oint(predict_qchunk)
        self.empty_cache = bool(empty_cache)
        self.mem_usage_ratio = None if mem_usage_ratio is None else float(mem_usage_ratio)
        self._model = None

    def _recipe_env(self) -> dict[str, int | None]:
        ft = self.ft_steps if self.ft_steps is not None else (
            0 if not self.fine_tune else None
        )
        return {
            "MITRA_FT_STEPS": ft,
            "MITRA_SUPPORT_CAP": self.support_cap,
            "MITRA_PREDICT_SUPPORT_CAP": self.predict_support_cap,
            "MITRA_FAST_PREDICT_QCHUNK": self.predict_qchunk,
        }

    @staticmethod
    def _release_cuda() -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - torch/CUDA optional
            pass

    def fit(self, X, y) -> "MitraFinetuneWrapper":
        """Register training data (Mitra fine-tunes at predict time)."""
        MitraFinetune = _import_mitra_finetune()
        y_arr = np.asarray(
            y.values if hasattr(y, "values") else y, dtype=np.float64
        ).reshape(-1)
        finite = np.isfinite(y_arr)
        if not finite.all():
            X = X.iloc[finite] if hasattr(X, "iloc") else np.asarray(X)[finite]
            y_arr = y_arr[finite]
        if y_arr.size == 0:
            raise ValueError("No finite training targets after sanitization")
        self._set_train_range(y_arr)

        self._model = MitraFinetune(
            checkpoint_dir=self.checkpoint_dir,
            problem_type="regression",
            time_limit=self.time_limit,
            eval_metric=self.eval_metric,
            device=self.device,
            random_state=self.random_state,
            in_process=True,  # so env-var overrides take effect in this process
            num_bag_folds=self.num_bag_folds,
        )
        with _patch_ag_mem_ratio(self.mem_usage_ratio):
            self._model.fit(X, y_arr)
        return self

    def predict(self, X) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("fit() must be called before predict()")
        try:
            with _patch_ag_mem_ratio(self.mem_usage_ratio), _apply_recipe_env(
                self._recipe_env()
            ):
                return np.asarray(self._model.predict(X), dtype=np.float64)
        finally:
            if self.empty_cache:
                self._release_cuda()

    def predict_distribution(self, X) -> DistributionPrediction:
        """Bagged fine-tune + predict; return the grid-native distribution.

        Calls ``MitraFinetune.predict_distribution`` and collapses the returned
        per-child ``RegressionDistribution`` to one shared union-edge PMF grid.
        ``mean`` uses the library's own ``point_prediction`` (same forward pass).
        """
        if self._model is None:
            raise RuntimeError("fit() must be called before predict_distribution()")
        try:
            with _patch_ag_mem_ratio(self.mem_usage_ratio), _apply_recipe_env(
                self._recipe_env()
            ):
                dist = self._model.predict_distribution(X)
        finally:
            if self.empty_cache:
                self._release_cuda()

        knots, probs = _mixture_to_shared_grid(dist.bin_edges, dist.probabilities)
        if dist.point_prediction is not None:
            mean = np.asarray(dist.point_prediction, dtype=np.float64)
        else:
            mids = 0.5 * (knots[:-1] + knots[1:])
            mean = (probs * mids[None, :]).sum(axis=1)

        return DistributionPrediction.from_histogram(
            bin_edges=knots,
            probas=probs,
            mean=mean,
            train_range=self._y_train_range,
            is_grid_native=True,
        )
