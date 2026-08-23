"""Model registry for the multivariate benchmark.

Each entry is a zero-arg factory returning a fresh, unfitted
:class:`MultivariateWrapper`. We cross two *composition strategies* with two
*base samplers*:

Strategies
----------
* **independent** — :class:`IndependentMultiOutputWrapper` (baseline **A**):
  each target dimension is predicted independently; the joint has no learned
  cross-dimensional dependence.
* **copula** — :class:`CopulaMultiOutputWrapper`: the same conditional marginals
  as *independent*, glued with a vine copula fit on their PIT pseudo-
  observations; captures residual cross-dimensional dependence.
* **chained** — :class:`ChainedMultiOutputWrapper` (baseline **B**): a sampling
  analogue of the product rule ``Π_k p(y_k | x, y_{<k})``; captures
  cross-dimensional dependence.

Direct multivariate model
--------------------------
* **forest_diffusion** — :class:`ForestDiffusionMultiOutputWrapper`: a *native*
  multivariate sampler (Forest-Diffusion / Forest-Flow) that diffuses the whole
  ``d``-dimensional target block jointly, conditioned on the features.  It is
  **not** crossed with the composition strategies — there is a single
  ``forest_diffusion`` entry, no independent / copula / chained variants.

Base samplers
-------------
* **tabpfn_v3** — :class:`TabPFNSampler` on the TabPFN v3 checkpoint.
* **tabpfn_realv2_5** — :class:`TabPFNSampler` on the TabPFN v2.5 (``real``)
  checkpoint.
* **tabicl** — :class:`TabICLSampler`.

Everything is built directly on the ``tabpfn`` / ``tabicl`` libraries; nothing
here imports ``scoringbench.univariate``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from . import config as cfg
from .wrappers import (
    ChainedMultiOutputWrapper,
    CopulaMultiOutputWrapper,
    ForestDiffusionMultiOutputWrapper,
    IndependentMultiOutputWrapper,
    TabICLSampler,
    TabPFNSampler,
)

# ---------------------------------------------------------------------------
# TabPFN checkpoint paths (mirrors the univariate front script)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH_MAP = {
    "realv2_5": str(_PROJECT_ROOT / "tabpfn-v2.5-regressor-v2.5_real.ckpt"),
    "v2_6": str(_PROJECT_ROOT / "tabpfn-v2.6-regressor-v2.6_default.ckpt"),
    "v3": str(_PROJECT_ROOT / "tabpfn-v3-regressor-v3_default.ckpt"),
}


# ---------------------------------------------------------------------------
# Per-dimension base sampler factories
# ---------------------------------------------------------------------------

def _tabpfn_v3_sampler() -> TabPFNSampler:
    return TabPFNSampler(
        model_path=MODEL_PATH_MAP["v3"],
        ignore_pretraining_limits=True,
    )


def _tabpfn_realv2_5_sampler() -> TabPFNSampler:
    return TabPFNSampler(
        model_path=MODEL_PATH_MAP["realv2_5"],
        ignore_pretraining_limits=True,
    )


def _tabicl_sampler() -> TabICLSampler:
    return TabICLSampler()


# Named per-dim samplers we cross with the composition strategies.
BASE_SAMPLERS: dict[str, Callable[[], object]] = {
    "tabpfn_v3": _tabpfn_v3_sampler,
    "tabpfn_realv2_5": _tabpfn_realv2_5_sampler,
    "tabicl": _tabicl_sampler,
}


# ---------------------------------------------------------------------------
# Model registry — {name: zero-arg factory -> MultivariateWrapper}
# ---------------------------------------------------------------------------

def _make_independent(sampler_factory: Callable[[], object]):
    return lambda: IndependentMultiOutputWrapper(
        sampler_factory=sampler_factory,
        n_draws=int(cfg.N_DRAWS),
        seed=int(cfg.SEED),
    )


def _make_copula(sampler_factory: Callable[[], object]):
    return lambda: CopulaMultiOutputWrapper(
        sampler_factory=sampler_factory,
        n_draws=int(cfg.N_DRAWS),
        seed=int(cfg.SEED),
        pit_jitter=float(getattr(cfg, "COPULA_PIT_JITTER", 1e-4)),
    )


def _make_chained(sampler_factory: Callable[[], object]):
    return lambda: ChainedMultiOutputWrapper(
        sampler_factory=sampler_factory,
        n_draws=int(cfg.N_DRAWS),
        seed=int(cfg.SEED),
        n_orders=int(getattr(cfg, "CHAINED_N_ORDERS", 3)),
    )


def _forest_diffusion() -> ForestDiffusionMultiOutputWrapper:
    # Native multivariate sampler: the whole d-dim target block is diffused
    # jointly (no per-dimension composition), so there is exactly one direct
    # model — no independent / copula / chained variants.
    return ForestDiffusionMultiOutputWrapper(
        n_draws=int(cfg.N_DRAWS),
        random_state=int(cfg.SEED),
    )


#: Comment out entries you do not want to run locally.
MODELS: dict[str, Callable[[], object]] = {}
for _base_name, _base_factory in BASE_SAMPLERS.items():
    MODELS[f"independent_{_base_name}"] = _make_independent(_base_factory)
    MODELS[f"copula_{_base_name}"] = _make_copula(_base_factory)
    MODELS[f"chained_{_base_name}"] = _make_chained(_base_factory)

# Direct multivariate sampler (no composition strategy applied).
MODELS["forest_diffusion"] = _forest_diffusion
