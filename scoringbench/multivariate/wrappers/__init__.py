"""Multivariate ScoringBench model wrappers.

Layout mirrors ``scoringbench.univariate.wrappers`` but the wrappers are built
*directly* on the base libraries (``tabpfn``, ``tabicl``) — nothing here imports
from ``scoringbench.univariate``.

* :class:`BaseSampler` — abstract per-dimension conditional sampler with shared
  monotone-PCHIP ``cdf`` / ``quantile`` / ``sample``.
* :class:`TabPFNSampler`, :class:`TabICLSampler` — concrete samplers, one per
  file, each implementing only the model-specific CDF grid.
* :class:`IndependentMultiOutputWrapper`, :class:`CopulaMultiOutputWrapper`,
  :class:`ChainedMultiOutputWrapper` — the three composition modes, all sharing
  ``_ComposedMultiOutputWrapper``.
"""

from __future__ import annotations

from .base import MultivariateWrapper, as_2d_features, as_2d_targets
from .base_sampler import BaseSampler
from .baselines import (
    ChainedMultiOutputWrapper,
    CopulaMultiOutputWrapper,
    IndependentMultiOutputWrapper,
)
from .forest_diffusion_sampler import ForestDiffusionMultiOutputWrapper
from .sample_based import SampleBasedWrapper
from .tabicl_sampler import TabICLSampler
from .tabpfn_sampler import TabPFNSampler

__all__ = [
    "MultivariateWrapper",
    "SampleBasedWrapper",
    "BaseSampler",
    "TabPFNSampler",
    "TabICLSampler",
    "IndependentMultiOutputWrapper",
    "CopulaMultiOutputWrapper",
    "ChainedMultiOutputWrapper",
    "ForestDiffusionMultiOutputWrapper",
    "as_2d_features",
    "as_2d_targets",
]
