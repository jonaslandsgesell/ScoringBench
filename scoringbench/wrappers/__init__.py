"""Probabilistic model wrappers for ScoringBench.

Re-exports all wrappers for backward compatibility. Individual wrappers
live in their own sub-modules:

    wrappers/base.py                  — DistributionPrediction, ProbabilisticWrapper
    wrappers/tabpfn.py                — TabPFNWrapper
    wrappers/tabicl.py                — TabICLWrapper
    wrappers/xgb_vector.py            — XGBVectorWrapper
"""

from .base import DistributionPrediction, ProbabilisticWrapper
from .tabpfn import TabPFNWrapper, FinetuneTabPFNWrapper
from .tabicl import TabICLWrapper, FinetuneTabICLWrapper
from .xgb_vector import XGBVectorWrapper, XGBQuantileVectorWrapper
from .xgblss_wrapper import XGBLSSWrapper
from .pytabkit import (
    PytabkitRealMLPWrapper,
    PytabkitRealMLPHPOWrapper,
    PytabkitTabMDWrapper,
    PytabkitTabMHPOWrapper,
)
from .catboost_wrapper import CatBoostQuantileWrapper

__all__ = [
    "DistributionPrediction",
    "ProbabilisticWrapper",
    "TabPFNWrapper",
    "FinetuneTabPFNWrapper",
    "TabICLWrapper",
    "FinetuneTabICLWrapper",
    "XGBVectorWrapper",
    "XGBQuantileVectorWrapper",
    "XGBLSSWrapper",
    "PytabkitRealMLPWrapper",
    "PytabkitRealMLPHPOWrapper",
    "PytabkitTabMDWrapper",
    "PytabkitTabMHPOWrapper",
    "CatBoostQuantileWrapper",
]
