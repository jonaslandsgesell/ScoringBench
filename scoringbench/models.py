"""Model registry — kept for backwards compatibility.

Models are now defined directly in run_bench_regression.py as the MODELS dict.
Wrappers live in scoringbench/wrappers.py.

To add a new model: subclass ProbabilisticWrapper in wrappers.py,
then add it to MODELS in run_bench_regression.py.
"""

from .wrappers import ProbabilisticWrapper, DistributionPrediction, TabPFNWrapper, FinetuneTabPFNWrapper, TabICLWrapper, XGBVectorWrapper  # noqa: F401
