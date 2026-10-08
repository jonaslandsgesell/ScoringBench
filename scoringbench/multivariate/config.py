"""Benchmark-wide configuration for the multivariate ScoringBench.

Shared evaluation and wrapper settings apply to both data sources. The
source-specific sections configure Source 1 (``--source scoringbench``:
real-data feature promotion) and Source 2 (``--source synthetic``: nonlinear
conditional means with randomized R-vine residuals).
"""

# ---------------------------------------------------------------------------
# Shared CV constants (both sources; mirrors univariate)
# ---------------------------------------------------------------------------
SEED = 42
N_FOLDS = 5
N_REPEATS_CV = 1
SAMPLE_SIZE = 3000

# ---------------------------------------------------------------------------
# Shared multivariate settings (both sources)
# ---------------------------------------------------------------------------

# Target dimension d for both sources. Source 1 promotes (d-1) feature columns
# alongside the original target. Source 2 samples a d-dimensional residual
# vine independently of SYNTHETIC_N_FEATURES feature columns.
# This value is echoed into the output folder name together with SAMPLE_SIZE
# so different d / sample-size sweeps never overwrite each other.
TARGET_DIM = 2

# Number of Monte-Carlo draws every model emits per test instance.  Pinned
# benchmark-wide because the Monte-Carlo estimates used by the scoring
# rules still have a finite-sample bias/variance that depends on the number of
# draws m; fixing m across all models keeps the comparison apples-to-apples.
# (The energy-score term-2 estimator 1/(m(m-1)) Σ_{i≠j} is unbiased for every
# m ≥ 2, but its variance — and the variogram/DSS moment estimates — still
# shrink with m, so a shared m is required for a fair leaderboard.)
N_DRAWS = 600

# ---------------------------------------------------------------------------
# Monte-Carlo convergence diagnostic (MPSRF / multivariate Gelman–Rubin)
# ---------------------------------------------------------------------------
# Diagnostic parameters (n_groups, epsilon, max_instances, max_samples,
# batch_size) live in ``scoringbench.multivariate.convergence`` as
# DEFAULT_* constants — they are properties of the diagnostic algorithm, not
# of the benchmark.  Only the on/off switch belongs here.
CONV_ENABLED = True          # set False to skip adaptive top-up in wrappers

# ---------------------------------------------------------------------------
# Shared baseline-wrapper settings (both sources)
# ---------------------------------------------------------------------------

# Number of random chain permutations the *chained* baseline averages over.
# The chain-rule factorization is exact for any order, but with imperfect
# conditional models the sampled joint is order-dependent (exposure bias
# compounds down the chain).  Averaging over a few orders desensitises the
# estimate at a proportional fit-cost increase (CHAINED_N_ORDERS × d models).
CHAINED_N_ORDERS = 3

# Tiny uniform jitter added to the copula PIT pseudo-observations before the
# vine is fit, to break ties introduced by TabPFN's piecewise bar CDF and any
# degenerate constant rows.  0 disables jitter.
COPULA_PIT_JITTER = 1e-4


# ---------------------------------------------------------------------------
# Source 2: synthetic (nonlinear means with randomized R-vine residuals)
# ---------------------------------------------------------------------------
SYNTHETIC_N_FEATURES = 20
SYNTHETIC_DEPENDENCE_TYPES = ("tail", "no_tail", "mixed")
# Keep only the strong regime in the benchmark suite.
SYNTHETIC_DEPENDENCE_STRENGTHS = ("strong",)
SYNTHETIC_DECAY = 0.8
SYNTHETIC_TAU_UPPER = 1.0
SYNTHETIC_N_DATASETS = 100
SYNTHETIC_DATA_SUBDIR = "datasets/synthetic"
