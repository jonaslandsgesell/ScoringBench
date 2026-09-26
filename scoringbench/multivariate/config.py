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
# benchmark-wide because the *fair* pairwise estimators used by the scoring
# rules still have a finite-sample bias/variance that depends on the number of
# draws m; fixing m across all models keeps the comparison apples-to-apples.
# (The energy-score term-2 estimator 1/(m(m-1)) Σ_{i≠j} is unbiased for every
# m ≥ 2, but its variance — and the variogram/DSS moment estimates — still
# shrink with m, so a shared m is required for a fair leaderboard.)
N_DRAWS = 300

# ---------------------------------------------------------------------------
# Shared baseline-wrapper settings (both sources)
# ---------------------------------------------------------------------------

# Number of random chain permutations the *chained* baseline averages over.
# The chain-rule factorization is exact for any order, but with imperfect
# conditional models the sampled joint is order-dependent (exposure bias
# compounds down the chain).  Averaging over a few orders desensitises the
# estimate at a proportional fit-cost increase (CHAINED_N_ORDERS × d models).
# Set to 1 to recover the classic single fixed-order chain.
CHAINED_N_ORDERS = 3

# Tiny uniform jitter added to the copula PIT pseudo-observations before the
# vine is fit, to break ties introduced by TabPFN's piecewise bar CDF and any
# degenerate constant rows.  0 disables jitter.
COPULA_PIT_JITTER = 1e-4

# ---------------------------------------------------------------------------
# Source 1: scoringbench (real-data feature promotion)
# ---------------------------------------------------------------------------

# When promoting feature columns to target dimensions we residualise each
# candidate target against the *remaining* features and measure the residual
# cross-target Spearman dependence.  A plain OLS residualizer only removes the
# *linear* conditional mean, so any nonlinear signal in X leaks into the
# residuals and is misread as cross-target dependence.  A small, fast gradient-
# boosted-tree regressor (XGBoost) captures nonlinear conditional means, leaving
# cleaner residuals whose remaining Spearman correlation reflects genuine
# residual dependence.  These knobs keep the O(p^2 * n_promote) inner fits cheap;
# residual outputs are cached within a selection run so repeated (target,
# feature-set) combinations are only fit once.
RESIDUALIZER_N_ESTIMATORS = 100
RESIDUALIZER_MAX_DEPTH = 4
RESIDUALIZER_LEARNING_RATE = 0.3
RESIDUALIZER_SUBSAMPLE = 1.0
# RESIDUALIZER_N_JOBS is intentionally not used: the residualizer is called in
# a tight greedy loop and nthread=1 per fit is faster than spawning a full
# thread pool for each small fit (benchmarked ~40x speedup on this machine).
# Parallelism is available at the outer CV / dataset level instead.

# Maximum number of rows used to *fit* the residualizer (predict is always on
# all rows so Spearman ranks remain representative).  Reduces cost for large
# datasets without biasing the rank-correlation criterion.
RESIDUALIZER_MAX_ROWS = 2000

# Feature-count threshold above which the residualizer falls back from XGBoost
# to plain OLS.  In the greedy loop the pre-screening step (below) already
# limits the candidate pool, so X_rest is typically small; this threshold is a
# safety net for datasets that are very wide even after screening.
RESIDUALIZER_LINEAR_FALLBACK_THRESHOLD = 50

# Size of the candidate pool fed into the expensive greedy XGBoost loop.
# Columns are pre-screened by marginal Spearman with y; only the top-k survive.
# Set to a very large number to disable pre-screening.
RESIDUALIZER_PRESCREENING_KEEP = 20  # overridden to max(20, 3*n_promote) if larger

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
