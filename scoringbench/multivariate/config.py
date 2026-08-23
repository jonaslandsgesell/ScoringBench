"""Benchmark-wide configuration for the multivariate ScoringBench.

Mirrors ``scoringbench.univariate.config`` and adds two multivariate-specific
knobs: ``TARGET_DIM`` (the target dimension ``d``) and ``N_DRAWS`` (the number
of Monte-Carlo draws every model must produce per test instance).
"""

# ---------------------------------------------------------------------------
# Shared CV constants (mirrors univariate)
# ---------------------------------------------------------------------------
SEED = 42
N_FOLDS = 5
N_REPEATS_CV = 1
SAMPLE_SIZE = 3000

# ---------------------------------------------------------------------------
# Multivariate-specific constants
# ---------------------------------------------------------------------------

# Target dimension d.  Source 1 datasets promote the (d-1) most-correlated
# feature columns to targets, producing a d-dimensional target vector Y.
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
# Baseline-wrapper robustness knobs
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
# Feature-promotion residualizer (dataset construction)
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
# Synthetic source (Source 2: explicit copula-coupled dependent targets)
# ---------------------------------------------------------------------------
# The synthetic source constructs a d-dimensional regression problem whose
# targets share dependence a product-of-marginals (independent) model CANNOT
# represent: each target is a smooth function of the features plus a residual,
# and the residuals across targets are coupled by an EXPLICITLY-constructed vine
# copula (pyvinecopulib). Because the copula acts on the residuals (i.e. the
# conditional-on-X part), the dependence survives conditioning on X — exactly
# the structure copula / chained models can recover but independent models miss.

# Number of feature columns X in a synthetic dataset. Must be >= 1 so models
# always get a non-empty design matrix.
SYNTHETIC_N_FEATURES = 5

# Std-dev of the (copula-coupled) residual added to each target's conditional
# mean f_k(X). Sets the signal-to-noise ratio: the copula dependence lives
# entirely in this residual block, so a larger scale makes the multivariate
# structure DOMINATE the X-explained mean. This is deliberately large relative
# to SYNTHETIC_MEAN_SCALE so that a model which ignores the cross-target
# dependence (an independent / product-of-marginals model) is badly
# misspecified: the bulk of the joint's information lives in the residual
# copula it cannot represent.
SYNTHETIC_NOISE_SCALE = 3.0

# Amplitude of the smooth conditional mean f_k(X) = mean_scale * tanh(X @ W).
# Kept small relative to SYNTHETIC_NOISE_SCALE so the dependence (not the mean)
# is the dominant source of variance -> independence fails hard.
SYNTHETIC_MEAN_SCALE = 0.3

# Kendall's tau of the pairwise copulas used to couple the residuals. Higher =
# stronger residual dependence = larger independent-vs-copula score gap.
# Two near-comonotone strengths are enumerated; even the weaker one (0.7) leaves
# an independent model far behind, while 0.9 is near-deterministic coupling.
SYNTHETIC_TAUS = (0.7, 0.9)

# Copula families enumerated (one vine per family; all pair-copulas in a vine
# share the family and the tau above). Names must be pyvinecopulib BicopFamily
# members. Asymmetric families (clayton/gumbel/joe) give tail dependence that
# Gaussian-copula fits cannot fully capture, further separating the models.
SYNTHETIC_FAMILIES = ("gaussian", "clayton", "gumbel", "frank")

# Total number of synthetic datasets generated per (target_dim, sample_size)
# shape. The 4 families x 2 taus = 8 cells are populated with independently
# seeded REPLICATES; this many datasets are distributed across the 8 cells as
# evenly as possible (cells differ by at most one replicate). Each replicate is
# a distinct random DGP instance, so more datasets = tighter leaderboard error
# bars and broader coverage of the copula families / dependence strengths.
SYNTHETIC_N_DATASETS = 100

# Root directory holding the FROZEN synthetic artifacts (committed). Artifacts
# are scoped by shape into per-(d, n) subfolders ``datasets/synthetic/d{d}_n{n}/``
# (each with its own manifest.json). Loading reads those exact bytes so results
# are reproducible across numpy / pyvinecopulib versions and NEVER regenerates on
# the fly — a missing shape raises FileNotFoundError pointing at the generator.
# Relative to the multivariate package's parent (the repo's scoringbench/..).
SYNTHETIC_DATA_SUBDIR = "datasets/synthetic"
