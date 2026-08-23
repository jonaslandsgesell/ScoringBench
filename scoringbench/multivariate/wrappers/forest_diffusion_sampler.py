"""Forest-Diffusion multivariate wrapper for the multivariate ScoringBench.

Conditional generative regression via **Forest-Diffusion / Forest-Flow**: an
XGBoost-based diffusion (or flow-matching) model over tabular data.  Unlike the
``independent`` / ``copula`` / ``chained`` compositions in ``baselines.py`` —
which glue *per-dimension univariate* samplers — this model is a *native*
multivariate sampler: it diffuses the whole ``d``-dimensional target vector
``Y`` at once, conditioned on the features ``X`` (passed as ``X_covs``), so the
learned cross-target dependence comes directly from the joint diffusion.  No
chained or copula variant is provided; there is exactly one direct multivariate
model.

The target block ``Y`` (shape ``(n, d)``) is the diffused data and the features
``X`` are conditioning covariates (``X_covs``, available in
ForestDiffusion >= 1.0.6).  ``generate(batch_size=n_test, X_covs=X_test)`` then
returns one joint conditional draw of ``Y`` per test row; we call it repeatedly
to accumulate the ``m`` draws every model must emit.

Reference
---------
Alexia Jolicoeur-Martineau, Kilian Fatras, Tal Kachman.
"Generating and Imputing Tabular Data via Diffusion and Flow-based
Gradient-Boosted Trees." AISTATS 2024. arXiv:2309.09968.
Code: https://github.com/SamsungSAILMontreal/ForestDiffusion (MIT-style license).
"""

from __future__ import annotations

import numpy as np

from ..config import N_DRAWS
from .base import as_2d_features, as_2d_targets
from .sample_based import SampleBasedWrapper


class ForestDiffusionMultiOutputWrapper(SampleBasedWrapper):
    """Native multivariate Forest-Diffusion / Forest-Flow conditional regressor.

    The full target block ``Y`` (all ``d`` columns) is diffused jointly and the
    feature matrix ``X`` is supplied as conditioning covariates (``X_covs``).
    At sample time we integrate the learned reverse process once per requested
    draw, yielding a joint conditional draw of ``Y`` for every test row.  The
    :class:`SampleBasedWrapper` base accumulates draws in chunks under a
    wall-clock budget and packages them into a
    :class:`~..prediction.MultivariateSamplePrediction` of shape
    ``(n_test, m, d)``.

    Parameters
    ----------
    n_t : int
        Number of noise levels / discretisation steps.  Governs both the number
        of trees trained (``d * n_t``) and the integration resolution.  Flow
        matching works well with modest values (10-30); ``vp`` diffusion tends
        to prefer larger values.
    duplicate_K : int
        Number of noise realisations per training row (data is tiled
        ``duplicate_K`` times to give the trees more signal).
    diffusion_type : {"vp", "flow"}
        ``"flow"`` (flow-matching ODE) is generally faster and higher quality;
        ``"vp"`` (variance-preserving SDE) is the diffusion variant.
    n_estimators : int
        Number of boosting rounds per XGBoost regressor.
    max_depth : int
        Maximum tree depth for each XGBoost regressor.
    n_jobs : int
        CPUs used by the parallel tree training (``-1`` = all cores).
    n_draws : int
        Target number of joint conditional draws per test instance (defaults to
        the benchmark-wide ``config.N_DRAWS`` so every model emits the same m).
    sample_chunk : int
        Draws requested per call to :meth:`_draw_samples` (the wall-clock budget
        is checked between chunks).
    random_state : int
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_t: int = 25,
        duplicate_K: int = 100,
        diffusion_type: str = "flow",
        n_estimators: int = 100,
        max_depth: int = 7,
        n_jobs: int = -1,
        n_draws: int = int(N_DRAWS),
        sample_chunk: int = 10,
        random_state: int = 0,
    ):
        self.n_t = int(n_t)
        self.duplicate_K = int(duplicate_K)
        self.diffusion_type = str(diffusion_type)
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.n_jobs = int(n_jobs)
        self.random_state = int(random_state)

        self.N_SAMPLES = int(n_draws)
        self.SAMPLE_CHUNK = int(sample_chunk)

        self._model = None
        self._n_features: int | None = None
        self._d: int | None = None
        # Running count of draws already produced in the current
        # ``predict_ensemble`` call.  The sample-based base accumulates draws by
        # calling ``_draw_samples`` once per chunk; we offset the per-chunk RNG
        # seed by this counter so consecutive chunks draw *distinct* noise
        # instead of repeating the same block (which would collapse the ensemble
        # to ``SAMPLE_CHUNK`` unique values).  Reset in ``_collect_samples``.
        self._draw_offset: int = 0

    @staticmethod
    def _sanitize_X(X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return np.nan_to_num(X, nan=0.0, posinf=1e7, neginf=-1e7)

    def fit(self, X, Y) -> "ForestDiffusionMultiOutputWrapper":
        from ForestDiffusion import ForestDiffusionModel

        X = self._sanitize_X(as_2d_features(X))
        Y = as_2d_targets(Y)

        # Drop rows with any non-finite target coordinate (a partially observed
        # target vector cannot seed the joint diffusion).
        valid = np.isfinite(Y).all(axis=1)
        X, Y = X[valid], Y[valid]
        if len(Y) == 0:
            raise ValueError("No valid (finite) training samples after sanitization")

        self._n_features = X.shape[1]
        self._d = Y.shape[1]

        # Model the full d-column target block, conditioned on the features via
        # X_covs.  ``p_in_one=False`` trains one regressor per modelled column
        # and assigns column-wise; this is the robust path for both c=1 and the
        # multi-column (c=d) case (the ``p_in_one=True`` fast path has a
        # broadcasting bug for single-column c=1, see the univariate wrapper).
        self._model = ForestDiffusionModel(
            Y,
            X_covs=X,
            label_y=None,
            n_t=self.n_t,
            duplicate_K=self.duplicate_K,
            diffusion_type=self.diffusion_type,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            bin_indexes=[],
            cat_indexes=[],
            int_indexes=[],
            p_in_one=False,
            n_jobs=self.n_jobs,
            seed=self.random_state,
        )
        return self

    def _draw_samples(self, X, n_samples: int) -> np.ndarray:
        if self._model is None:
            raise RuntimeError(
                "ForestDiffusionMultiOutputWrapper must be fit before sampling."
            )

        X = self._sanitize_X(as_2d_features(X))
        n_test = X.shape[0]
        d = int(self._d)

        # ForestDiffusion draws its prior noise from the *global* ``np.random``
        # state (``generate`` calls ``np.random.normal`` with no internal
        # re-seed), and ``fit`` pins that global state via ``np.random.seed``.
        # Two independent risks follow, both of which we neutralise here:
        #
        #   1. Determinism would otherwise leak through global state — any other
        #      code touching ``np.random`` between fit and here (or between the
        #      chunked calls of the sample-based base) would silently shift the
        #      draw sequence.  We snapshot the legacy global RNG, seed it from
        #      the wrapper's own seed for the duration of the loop, then restore
        #      it, so sampling is reproducible and self-contained.
        #   2. Draw *collapse* — if the loop reused the same noise every call the
        #      draws would be near-identical (a degenerate predictive law).
        #      Because each ``generate`` consumes fresh noise from the advancing
        #      state, and we seed once *before* the loop (not per iteration),
        #      every draw uses a distinct noise vector.  Empirically all draws
        #      are distinct with the expected predictive spread and cross-target
        #      correlation (see ``tests/multivariate/test_forest_diffusion.py``).
        draws = np.empty((n_test, int(n_samples), d), dtype=np.float64)
        prev_state = np.random.get_state()
        try:
            # Seed offset by the running draw count so each chunk continues a
            # distinct noise sequence rather than restarting from the same seed
            # (which would repeat identical blocks of SAMPLE_CHUNK draws).
            np.random.seed(self.random_state + 1 + self._draw_offset)
            for j in range(int(n_samples)):
                gen = np.asarray(
                    self._model.generate(batch_size=n_test, X_covs=X),
                    dtype=np.float64,
                )
                # gen is (n_test, d) — the d modelled target columns; no label_y.
                if gen.ndim == 1:
                    gen = gen[:, None]
                draws[:, j, :] = gen[:, :d]
        finally:
            np.random.set_state(prev_state)

        self._draw_offset += int(n_samples)
        return draws

    def _collect_samples(self, X) -> np.ndarray:
        # Reset the per-call draw offset so every ``predict_ensemble`` call is
        # reproducible (starts from the same seed) yet its internal chunks each
        # advance to distinct noise.  See ``_draw_samples``.
        self._draw_offset = 0
        return super()._collect_samples(X)
