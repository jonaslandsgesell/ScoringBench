"""ForestDiffusion wrapper for ScoringBench.

Conditional generative regression via **Forest-Diffusion / Forest-Flow**: an
XGBoost-based diffusion (or flow-matching) model over tabular data. A genuinely
sample-based model — ``fit`` trains ``p * n_t`` gradient-boosted trees that
parameterise the score / vector field, and each :meth:`_draw_samples` call
integrates the reverse ODE/SDE to draw the target ``y`` conditional on the test
features, which the :class:`SampleBasedWrapper` base converts into the standard
PMF.

We model the single target variable ``y`` as the diffused data and pass the
features ``X`` as conditioning covariates (``X_covs``, available in
ForestDiffusion >= 1.0.6). ``generate(batch_size=n_test, X_covs=X_test)`` then
returns one conditional draw of ``y`` per test row; we call it repeatedly to
accumulate the required number of samples per instance.

Reference
---------
Alexia Jolicoeur-Martineau, Kilian Fatras, Tal Kachman.
"Generating and Imputing Tabular Data via Diffusion and Flow-based
Gradient-Boosted Trees." AISTATS 2024. arXiv:2309.09968.
Code: https://github.com/SamsungSAILMontreal/ForestDiffusion (MIT-style license).
"""

from __future__ import annotations

import numpy as np

from .sample_based import SampleBasedWrapper


class ForestDiffusionWrapper(SampleBasedWrapper):
    """Forest-Diffusion / Forest-Flow conditional generative regressor.

    The target ``y`` is treated as the single diffused variable and the feature
    matrix ``X`` is supplied as conditioning covariates (``X_covs``). At sample
    time we integrate the learned reverse process once per requested draw,
    yielding conditional draws of ``y`` for every test row.

    Parameters
    ----------
    n_t : int
        Number of noise levels / discretisation steps. Governs both the number
        of trees trained (``p * n_t``) and the integration resolution. Flow
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
    n_samples : int
        Target number of conditional draws per test instance.
    sample_chunk : int
        Draws requested per call to ``_draw_samples`` (the wall-clock budget is
        checked between chunks).
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
        n_samples: int = 100,
        sample_chunk: int = 10,
        random_state: int = 0,
    ):
        self.n_t = n_t
        self.duplicate_K = duplicate_K
        self.diffusion_type = diffusion_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.N_SAMPLES = int(n_samples)
        self.SAMPLE_CHUNK = int(sample_chunk)

        self._model = None
        self._n_features = None
        self._y_lo = None
        self._y_hi = None

    @staticmethod
    def _sanitize_X(X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return np.nan_to_num(X, nan=0.0, posinf=1e7, neginf=-1e7)

    def fit(self, X, y) -> "ForestDiffusionWrapper":
        from ForestDiffusion import ForestDiffusionModel

        X = self._sanitize_X(X)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        valid = np.isfinite(y)
        X, y = X[valid], y[valid]
        if len(y) == 0:
            raise ValueError("No valid (finite) training samples after sanitization")

        self._n_features = X.shape[1]
        self._y_lo = float(np.min(y))
        self._y_hi = float(np.max(y))

        # Model only the target column, conditioned on the features via X_covs.
        # NOTE: ``p_in_one=False`` is required here. With a single modelled
        # column (c=1) the default ``p_in_one=True`` path uses a multi-output
        # XGBoost predict whose 1-D output cannot broadcast into the internal
        # (b, 1) buffer, raising a shape-mismatch during ``generate``. The
        # per-column path (``p_in_one=False``) trains one regressor per column
        # and assigns column-wise, which is correct for c=1.
        y_col = y.reshape(-1, 1)
        self._model = ForestDiffusionModel(
            y_col,
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
            raise RuntimeError("ForestDiffusionWrapper must be fit before sampling.")

        X = self._sanitize_X(X)
        n_test = X.shape[0]

        # Each generate() call yields exactly one conditional draw of y per row
        # (batch_size must equal the number of conditioning rows). Loop to
        # accumulate the requested number of draws per test instance.
        draws = np.empty((n_test, int(n_samples)), dtype=np.float64)
        for j in range(int(n_samples)):
            gen = np.asarray(
                self._model.generate(batch_size=n_test, X_covs=X),
                dtype=np.float64,
            )
            # gen is (n_test, 1) since we model a single variable and pass no
            # label_y; take the first (only) modelled column as the y draw.
            draws[:, j] = gen[:, 0]

        return draws
