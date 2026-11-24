"""CausalPFN univariate regression wrapper for ScoringBench.

CausalPFN (Balazadeh, Kamkari, Thomas, Li, Ma, Cresswell, Krishnan, 2025 --
*CausalPFN: Amortized Causal Effect Estimation via In-Context Learning*,
https://arxiv.org/abs/2506.07918; code https://github.com/vdblm/CausalPFN,
Apache-2.0) is a TabDPT-style in-context transformer that, given a context of
``(x, t, y)`` rows, predicts a **histogram over the conditional expected
potential outcome** ``E[Y(t) | x]`` for query rows: 1024 uniform bins on
``[-10, 10]`` in units of the context-standardized outcome.

Regression mapping
------------------
Every row (context and query) is fed with treatment ``t = 0``, i.e. as a
single-arm observational study, so ``E[Y(0) | x] = E[Y | x]``.  The standardizer
is then the plain context mean / std of ``y``, the bin grid is SHARED by all
query rows, and the model's softmax over bins is used verbatim as the
predictive PMF (grid-native, like TabDPT / TabPFN bar distributions).

Temperature
-----------
The head was trained with HL-Gauss targets (a Gaussian of width ``sigma=0.5``
standardized units around ``E[Y(t)|x]``), so at temperature 1 its histogram is
the label-smoothing kernel (``sd ~ 0.5 * sd(y)``) and NOT a predictive
distribution of ``Y``.  CausalPFN ships a temperature-calibration step against
observed outcomes for exactly this reason (``calibrate=True``, n-fold CV on the
training data).  We follow it, but select the temperature by the held-out
**log score** of the observed training targets (exact on the histogram) rather
than the package's sampled integrated-coverage error.  ``temperature`` may also
be fixed to a float (``1.0`` = raw head).
"""

from __future__ import annotations

import numpy as np

from .base import DistributionPrediction, ProbabilisticWrapper


_MODEL_CACHE: dict = {}


def _load_icl_model(device: str, model_path: str):
    """Load (and cache per device/path) CausalPFN's ``InContextModel``."""
    key = (device, model_path)
    if key not in _MODEL_CACHE:
        from causalpfn import CATEEstimator

        est = CATEEstimator(device=device, model_path=model_path)
        est.load_model()
        est.icl_model.eval()
        _MODEL_CACHE[key] = est.icl_model
    return _MODEL_CACHE[key]


class CausalPFNWrapper(ProbabilisticWrapper):
    """Wraps CausalPFN's in-context model as a probabilistic regressor.

    Parameters
    ----------
    device : str | None
        Torch device; ``None`` auto-selects CUDA when available.
    temperature : "auto" | float
        ``"auto"`` calibrates the softmax temperature by ``n_folds``-fold
        in-context CV log score on the training data; a float fixes it.
    n_folds : int
        Folds for the temperature calibration.
    temperature_grid : array-like | None
        Candidate temperatures (default: 200 log-spaced values in [1e-3, 10],
        the package's own calibration range).
    max_context_length : int
        Maximum context rows (the package default, 4096); larger training sets
        are randomly subsampled once at fit time.
    query_batch_size : int
        Query rows per forward pass (queries attend only to the context).
    model_path : str
        Checkpoint (Hugging Face repo id or local path).
    seed : int
        Seed for context subsampling and the calibration fold split.
    """

    def __init__(self, device: str | None = None, temperature="auto", n_folds: int = 3,
                 temperature_grid=None, max_context_length: int = 4096,
                 query_batch_size: int = 1024, model_path: str = "vdblm/causalpfn",
                 seed: int = 0):
        import torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.temperature = temperature
        self.n_folds = int(n_folds)
        self.temperature_grid = (np.logspace(-3, 1, 200) if temperature_grid is None
                                 else np.asarray(temperature_grid, dtype=np.float64))
        self.max_context_length = int(max_context_length)
        self.query_batch_size = int(query_batch_size)
        self.model_path = model_path
        self.seed = int(seed)
        self._model = _load_icl_model(device, model_path)
        self.temperature_ = None

    # -- preprocessing ------------------------------------------------------
    @staticmethod
    def _to_float(X) -> np.ndarray:
        X = np.asarray(X.values if hasattr(X, "values") else X, dtype=np.float64)
        if X.ndim == 1:
            X = X[:, None]
        # The model's normalizer masks NaN and zero-fills it afterwards.
        return np.where(np.isfinite(X), X, np.nan)

    def _transform_X(self, X) -> np.ndarray:
        X = self._to_float(X)
        if self._svd is not None:
            X = self._svd.transform(np.nan_to_num(X - self._x_fill_center, nan=0.0))
        return X

    # -- core forward -------------------------------------------------------
    def _logits(self, X_ctx, y_ctx_std, X_q):
        """Raw (temperature-1) logits of shape (n_query, n_bins) as a float32 tensor."""
        import torch

        m = self._model
        dev = self.device
        xc = torch.as_tensor(X_ctx, dtype=torch.float32, device=dev)
        yc = torch.as_tensor(y_ctx_std, dtype=torch.float32, device=dev)
        ctx = torch.cat([torch.zeros(xc.shape[0], 1, device=dev), xc], dim=1)
        out = []
        with torch.no_grad():
            for s in range(0, X_q.shape[0], self.query_batch_size):
                xq = torch.as_tensor(X_q[s:s + self.query_batch_size], dtype=torch.float32, device=dev)
                q = torch.cat([torch.zeros(xq.shape[0], 1, device=dev), xq], dim=1)
                x_src, y_src = m.prepare_input(torch.cat([ctx, q], dim=0)[None], yc[None])
                lg = m.model(x_src.transpose(0, 1), y_src.transpose(0, 1)).transpose(0, 1)
                out.append(lg[0, :, -m.model.nbins:].float())
        return torch.cat(out, dim=0)

    @staticmethod
    def _standardize(y):
        mu = float(np.mean(y))
        sd = float(np.std(y)) + 1e-20  # population std, as CausalPFN does
        return mu, sd

    # -- temperature calibration -------------------------------------------
    def _calibrate_temperature(self, X, y, rng) -> float:
        import torch

        edges_std = self._model.bin_edges.detach().cpu().numpy().astype(np.float64)
        width_std = edges_std[1] - edges_std[0]
        temps = torch.as_tensor(self.temperature_grid, dtype=torch.float32, device=self.device)
        nll = torch.zeros_like(temps)
        folds = np.array_split(rng.permutation(len(y)), self.n_folds)
        for val in folds:
            trn = np.setdiff1d(np.arange(len(y)), val)
            mu, sd = self._standardize(y[trn])
            lg = self._logits(X[trn], (y[trn] - mu) / sd, X[val])          # (n_val, B)
            z = (y[val] - mu) / sd
            idx = np.clip(np.searchsorted(edges_std, z, side="right") - 1, 0, lg.shape[1] - 1)
            idx_t = torch.as_tensor(idx, device=self.device)
            for i in range(0, len(temps), 20):
                tb = temps[i:i + 20]
                logp = torch.log_softmax(lg[None] / tb[:, None, None], dim=-1)   # (k, n_val, B)
                lp = logp.gather(-1, idx_t[None, :, None].expand(len(tb), -1, 1))[..., 0]
                # density in raw units: p_k / (width_std * sd)
                nll[i:i + 20] += -(lp.sum(dim=1) - len(val) * np.log(width_std * sd))
        return float(self.temperature_grid[int(torch.argmin(nll).item())])

    # -- public API ---------------------------------------------------------
    def fit(self, X, y) -> "CausalPFNWrapper":
        X = self._to_float(X)
        y = np.asarray(y.values if hasattr(y, "values") else y, dtype=np.float64).reshape(-1)
        finite = np.isfinite(y)
        if not finite.any():
            raise ValueError("CausalPFN requires at least one finite training target")
        X, y = X[finite], y[finite]
        self._set_train_range(y)

        rng = np.random.default_rng(self.seed)
        # One feature slot is the treatment column; reduce wider tables with
        # TruncatedSVD exactly as the package does.
        max_feat = self._model.model.num_features - 1
        self._svd = None
        self._x_fill_center = None
        if X.shape[1] > max_feat:
            from sklearn.decomposition import TruncatedSVD

            self._x_fill_center = np.nanmean(X, axis=0)
            self._x_fill_center = np.where(np.isfinite(self._x_fill_center), self._x_fill_center, 0.0)
            self._svd = TruncatedSVD(n_components=max_feat, algorithm="arpack", random_state=self.seed)
            X = self._svd.fit_transform(np.nan_to_num(X - self._x_fill_center, nan=0.0))

        if len(y) > self.max_context_length:
            keep = np.sort(rng.choice(len(y), self.max_context_length, replace=False))
            X, y = X[keep], y[keep]
        self._X_ctx, self._y_ctx = X, y
        self._mu, self._sd = self._standardize(y)

        if isinstance(self.temperature, str):
            if self.temperature != "auto":
                raise ValueError(f"temperature must be 'auto' or a float; got {self.temperature!r}")
            self.temperature_ = self._calibrate_temperature(X, y, rng) if len(y) >= 2 * self.n_folds else 1.0
        else:
            self.temperature_ = float(self.temperature)
        return self

    def predict_distribution(self, X) -> DistributionPrediction:
        import torch

        Xq = self._transform_X(X)
        lg = self._logits(self._X_ctx, (self._y_ctx - self._mu) / self._sd, Xq)
        probas = torch.softmax(lg / self.temperature_, dim=-1).double().cpu().numpy()
        probas /= probas.sum(axis=1, keepdims=True)
        bin_edges = self._model.bin_edges.detach().cpu().numpy().astype(np.float64) * self._sd + self._mu
        return DistributionPrediction.from_histogram(
            bin_edges, probas, train_range=self._y_train_range, is_grid_native=True,
        )

    def predict(self, X) -> np.ndarray:
        return np.asarray(self.predict_distribution(X).mean, dtype=np.float64).reshape(-1)
