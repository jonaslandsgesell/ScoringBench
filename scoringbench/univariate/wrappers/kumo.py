"""KumoTabular (NVIDIA structured-data-models / SDM) wrappers for ScoringBench.

This wrapper ports the official NVIDIA / Kumo ScoringBench adapter
(`benchmark/tabular/scoringbench/models.py`, Apache-2.0) into ScoringBench's
own wrapper conventions, retaining as much of the developers' logic as
possible:

* The model is one of SDM's in-context learning models,
  ``sdm.models.KumoTabular(task="regression", size=..., device=...)``.
* Features are inferred with ``sdm.infer_stypes(X)`` and fed as an SDM
  ``TableTensor``; the target is a single ``"numerical"`` column.
* ``fit`` runs an in-context ensemble of ``num_estimators`` estimators under
  ``torch.amp.autocast`` (float16 on CUDA) with a seeded ``torch.Generator``.
* ``predict`` returns the 999-quantile bank at
  ``linspace(0.001, 0.999, 999)``; those quantiles are fed through the shared
  :func:`quantiles_to_distribution` mapping (identical to the TabICL /
  CatBoost / XGB-quantile / NGBoost / EXAONE wrappers), so every quantile
  model is discretized the same way.

Three sizes are exposed, mirroring the upstream benchmark's ``num_estimators``
choices:

* ``KumoTabularSmallWrapper``  — size="small",  num_estimators=8
* ``KumoTabularMediumWrapper`` — size="medium", num_estimators=8
* ``KumoTabularLargeWrapper``  — size="large",  num_estimators=16 (upstream default)

Setup
-----
Install ``structured-data-models`` (provides the ``sdm`` package)::

    pip install "sdm @ git+https://github.com/NVIDIA/structured-data-models.git"

"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from .base import DistributionPrediction, ProbabilisticWrapper
from .quantile_based import quantiles_to_distribution

# Quantile levels the KumoTabular regression head is queried at, verbatim from
# the upstream NVIDIA/SDM ScoringBench adapter.
_QUANTILE_LEVELS = np.linspace(0.001, 0.999, 999)


def _as_named_frame(X) -> pd.DataFrame:
    """Return ``X`` as a DataFrame with string column names.

    ScoringBench hands wrappers real DataFrames (string columns), but the test
    suite and other callers may pass raw arrays.  ``sdm.infer_stypes`` /
    ``TableTensor.from_pandas`` assert that column names are strings, so a
    bare ``pd.DataFrame(array)`` (integer labels ``0, 1, ...``) would fail.
    Coerce non-string labels to ``"f{i}"`` while leaving existing string names
    untouched.
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(np.asarray(X))
    if not all(isinstance(c, str) for c in X.columns):
        X = X.rename(columns={c: f"f{i}" for i, c in enumerate(X.columns)})
    return X


class KumoTabularWrapper(ProbabilisticWrapper):
    """Wraps ``sdm.models.KumoTabular`` (regression) for ScoringBench.

    Parameters
    ----------
    size : {"small", "medium", "large"}
        KumoTabular model size passed to ``sdm.models.KumoTabular``.
    num_estimators : int
        Number of in-context ensemble estimators used at ``fit`` time. The
        upstream benchmark uses 8 for small/medium and 16 for large.
    device : str or torch.device, optional
        Torch device. Defaults to ``"cuda"`` when available, else ``"cpu"``.
    seed : int
        Seed for the ``torch.Generator`` driving the in-context ensemble.
    batch_size : int, optional
        Query rows per forward pass. ``None`` predicts all rows at once.
    autocast_dtype : torch.dtype, optional
        Autocast dtype used on CUDA (upstream uses ``torch.float16``).
    """

    def __init__(
        self,
        *,
        size: Literal["small", "medium", "large"] = "large",
        num_estimators: int | None = None,
        device: "str | object | None" = None,
        seed: int = 42,
        batch_size: int | None = None,
        autocast_dtype: "object | None" = None,
    ) -> None:
        import torch

        import sdm  # noqa: F401  (validate the dependency is importable early)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.size = size
        # Match the upstream num_estimators policy when not overridden:
        # 16 for the large model, 8 otherwise.
        if num_estimators is None:
            num_estimators = 16 if size == "large" else 8
        self.num_estimators = int(num_estimators)

        self.seed = int(seed)
        self.batch_size = batch_size
        self.autocast_dtype = (
            autocast_dtype if autocast_dtype is not None else torch.float16
        )

        self.model = sdm.models.KumoTabular(
            task="regression", size=size, device=self.device
        )
        self.stypes: dict | None = None

    # ------------------------------------------------------------------
    def fit(self, X, y) -> "KumoTabularWrapper":
        import torch

        import sdm

        X = _as_named_frame(X)
        y = y if isinstance(y, pd.Series) else pd.Series(np.asarray(y).reshape(-1))

        self._set_train_range(y)
        self.stypes = sdm.infer_stypes(X)

        x_context = sdm.TableTensor.from_pandas(
            df=X,
            stypes=self.stypes,
            device=self.device,
        )
        target_name = str(y.name) if y.name is not None else "__target__"
        y_context = sdm.TableTensor.from_pandas(
            df=y.rename(target_name).to_frame(),
            stypes={target_name: "numerical"},
            device=self.device,
        )

        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        with torch.amp.autocast(
            self.device.type,
            self.autocast_dtype,
            enabled=x_context.is_cuda,
        ):
            self.model.fit(
                x=x_context,
                y=y_context,
                num_estimators=self.num_estimators,
                generator=generator,
            )

        return self

    # ------------------------------------------------------------------
    def _predict_quantiles(self, X) -> np.ndarray:
        """Return the raw (n_rows, 999) quantile bank for ``X``."""
        import torch

        import sdm

        if self.stypes is None:
            raise RuntimeError("KumoTabularWrapper.fit must be called before predict")

        X = _as_named_frame(X)

        x_query = sdm.TableTensor.from_pandas(
            df=X,
            stypes=self.stypes,
            device=self.device,
        )

        outs = []
        for batch in x_query.split(self.batch_size or len(x_query), dim=-2):
            with torch.amp.autocast(
                self.device.type,
                self.autocast_dtype,
                enabled=x_query.is_cuda,
            ):
                outs.append(self.model.predict(x=batch).numerical)

        return torch.cat(outs, dim=-2).cpu().numpy()

    # ------------------------------------------------------------------
    def predict_distribution(self, X) -> DistributionPrediction:
        assert self._y_train_range is not None
        q = self._predict_quantiles(X)
        return quantiles_to_distribution(
            q,
            _QUANTILE_LEVELS,
            train_range=self._y_train_range,
        )

    def predict(self, X) -> np.ndarray:
        return self.predict_distribution(X).mean


class KumoTabularSmallWrapper(KumoTabularWrapper):
    """KumoTabular (size="small", num_estimators=8)."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("size", "small")
        kwargs.setdefault("num_estimators", 8)
        super().__init__(**kwargs)


class KumoTabularMediumWrapper(KumoTabularWrapper):
    """KumoTabular (size="medium", num_estimators=8)."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("size", "medium")
        kwargs.setdefault("num_estimators", 8)
        super().__init__(**kwargs)


class KumoTabularLargeWrapper(KumoTabularWrapper):
    """KumoTabular (size="large", num_estimators=16) — upstream default."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("size", "large")
        kwargs.setdefault("num_estimators", 16)
        super().__init__(**kwargs)
