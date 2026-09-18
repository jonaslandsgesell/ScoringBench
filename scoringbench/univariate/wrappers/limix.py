"""LimiX-2's regression PMF, not its mean-valued feature imputations.

Default ensemble members share one standardized target grid. Their softmax
probabilities can therefore be averaged directly, unlike Mitra's different-grid
children. Keep the finite-bin CDF used by LimiX's ensemble decoder verbatim;
the point prediction retains the library's endpoint half-normal correction.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
from threading import RLock
import typing
from unittest.mock import patch

import numpy as np

from .base import DistributionPrediction, ProbabilisticWrapper


_LIMIX_ROOT = Path(__file__).resolve().parents[3] / "additional_models" / "LimiX"
_IMPORT_ROOTS = {"inference", "model", "utils"}
_BACKEND_MODULES = {}
_IMPORT_LOCK = RLock()
_CHECKPOINT_REVISION = "4003e952982bdc4dda3cceed95ee35f9e6b835a3"


def _backend_modules():
    return {
        name: module for name, module in tuple(sys.modules.items())
        if name.split(".", 1)[0] in _IMPORT_ROOTS
    }


@contextmanager
def _limix_imports():
    """Isolate the vendor's unqualified model/utils/inference import names."""
    with _IMPORT_LOCK:
        previous = _backend_modules()
        previous_path = sys.path[:]
        for name in previous:
            del sys.modules[name]
        sys.modules.update(_BACKEND_MODULES)
        sys.path.insert(0, str(_LIMIX_ROOT))
        try:
            yield
        finally:
            loaded = _backend_modules()
            _BACKEND_MODULES.update(loaded)
            for name in loaded:
                del sys.modules[name]
            sys.modules.update(previous)
            sys.path[:] = previous_path


class _CaptureRegressionPMF:
    def get_reg_pred_result(self, inputs, y_train):
        import torch

        borders = self.model._reg_borders.detach().double().cpu().numpy()
        members = [
            torch.softmax(
                member[0].reshape(-1, len(borders) - 1).float()
                / self.softmax_temperature,
                dim=-1,
            )
            for member in inputs
        ]
        probabilities = torch.stack(members).mean(dim=0).double().cpu().numpy()
        if not np.isfinite(probabilities).all():
            raise ValueError("LimiX produced non-finite regression probabilities")
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        self.distribution_probas_ = probabilities
        self.distribution_edges_ = borders * self.y_std + self.y_mean
        return super().get_reg_pred_result(inputs, y_train)


class LimiXWrapper(ProbabilisticWrapper):
    """Expose the LimiX-2 regression head as a grid-native histogram.

    Requires the vendored LimiX source and its runtime dependencies. Weights
    resolve from model_path, LIMIX_CHECKPOINT, or the pinned Hugging Face release.
    Python 3.11 uses typing_extensions.override during backend imports only.
    n_estimators selects the first 1..8 default feature-preprocessing members;
    no target transformation, imputation or artificial sampling is introduced.
    """

    def __init__(
        self,
        model_path=None,
        *,
        device="cuda",
        n_estimators=8,
        random_state=0,
        softmax_temperature=0.9,
        mix_precision=False,
        test_batch_size=16384,
    ):
        if not isinstance(n_estimators, (int, np.integer)) or not 1 <= n_estimators <= 8:
            raise ValueError("n_estimators must be an integer in 1..8")
        if not np.isfinite(softmax_temperature) or softmax_temperature <= 0:
            raise ValueError("softmax_temperature must be finite and positive")
        self.model_path = model_path
        self.device = device
        self.n_estimators = int(n_estimators)
        self.random_state = random_state
        self.softmax_temperature = softmax_temperature
        self.mix_precision = mix_precision
        self.test_batch_size = test_batch_size
        self._predictor = None
        self._cached_features = None
        self._cached_distribution = None

    def _build_predictor(self):
        import torch
        from typing_extensions import override

        config_path = _LIMIX_ROOT / "config" / "reg_default_noretrieval_v2.json"
        config = json.loads(config_path.read_text())
        config["pipelines"] = config["pipelines"][:self.n_estimators]
        if any("TargetTransform" in member for member in config["pipelines"]):
            raise ValueError("LimiX member target transforms need separate PMF grids")
        checkpoint = self.model_path or os.environ.get("LIMIX_CHECKPOINT")
        if checkpoint is None:
            from huggingface_hub import hf_hub_download

            checkpoint = hf_hub_download(
                "stable-ai/LimiX-2", "LimiX-2.ckpt", revision=_CHECKPOINT_REVISION
            )
        self.resolved_model_path_ = str(checkpoint)
        device = self.device
        if device in (None, "auto"):
            device = "cuda" if torch.cuda.is_available() else "cpu"

        with _limix_imports(), patch.object(typing, "override", override, create=True):
            from inference.predictor import get_predictor_version
            from model.version import resolve_arch_line, resolve_arch_version

            checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if resolve_arch_line(resolve_arch_version(checkpoint_data)) != "v2_0":
                raise ValueError("LimiXWrapper requires the LimiX-2 architecture")
            predictor_type = get_predictor_version("2.0")
            cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))

            class DistributionPredictor(_CaptureRegressionPMF, predictor_type):
                class CacheManager(predictor_type.CacheManager):
                    def __init__(self, cache_dir):
                        super().__init__(cache_dir=str(cache_root / "limix"))

            return DistributionPredictor(
                device=torch.device(device),
                model_path=str(checkpoint),
                ckpt=checkpoint_data,
                inference_config=config,
                regression_decoder_type="bucket",
                use_data_cache=False,
                enable_preprocess_parallel=False,
                mix_precision=self.mix_precision,
                softmax_temperature=self.softmax_temperature,
                seed=self.random_state,
                test_batch_size=self.test_batch_size,
            )

    def fit(self, features, targets):
        features = np.asarray(features)
        targets = np.asarray(targets, dtype=float).reshape(-1)
        if features.ndim != 2 or len(features) != len(targets):
            raise ValueError("Expected a feature matrix and one target per row")
        valid = np.isfinite(targets)
        features, targets = features[valid], targets[valid]
        if targets.size < 2 or np.ptp(targets) == 0:
            raise ValueError("LimiX requires at least two distinct finite targets")
        self._set_train_range(targets)
        self._features_train = features.copy()
        self._targets_train = targets.copy()
        self._cached_features = None
        self._cached_distribution = None
        self._predictor = self._build_predictor()
        return self

    def predict(self, features):
        return self.predict_distribution(features).mean.copy()

    def predict_distribution(self, features):
        import pandas as pd
        import torch

        if self._predictor is None:
            raise RuntimeError("fit() must be called before predict_distribution()")
        features = np.asarray(features)
        if features.ndim != 2 or features.shape[1] != self._features_train.shape[1]:
            raise ValueError("Prediction features must match the fitted feature count")
        if not len(features):
            raise ValueError("Prediction requires at least one row")
        if self._cached_features is not None and pd.DataFrame(features).equals(
            pd.DataFrame(self._cached_features)
        ):
            return self._cached_distribution

        with _limix_imports(), torch.inference_mode():
            predictions = self._predictor.predict(
                self._features_train, self._targets_train, features,
                task_type="Regression",
            )
        probabilities = self._predictor.distribution_probas_
        edges = self._predictor.distribution_edges_
        if probabilities.shape[0] != len(features):
            raise ValueError("LimiX PMF rows do not match the query rows")
        if not np.isfinite(edges).all() or not np.all(np.diff(edges) > 0):
            raise ValueError("LimiX returned invalid target bin edges")
        distribution = DistributionPrediction.from_histogram(
            edges, probabilities, mean=np.asarray(predictions, dtype=float).reshape(-1),
            train_range=self._y_train_range, is_grid_native=True,
        )
        self._cached_features = features.copy()
        self._cached_distribution = distribution
        return distribution