"""Continuous-target quality audit and disjoint curation/evaluation."""

import hashlib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from scoringbench.univariate.discreteness import target_discreteness
from .ci_screen import EFFECT_MIN, curation_split, leaky_features, screen_kci


def target_audit(targets, name: str | None = None) -> list[dict]:
    """Conservative numeric diagnostics; not a substitute for semantic review."""
    frame = pd.DataFrame(targets)
    raw_categories = getattr(targets, "attrs", {}).get("categorical_targets", [])
    diagnostics = []
    for column in frame:
        values = frame[column]
        categorical = (bool(raw_categories) or not pd.api.types.is_numeric_dtype(values.dtype)
                       or pd.api.types.is_bool_dtype(values.dtype))
        numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        result = target_discreteness(numeric, name=name)
        result = {key: (None if isinstance(value, float) and not np.isfinite(value) else value)
                  for key, value in result.items()}
        result.update(column=str(column), categorical=bool(categorical))
        diagnostics.append(result)
    return diagnostics


def data_fingerprint(X, Y) -> str:
    """Row-order-invariant fingerprint of the numeric feature/target table."""
    frame = pd.DataFrame(np.c_[np.asarray(X, float), np.asarray(Y, float)])
    hashes = np.sort(pd.util.hash_pandas_object(frame, index=False).to_numpy())
    digest = hashlib.sha256()
    digest.update(str((frame.shape, np.shape(Y)[1])).encode())
    digest.update(hashes.tobytes())
    return digest.hexdigest()


def screen_dataset(X, Y, seed: int = 0, *, name: str | None = None, run_test: bool = True,
                   effect_min: float = EFFECT_MIN) -> tuple[dict, np.ndarray]:
    """Screen a fixed holdout and return the disjoint benchmark row indices."""
    diagnostics = target_audit(Y, name=name)
    feature_names = [str(column) for column in pd.DataFrame(X).columns]
    X = np.asarray(X, float)
    Y = pd.DataFrame(Y).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if X.ndim != 2 or Y.ndim != 2 or len(X) != len(Y):
        raise ValueError("X and Y must be aligned two-dimensional arrays")
    valid = np.isfinite(Y).all(axis=1)
    curation = curation_split(len(Y), seed=seed)
    benchmark = np.setdiff1d(np.flatnonzero(valid), curation)
    curation = curation[valid[curation]]
    is_ordinal = any(
        item.get("categorical") or
        (item.get("is_discrete") and item.get("rule") != "atom_mass")
        for item in diagnostics
    )
    has_atoms = any(item.get("is_discrete") and item.get("rule") == "atom_mass" for item in diagnostics)
    audit = dict(seed=seed, n_curation=len(curation), n_benchmark=len(benchmark),
                 n_missing_targets=int((~valid).sum()),
                 fingerprint=data_fingerprint(X, Y),
                 target_diagnostics=diagnostics,
                 is_ordinal=is_ordinal,
                 has_atoms=has_atoms,
                 curation_indices=curation.tolist(),
                 curation_fingerprint=data_fingerprint(X[curation], Y[curation]))
    rejected = dict(selected=False, scale_ablation=False, kci_p=None,
                    residual_dcor=None, scale_ratio=None,
                    n_features=X.shape[1], n_targets=Y.shape[1], **audit)
    if is_ordinal:
        return dict(rejected, reason="noncontinuous_target"), benchmark
    features = X[curation]
    usable = np.isfinite(features).any(axis=0)
    features = features[:, usable]
    features[~np.isfinite(features)] = np.nan
    if features.shape[1]:
        features = SimpleImputer(strategy="median").fit_transform(features)
    if len(curation) >= max(20, 5 * (Y.shape[1] + 1)):
        leakage = leaky_features(features, Y[curation])
        if leakage.any():
            names = np.asarray(feature_names)[usable][leakage].tolist()
            return dict(rejected, reason="target_leakage", leaky_features=names), benchmark
        centered = Y[curation] - Y[curation].mean(axis=0)
        scales = centered.std(axis=0)
        if np.all(scales > 0) and np.linalg.matrix_rank(centered / scales) < Y.shape[1]:
            return dict(rejected, reason="redundant_targets"), benchmark
    if not run_test:
        return dict(rejected, reason="quality_checked"), benchmark
    result = screen_kci(features, Y[curation], effect_min=effect_min, seed=seed)
    result.update(audit)
    return result, benchmark
