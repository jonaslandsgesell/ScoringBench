"""Source 2: synthetic copula-coupled multivariate regression targets.

This module is a *dataset source* for the multivariate benchmark. It plugs into
the source registry (:mod:`scoringbench.multivariate.sources`) via the two
public entry points every source exposes:

* :func:`enumerate_synthetic` — deterministic list of dataset configs, and
* :func:`load_synthetic` — ``(X, Y)`` DataFrames for one config.

Design (why these datasets are hard for independent models)
-----------------------------------------------------------
Each dataset is a ``d``-target regression problem built as **feature signal +
copula-coupled residual**::

    Y_k = f_k(X) + noise_scale * eps_k ,   k = 0 .. d-1

The features ``X`` are i.i.d. standard normal; ``f_k`` is a smooth
random-but-fixed function of ``X`` (a random linear combination passed through a
nonlinearity). The residual vector ``eps = (eps_0, .., eps_{d-1})`` is drawn from
an **explicitly constructed vine copula** (pyvinecopulib) with standard-normal
marginals. Crucially the copula acts on the *residual* — the part of ``Y`` that
``X`` does NOT explain — so the cross-target dependence **survives conditioning
on X**. That is exactly the structure that

* a product-of-marginals / *independent* model cannot represent (its joint is a
  product, so it collapses the residual dependence to zero), while
* a *copula* or *chained* model can recover.

The copula is built with **fixed** family and parameter per dataset (no fitting)
so the only source of run-to-run variation is the sampler RNG — which is why we
freeze the generated arrays as artifacts (see below).

Reproducibility (Option B: frozen artifacts + regenerate fallback)
------------------------------------------------------------------
The generated parquet files under ``datasets/synthetic/`` are the ground truth:
:func:`load_synthetic` reads those exact bytes and (when a manifest is present)
verifies their sha256. Only on a **cache miss** does it regenerate via
:func:`_generate`, emitting a warning that regenerated values MAY differ across
numpy / pyvinecopulib versions. Regenerate the committed set with
``scripts/generate_synthetic.py``.

Determinism
-----------
All randomness derives from ``numpy.random.SeedSequence(seed)`` spawned into
independent child streams (features, target functions, residual copula sim). The
pyvinecopulib integer seed list is derived from a spawned child's
``generate_state`` so the copula sampler is seeded deterministically too. No use
of the global ``numpy.random`` state.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import config as cfg

# ---------------------------------------------------------------------------
# Artifact location + manifest
# ---------------------------------------------------------------------------
# config.SYNTHETIC_DATA_SUBDIR is relative to the repo root (the parent of the
# ``scoringbench`` package). __file__ is .../scoringbench/multivariate/this.py,
# so parents[2] is the repo root.
#
# Artifacts are scoped by (target_dim d, sample_size n) into a SUBFOLDER
# ``datasets/synthetic/d{d}_n{n}/`` so that different d / n sweeps never collide
# (each subfolder carries its own ``manifest.json``). The frozen set for a given
# shape must be generated explicitly with ``scripts/generate_synthetic.py``;
# loading NEVER regenerates on the fly (see :func:`load_synthetic`).
_REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_DIR = _REPO_ROOT / cfg.SYNTHETIC_DATA_SUBDIR
MANIFEST_NAME = "manifest.json"

SOURCE_NAME = "synthetic"


def _shape_subdir(target_dim: int, sample_size: int) -> Path:
    """Per-(d, n) artifact subfolder: ``datasets/synthetic/d{d}_n{n}/``."""
    return SYNTHETIC_DIR / f"d{int(target_dim)}_n{int(sample_size)}"

# Column naming mirrors the feature-promotion source
# (datasets.promote_features_to_targets): target_0 first, then target_1..; the
# synthetic loader MUST return the same contract so downstream code is
# source-agnostic.
_TARGET_PREFIX = "target_"
_FEATURE_PREFIX = "feature_"


# ---------------------------------------------------------------------------
# Deterministic RNG stream layout
# ---------------------------------------------------------------------------
# A single SeedSequence(seed) is spawned into these named streams. The order is
# part of the reproducibility contract — do NOT reorder or the frozen artifacts
# would no longer match a regeneration.
_STREAM_NAMES = (
    "features",       # X matrix
    "target_coeffs",  # linear coeffs + nonlinearity mixing for f_k(X)
    "copula_sim",     # residual copula .simulate() seeds
)


def _spawn_streams(seed: int) -> dict[str, np.random.Generator]:
    """Spawn one independent :class:`numpy.random.Generator` per named stream."""
    children = np.random.SeedSequence(int(seed)).spawn(len(_STREAM_NAMES))
    return {name: np.random.default_rng(child)
            for name, child in zip(_STREAM_NAMES, children)}


def _pv_seeds(rng: np.random.Generator, k: int = 5) -> list[int]:
    """Derive a deterministic integer seed list for ``Vinecop.simulate``.

    pyvinecopulib expects a list of small non-negative ints; we draw them from
    the copula stream so the sampler is tied to the same SeedSequence tree.
    """
    return rng.integers(0, 2**31 - 1, size=k).tolist()


# ---------------------------------------------------------------------------
# Explicit vine construction (no fitting)
# ---------------------------------------------------------------------------

def _build_vine(family: str, tau: float, d: int):
    """Construct a D-vine with all pair-copulas fixed to (family, tau).

    Uses pyvinecopulib's explicit-construction API so the copula is fully
    determined by ``(family, tau, d)`` — there is no data-dependent fitting, so
    the copula parameters are identical on every machine/version.
    """
    import pyvinecopulib as pv

    fam = getattr(pv.BicopFamily, family)
    struct = pv.DVineStructure(order=list(range(1, d + 1)))
    pair_copulas = []
    for tree in range(d - 1):
        row = []
        for _ in range(d - 1 - tree):
            bicop = pv.Bicop(family=fam)
            params = np.atleast_2d(bicop.tau_to_parameters(tau))
            row.append(pv.Bicop(family=fam, parameters=params))
        pair_copulas.append(row)
    return pv.Vinecop.from_structure(structure=struct, pair_copulas=pair_copulas)


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------

def _dataset_name(family: str, tau: float, replicate: int, d: int) -> str:
    """Stable, filesystem-safe dataset name encoding all generating params."""
    tau_tag = f"{tau:.2f}".replace(".", "p")
    return f"syn_{family}_tau{tau_tag}_d{d}_r{replicate}"


def _replicates_per_cell() -> list[int]:
    """Distribute ``config.SYNTHETIC_N_DATASETS`` across the family x tau cells.

    Returns the replicate count for each of the ``len(families) * len(taus)``
    cells, in the fixed (family-major, tau-minor) iteration order. The total
    equals ``SYNTHETIC_N_DATASETS`` exactly; counts differ by at most one, and
    the first ``remainder`` cells get the extra replicate (deterministic).
    """
    n_cells = len(cfg.SYNTHETIC_FAMILIES) * len(cfg.SYNTHETIC_TAUS)
    total = int(cfg.SYNTHETIC_N_DATASETS)
    base, remainder = divmod(total, n_cells)
    return [base + (1 if i < remainder else 0) for i in range(n_cells)]


def enumerate_synthetic(target_dim: int, sample_size: int) -> list[dict[str, Any]]:
    """Deterministic list of synthetic dataset configs for the given shape.

    The grid enumerates ``families x taus`` cells (see ``config.SYNTHETIC_*``),
    each populated with independently seeded replicates so the total dataset
    count equals ``config.SYNTHETIC_N_DATASETS`` (distributed as evenly as
    possible across the cells). Each config is a plain dict carrying everything
    :func:`load_synthetic` / :func:`_generate` need, plus the provenance keys
    (``source``/``id``) that
    :func:`scoringbench.multivariate.results.build_results_rows` propagates.

    Seeds are assigned by a *stable enumeration index* so a given cell/replicate
    always maps to the same seed regardless of ``target_dim`` / ``sample_size``.
    """
    d = int(target_dim)
    n = int(sample_size)
    reps_per_cell = _replicates_per_cell()
    configs: list[dict[str, Any]] = []
    idx = 0
    cell = 0
    for family in cfg.SYNTHETIC_FAMILIES:
        for tau in cfg.SYNTHETIC_TAUS:
            for replicate in range(reps_per_cell[cell]):
                name = _dataset_name(family, tau, replicate, d)
                # Base seed is derived deterministically from the config so it
                # never depends on iteration order beyond the stable index.
                seed = int(cfg.SEED) * 1_000 + idx
                configs.append({
                    "name": name,
                    "id": name,
                    "source": SOURCE_NAME,
                    "family": family,
                    "tau": float(tau),
                    "replicate": int(replicate),
                    "seed": seed,
                    "n_samples": n,
                    "n_features": int(cfg.SYNTHETIC_N_FEATURES),
                    "target_dim": d,
                    "noise_scale": float(cfg.SYNTHETIC_NOISE_SCALE),
                    "mean_scale": float(cfg.SYNTHETIC_MEAN_SCALE),
                })
                idx += 1
            cell += 1
    return configs


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _target_functions(X: np.ndarray, d: int, rng: np.random.Generator,
                      mean_scale: float) -> np.ndarray:
    """Smooth conditional means ``f_k(X)`` for k = 0..d-1.

    Each target is ``mean_scale * tanh(X @ W_k)`` with a random-but-fixed
    projection ``W_k`` — a genuinely nonlinear, X-dependent mean that models
    must learn. ``tanh`` keeps the mean bounded so its variance is controlled by
    ``mean_scale`` alone; this matters because the whole point of the source is
    that the copula-coupled *residual* (added by the caller with a much larger
    scale) DOMINATES the mean, so a model that ignores the residual dependence
    fails hard. The dependence we test lives in the residual, not here.
    """
    _, p = X.shape
    W = rng.normal(size=(p, d))
    return mean_scale * np.tanh(X @ W)


def _generate(ds_config: dict[str, Any], target_dim: int | None = None
              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate one synthetic dataset from its config (no disk access).

    Returns ``(X, Y)`` DataFrames matching the feature-promotion contract:
    ``target_0`` first, features named ``feature_0..``; ``X`` always has >= 1
    column.
    """
    d = int(target_dim if target_dim is not None else ds_config["target_dim"])
    n = int(ds_config["n_samples"])
    p = int(ds_config["n_features"])
    family = str(ds_config["family"])
    tau = float(ds_config["tau"])
    noise_scale = float(ds_config["noise_scale"])
    mean_scale = float(ds_config.get("mean_scale", cfg.SYNTHETIC_MEAN_SCALE))
    if p < 1:
        raise ValueError(f"synthetic n_features must be >= 1, got {p}")
    if d < 1:
        raise ValueError(f"synthetic target_dim must be >= 1, got {d}")

    streams = _spawn_streams(int(ds_config["seed"]))

    # 1) Features
    X = streams["features"].standard_normal(size=(n, p))

    # 2) Conditional means f_k(X)
    mean = _target_functions(X, d, streams["target_coeffs"], mean_scale)

    # 3) Copula-coupled residuals with standard-normal marginals.
    if d >= 2:
        vine = _build_vine(family, tau, d)
        u = np.asarray(vine.simulate(n, seeds=_pv_seeds(streams["copula_sim"])))
        # PIT -> standard normal marginals (clip to avoid +/-inf at 0/1).
        u = np.clip(u, 1e-12, 1 - 1e-12)
        eps = norm.ppf(u)
    else:
        # Univariate degenerate case: no copula, plain normal residual.
        eps = streams["copula_sim"].standard_normal(size=(n, 1))

    Y = mean + noise_scale * eps

    X_df = pd.DataFrame(
        X, columns=[f"{_FEATURE_PREFIX}{j}" for j in range(p)]
    )
    Y_df = pd.DataFrame(
        Y, columns=[f"{_TARGET_PREFIX}{k}" for k in range(d)]
    )
    return X_df, Y_df


# ---------------------------------------------------------------------------
# Cache + manifest helpers
# ---------------------------------------------------------------------------

def _artifact_path(name: str, target_dim: int, sample_size: int) -> Path:
    return _shape_subdir(target_dim, sample_size) / f"{name}.parquet"


def _manifest_path(target_dim: int, sample_size: int) -> Path:
    return _shape_subdir(target_dim, sample_size) / MANIFEST_NAME


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest(target_dim: int, sample_size: int) -> dict[str, Any]:
    path = _manifest_path(target_dim, sample_size)
    if path.exists():
        with open(path) as fh:
            return json.load(fh)
    return {}


def _split_xy(df: pd.DataFrame, target_dim: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a stored frame into ``(X, Y)`` by column-name prefix.

    Artifacts store features and targets in one frame; targets are the columns
    named ``target_0..target_{d-1}`` (in order), everything else is a feature.
    """
    target_cols = [f"{_TARGET_PREFIX}{k}" for k in range(target_dim)]
    missing = [c for c in target_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"stored synthetic frame is missing target columns {missing}; "
            f"available columns: {list(df.columns)}"
        )
    feature_cols = [c for c in df.columns if c not in target_cols]
    if not feature_cols:
        raise ValueError("stored synthetic frame has no feature columns")
    return df[feature_cols].copy(), df[target_cols].copy()


# ---------------------------------------------------------------------------
# Loading (cache-first, regenerate fallback)
# ---------------------------------------------------------------------------

def load_synthetic(ds_config: dict[str, Any], target_dim: int | None = None
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load one FROZEN synthetic dataset ``(X, Y)`` from disk.

    This reads the committed parquet artifact for the dataset's ``(d, n)`` shape
    and (when a manifest entry exists) verifies its sha256. It **never**
    regenerates on the fly: if the artifact is missing it raises
    :class:`FileNotFoundError` with the exact command to generate it. This keeps
    results reproducible — the frozen bytes are the single source of truth, and
    silent version-dependent regeneration can never sneak in.
    """
    d = int(target_dim if target_dim is not None else ds_config["target_dim"])
    n = int(ds_config["n_samples"])
    name = str(ds_config["name"])
    path = _artifact_path(name, d, n)

    if not path.exists():
        subdir = _shape_subdir(d, n)
        raise FileNotFoundError(
            f"Synthetic artifact '{name}' for shape (d={d}, n={n}) was not found "
            f"at:\n    {path}\n"
            f"The synthetic source does NOT generate data on the fly. Generate "
            f"the frozen set for this shape explicitly, then re-run:\n"
            f"    PYTHONPATH=. python scripts/generate_synthetic.py "
            f"--target-dim {d} --sample-size {n}\n"
            f"(this writes the parquet artifacts + manifest.json into {subdir})"
        )

    manifest = _load_manifest(d, n)
    entry = manifest.get(name)
    if entry is not None and "sha256" in entry:
        actual = _sha256(path)
        if actual != entry["sha256"]:
            warnings.warn(
                f"synthetic artifact '{name}' sha256 mismatch "
                f"(manifest={entry['sha256'][:12]}.., file={actual[:12]}..); "
                f"the committed artifact may have been modified.",
                RuntimeWarning,
                stacklevel=2,
            )
    df = pd.read_parquet(path)
    return _split_xy(df, d)
