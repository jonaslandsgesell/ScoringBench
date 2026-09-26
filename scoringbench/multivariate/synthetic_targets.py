"""Nonlinear latent-factor regression with feature-dependent target coupling.

The design follows Nagler, Schellhase and Czado (2017), Section 4.1:
https://arxiv.org/html/1701.00845v2#S4.SS1
We thank Thibault Vatter and Thomas Nagler for their guidance on the synthetic
data generation strategy.

Each replicate draws a structure, per-edge families, signed beta-distributed
Kendall taus, and weaker dependence in higher trees. Structure sampling uses
pyvinecopulib's uniform R-vine sampler, as suggested by Vatter, rather than the
paper's independence-data structure selection heuristic.

The benchmark crosses strong dependence with tail/no-tail/mixed innovation
families. A shared nonlinear signal and latent noise drive all targets, with
smaller target-specific nonlinear signals and R-vine innovations. Balanced,
random target groups change their relative sign in a feature-defined regime.
Consequently the conditional copula varies with X, and earlier targets in any
chain reveal the shared latent state. Neither independent marginals nor one
feature-independent copula can represent this law. The realized vine describes
the innovations, not the final conditional copula; finite-sample rankings are
not guaranteed.

Model construction and observations use independent seeded streams. Artifacts
are frozen under ``datasets/synthetic/d{d}_n{n}``; loading never regenerates data.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.linalg import hadamard
from scipy.stats import norm

from . import config as cfg

if TYPE_CHECKING:
    import pyvinecopulib as pv

# ---------------------------------------------------------------------------
# Artifact location + manifest
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_DIR = _REPO_ROOT / cfg.SYNTHETIC_DATA_SUBDIR
MANIFEST_NAME = "manifest.json"

SOURCE_NAME = "synthetic"


def _shape_subdir(target_dim: int, sample_size: int) -> Path:
    """Artifact subfolder for a target dimension and sample size."""
    return SYNTHETIC_DIR / f"d{target_dim}_n{sample_size}"

# Column naming mirrors the feature-promotion source
# (datasets.promote_features_to_targets): target_0 first, then target_1..; the
# synthetic loader MUST return the same contract so downstream code is
# source-agnostic.
_TARGET_PREFIX = "target_"
_FEATURE_PREFIX = "feature_"


# ---------------------------------------------------------------------------
# Deterministic RNG stream layout
# ---------------------------------------------------------------------------
_STREAM_NAMES = (
    "copula_sim",
    "vine_structure",
    "pair_families",
    "pair_taus",
    "pair_rotations",
    "target_mean",
    "features",
    "shared_noise",
    "target_dependence",
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


_DEPENDENCE_TYPES = ("tail", "no_tail", "mixed")
_TAIL_FAMILIES = ("student", "clayton", "gumbel")
_NO_TAIL_FAMILIES = ("gaussian", "frank")
_TAU_BETA_PARAMETERS = {"weak": (1.0, 4.0), "strong": (5.0, 5.0)}


# ---------------------------------------------------------------------------
# Nonlinear conditional mean (always injected)
# ---------------------------------------------------------------------------
_MEAN_N_FEATURES = 1
_MEAN_N_CALIBRATION = 2048
_MEAN_N_RIDGE = 12
_MEAN_N_INTERACTIONS = 6
_MEAN_SIGNAL_FRACTION = 0.6
_MEAN_RIDGE_ACTIVATIONS = ("sin", "tanh", "softplus", "square")
_TARGET_SHARED_FRACTION = 0.98
_TARGET_MIN_REGIMES = 4
# Half of the shared nonlinear mean follows the feature-gated Hadamard signs.
# gm=0 leaves the mean learnable per-target but gives chaining a weak edge;
# gm=1 maximizes feature-dependent coupling but the per-target mean becomes
# unlearnable (GBM R^2 collapses to ~0). gm=0.5 keeps GBM R^2 in [0.30, 0.44]
# (learnable, linear R^2 ~ 0) while giving chaining a large, robust advantage.
_GATED_MEAN_FRACTION = 0.5
# Sharper two modes make it easier for a chain to recover which mode a row is in
# once earlier targets are observed, so the energy score rewards dependence more
# (oracle ES ceiling grows as the modes separate). 0.05 lifts realized chained
# energy gains to 7.6-17% across d/scenarios while keeping the latent continuous.
_SHARED_NOISE_WITHIN_MODE_SD = 0.05


def _apply_activation(name: str, z: np.ndarray) -> np.ndarray:
    if name == "sin":
        return np.sin(z)
    if name == "tanh":
        return np.tanh(z)
    if name == "softplus":
        return np.logaddexp(0.0, z)
    if name == "square":
        return z * z - 1.0  # centered (E[z^2]=1 for standard-normal-ish z)
    raise ValueError(f"unknown activation: {name!r}")


def _nonlinear_mean(features: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw a sparse, even nonlinear function with no population linear signal.

    Symmetrized ridge activations and products satisfy f(X) = f(-X). With
    independent normal features this makes Cov(X, f(X)) zero. A fixed,
    independently drawn calibration sample sets the function's scale without
    using the evaluation batch, so changing row count does not change f.
    """
    n_features = features.shape[1]
    n_active = min(_MEAN_N_FEATURES, n_features)
    active = rng.choice(n_features, size=n_active, replace=False)
    calibration = rng.standard_normal((_MEAN_N_CALIBRATION, n_active))
    inputs = np.concatenate((calibration, features[:, active]), axis=0)
    contributions = np.zeros(len(inputs), dtype=np.float64)

    for _ in range(_MEAN_N_RIDGE):
        weights = rng.standard_normal(n_active)
        weights /= np.linalg.norm(weights) + 1e-12
        projection = inputs @ weights
        bias = rng.uniform(-1.0, 1.0)
        activation = _MEAN_RIDGE_ACTIVATIONS[int(rng.integers(len(_MEAN_RIDGE_ACTIVATIONS)))]
        amplitude = rng.standard_normal()
        contributions += 0.5 * amplitude * (
            _apply_activation(activation, projection + bias)
            + _apply_activation(activation, -projection + bias)
        )

    for _ in range(_MEAN_N_INTERACTIONS):
        first = int(rng.integers(n_active))
        second = int(rng.integers(n_active))
        coefficient = rng.standard_normal()
        contributions += coefficient * (inputs[:, first] * inputs[:, second])

    reference = contributions[:_MEAN_N_CALIBRATION]
    return ((contributions[_MEAN_N_CALIBRATION:] - reference.mean())
            / max(float(reference.std()), 1e-12))


def _validate_integer(name: str, value: int, minimum: int = 1) -> None:
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value}")


def simulate_random_vine(
    dim: int, decay: float = 0.8, upper: float = 1.0, seed: int = 42,
    *, dependence_type: str = "mixed", dependence_strength: str = "strong",
) -> pv.Vinecop:
    """Draw a simplified R-vine using the randomized design of Section 4.1.

    Structures are sampled uniformly by pyvinecopulib. Each edge independently
    draws a family, an absolute tau from Beta(1, 4) (weak) or Beta(5, 5)
    (strong), and a sign. Tail families are Student-t (df=4), Clayton and
    Gumbel; no-tail families are Gaussian and Frank. The mixed regime chooses
    either group with probability 1/2, then uniformly within the group.

    Tree levels start at one: abs(tau) = min(beta_draw, upper) * decay**level.
    Thus the default first-tree expected strengths are 0.16 and 0.40. Unlike
    a positive tau floor, this permits arbitrarily weak higher-tree dependence.
    Clayton/Gumbel rotations are uniform over 0, 90, 180 and 270 degrees.
    Parameters are clipped only to the library's supported family bounds.

    ``dim`` must be a positive integer, ``seed`` a nonnegative integer,
    ``decay`` finite in [0, 1], and ``upper`` finite in (0, 1].
    """
    import pyvinecopulib as pv

    _validate_integer("dim", dim)
    _validate_integer("seed", seed, minimum=0)
    if not np.isfinite(decay) or not 0 <= decay <= 1:
        raise ValueError(f"decay must be in [0, 1], got {decay}")
    if not np.isfinite(upper) or not 0 < upper <= 1:
        raise ValueError(f"upper must be in (0, 1], got {upper}")
    if dependence_type not in _DEPENDENCE_TYPES:
        raise ValueError(f"unknown dependence_type: {dependence_type!r}")
    if dependence_strength not in _TAU_BETA_PARAMETERS:
        raise ValueError(f"unknown dependence_strength: {dependence_strength!r}")

    streams = _spawn_streams(seed)
    sample_structure = getattr(pv.RVineStructure, "sample", pv.RVineStructure.simulate)
    structure = sample_structure(dim, seeds=_pv_seeds(streams["vine_structure"]))
    family_rng = streams["pair_families"]
    tau_rng = streams["pair_taus"]
    rotation_rng = streams["pair_rotations"]
    beta_parameters = _TAU_BETA_PARAMETERS[dependence_strength]
    pair_copulas = []
    for tree in range(dim - 1):
        pair_copulas_tree = []
        for _ in range(dim - tree - 1):
            tail = dependence_type == "tail"
            if dependence_type == "mixed":
                tail = bool(family_rng.integers(2))
            families = _TAIL_FAMILIES if tail else _NO_TAIL_FAMILIES
            family = families[int(family_rng.integers(len(families)))]
            tau = min(float(tau_rng.beta(*beta_parameters)), upper) * decay ** (tree + 1)
            rotation = int(rotation_rng.choice((0, 90, 180, 270)))
            if tau <= np.finfo(np.float64).eps:
                pair_copulas_tree.append(pv.Bicop())
                continue

            bicop = pv.Bicop(family=getattr(pv.BicopFamily, family))
            if family not in ("clayton", "gumbel"):
                tau *= -1 if rotation in (90, 270) else 1
                rotation = 0
            if family == "student":
                parameters = np.array([[np.sin(np.pi * tau / 2)], [4.0]])
            else:
                parameters = bicop.tau_to_parameters(tau)
            parameters = np.clip(parameters, bicop.parameters_lower_bounds,
                                 bicop.parameters_upper_bounds)
            pair_copulas_tree.append(pv.Bicop(
                family=bicop.family, rotation=rotation, parameters=parameters,
            ))
        pair_copulas.append(pair_copulas_tree)
    return pv.Vinecop.from_structure(structure=structure, pair_copulas=pair_copulas)


def _sample_normal_data(vine: pv.Vinecop, n_samples: int,
                        rng: np.random.Generator) -> np.ndarray:
    sample = getattr(vine, "sample", vine.simulate)
    uniforms = np.asarray(sample(n_samples, seeds=_pv_seeds(rng)), dtype=np.float64)
    return norm.ppf(np.clip(uniforms, 1e-12, 1 - 1e-12))


def simulate_synthetic_data(
    n_samples: int, dim: int, decay: float = 0.8, upper: float = 1.0,
    seed: int = 42, *, dependence_type: str = "mixed",
    dependence_strength: str = "strong",
) -> np.ndarray:
    """Sample a randomized vine with standard-normal margins, including dim=1.

    The construction and observation streams are independent. Increasing the
    sample size never changes the generating vine. Endpoint uniforms are
    clipped before the normal quantile transform so the output stays finite.
    """
    _validate_integer("n_samples", n_samples)
    vine = simulate_random_vine(
        dim, decay=decay, upper=upper, seed=seed,
        dependence_type=dependence_type, dependence_strength=dependence_strength,
    )
    return _sample_normal_data(vine, n_samples, _spawn_streams(seed)["copula_sim"])


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------

def _dataset_name(parameters: dict[str, Any]) -> str:
    """Include the model definition in the identity, independently of row count."""
    model_parameters = {key: value for key, value in parameters.items()
                        if key != "n_samples"}
    digest = hashlib.sha256(json.dumps(
        model_parameters, sort_keys=True, allow_nan=False,
    ).encode()).hexdigest()[:12]
    return (f"syn_{parameters['dependence_type']}_"
            f"{parameters['dependence_strength']}_d{parameters['target_dim']}_"
            f"r{parameters['replicate']}_{digest}")


def _replicates_per_cell() -> list[int]:
    """Balance the requested total across dependence-type x strength cells."""
    for name, choices, allowed in (
        ("SYNTHETIC_DEPENDENCE_TYPES", cfg.SYNTHETIC_DEPENDENCE_TYPES, _DEPENDENCE_TYPES),
        ("SYNTHETIC_DEPENDENCE_STRENGTHS", cfg.SYNTHETIC_DEPENDENCE_STRENGTHS,
         _TAU_BETA_PARAMETERS),
    ):
        if not choices or len(set(choices)) != len(choices) or any(
            choice not in allowed for choice in choices
        ):
            raise ValueError(f"{name} must contain distinct choices from {tuple(allowed)}")
    _validate_integer("SYNTHETIC_N_DATASETS", cfg.SYNTHETIC_N_DATASETS)
    _validate_integer("SEED", cfg.SEED, minimum=0)
    n_cells = len(cfg.SYNTHETIC_DEPENDENCE_TYPES) * len(cfg.SYNTHETIC_DEPENDENCE_STRENGTHS)
    total = cfg.SYNTHETIC_N_DATASETS
    base, remainder = divmod(total, n_cells)
    return [base + (cell < remainder) for cell in range(n_cells)]


def enumerate_synthetic(target_dim: int, sample_size: int) -> list[dict[str, Any]]:
    """Enumerate balanced, independently randomized model replicates.

    Seeds depend on the base seed, scenario and replicate, never on
    enumeration position, suite size or observation count. Changing a model
    parameter changes its dataset identity, preventing stale result reuse.
    """
    _validate_integer("target_dim", target_dim)
    _validate_integer("sample_size", sample_size)
    reps_per_cell = _replicates_per_cell()
    configs: list[dict[str, Any]] = []
    cell = 0
    for dependence_type in cfg.SYNTHETIC_DEPENDENCE_TYPES:
        for dependence_strength in cfg.SYNTHETIC_DEPENDENCE_STRENGTHS:
            for replicate in range(reps_per_cell[cell]):
                seed = int(np.random.SeedSequence([
                    cfg.SEED, _DEPENDENCE_TYPES.index(dependence_type),
                    tuple(_TAU_BETA_PARAMETERS).index(dependence_strength), replicate,
                ]).generate_state(1, dtype=np.uint64)[0])
                parameters = {
                    "source": SOURCE_NAME,
                    "sampling": "feature_gated_shared_latent_with_vine_innovations",
                    "nonlinear_mean": {
                        "n_features": _MEAN_N_FEATURES,
                        "calibration_samples": _MEAN_N_CALIBRATION,
                        "n_ridge": _MEAN_N_RIDGE,
                        "n_interactions": _MEAN_N_INTERACTIONS,
                        "signal_fraction": _MEAN_SIGNAL_FRACTION,
                        "ridge_activations": list(_MEAN_RIDGE_ACTIVATIONS),
                        "symmetrized": True,
                    },
                    "target_dependence": {
                        "shared_fraction": _TARGET_SHARED_FRACTION,
                        "min_regimes": _TARGET_MIN_REGIMES,
                        "gated_mean_fraction": _GATED_MEAN_FRACTION,
                        "regimes": "hadamard_rows_by_feature_signs",
                        "shared_noise": "symmetric_two_normal_mixture_unit_variance",
                        "within_mode_sd": _SHARED_NOISE_WITHIN_MODE_SD,
                        "mean_gate_shared_feature": True,
                        "gate_applies_to": "shared_signal_and_noise",
                    },
                    "dependence_type": dependence_type,
                    "dependence_strength": dependence_strength,
                    "decay": float(cfg.SYNTHETIC_DECAY),
                    "tau_upper": float(cfg.SYNTHETIC_TAU_UPPER),
                    "replicate": replicate,
                    "seed": seed,
                    "n_samples": int(sample_size),
                    "n_features": int(cfg.SYNTHETIC_N_FEATURES),
                    "target_dim": int(target_dim),
                }
                name = _dataset_name(parameters)
                configs.append(dict(parameters, name=name, id=name))
            cell += 1
    return configs


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _generate_with_vine(ds_config: dict[str, Any], target_dim: int | None = None
                        ) -> tuple[pd.DataFrame, pd.DataFrame, pv.Vinecop]:
    """Generate coupled targets and retain the target-specific innovation vine."""
    dimension = ds_config["target_dim"] if target_dim is None else target_dim
    _validate_integer("target_dim", dimension)
    if dimension != ds_config["target_dim"]:
        raise ValueError("target_dim does not match the synthetic dataset config")
    n_samples = ds_config["n_samples"]
    n_features = ds_config["n_features"]
    _validate_integer("n_samples", n_samples)
    _validate_integer("n_features", n_features)

    vine = simulate_random_vine(
        dimension, decay=ds_config["decay"], upper=ds_config["tau_upper"],
        seed=ds_config["seed"], dependence_type=ds_config["dependence_type"],
        dependence_strength=ds_config["dependence_strength"],
    )
    streams = _spawn_streams(ds_config["seed"])
    features = streams["features"].standard_normal((n_samples, n_features))
    residuals = _sample_normal_data(vine, n_samples, streams["copula_sim"])
    shared_noise = _sample_shared_noise(n_samples, streams["shared_noise"])
    targets = _generate_targets(features, residuals, shared_noise, ds_config["seed"])
    return (
        pd.DataFrame(features, columns=[f"{_FEATURE_PREFIX}{index}" for index in range(n_features)]),
        pd.DataFrame(targets, columns=[f"{_TARGET_PREFIX}{index}" for index in range(dimension)]),
        vine,
    )


def _inject_nonlinear_mean(features: np.ndarray, targets: np.ndarray,
                           rng: np.random.Generator) -> np.ndarray:
    """Add nonlinear means without changing the feature-independent copula.

    Inputs in ``targets`` have standard-normal margins. Their variance must
    not be estimated from the observed targets: the fixed positive residual
    scale preserves the same copula at every X, including outside the sample.
    """
    signal = float(_MEAN_SIGNAL_FRACTION)
    weight_mean = np.sqrt(signal)
    weight_residual = np.sqrt(1.0 - signal)
    blended = np.empty_like(targets)
    for column in range(targets.shape[1]):
        mean_signal = _nonlinear_mean(features, rng)
        blended[:, column] = weight_mean * mean_signal + weight_residual * targets[:, column]
    return blended


def _target_loadings(features: np.ndarray, dimension: int, rng: np.random.Generator
                     ) -> tuple[int, np.ndarray, np.ndarray]:
    """Fixed mean signs and per-row noise signs from Hadamard rows chosen by feature signs.

    Orthogonal columns make every pairwise sign product average to zero over
    the equally likely regimes, so an X-independent copula sees little
    dependence while the conditional law given X is almost degenerate.
    """
    n_regimes = max(_TARGET_MIN_REGIMES, 1 << int(np.ceil(np.log2(max(dimension, 1)))))
    n_gates = int(np.log2(n_regimes))
    n_features = features.shape[1]
    chosen = rng.choice(n_features, size=n_gates + 1, replace=n_features <= n_gates)
    mean_feature, gates = int(chosen[0]), chosen[1:]
    regime = np.zeros(len(features), dtype=int)
    for bit, gate in enumerate(gates):
        regime |= (features[:, gate] > 0).astype(int) << bit
    columns = rng.permutation(n_regimes)[:dimension]
    signs = rng.choice((-1.0, 1.0), size=dimension)
    return mean_feature, signs, hadamard(n_regimes)[regime][:, columns] * signs


def _generate_targets(features: np.ndarray, innovations: np.ndarray,
                      shared_noise: np.ndarray, seed: int) -> np.ndarray:
    """Evaluate a fixed conditional law using caller-supplied stochastic draws.

    A shared nonlinear mean with fixed signs keeps E[Y|X] learnable. A
    continuous two-mode latent factor with feature-sign-dependent target signs
    dominates the noise: once X and any one target are known the others are
    nearly determined, in every chain order.
    """
    streams = _spawn_streams(seed)
    signal = float(_MEAN_SIGNAL_FRACTION)
    mean_feature, mean_signs, noise_signs = _target_loadings(
        features, innovations.shape[1], streams["target_dependence"],
    )
    shared_mean = _nonlinear_mean(features[:, [mean_feature]], streams["target_mean"])
    private = _inject_nonlinear_mean(features, innovations, streams["target_mean"])
    gated = float(_GATED_MEAN_FRACTION)
    mean_loadings = np.sqrt(1.0 - gated) * mean_signs + np.sqrt(gated) * noise_signs
    shared = (np.sqrt(signal) * mean_loadings * shared_mean[:, None]
              + np.sqrt(1.0 - signal) * noise_signs * shared_noise[:, None])
    return (np.sqrt(_TARGET_SHARED_FRACTION) * shared
            + np.sqrt(1.0 - _TARGET_SHARED_FRACTION) * private)


def _sample_shared_noise(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Unit-variance, continuous two-regime uncertainty shared across targets."""
    within_mode_sd = float(_SHARED_NOISE_WITHIN_MODE_SD)
    modes = rng.choice((-1.0, 1.0), size=n_samples)
    return (np.sqrt(1.0 - within_mode_sd ** 2) * modes
            + within_mode_sd * rng.standard_normal(n_samples))


def _generate(ds_config: dict[str, Any], target_dim: int | None = None
              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate one dataset in memory with the source's (X, Y) contract."""
    features, targets, _ = _generate_with_vine(ds_config, target_dim=target_dim)
    return features, targets


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
# Loading (frozen artifacts only)
# ---------------------------------------------------------------------------

def load_synthetic(ds_config: dict[str, Any], target_dim: int | None = None
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load frozen arrays after verifying config, checksum and schema."""
    dimension = ds_config["target_dim"] if target_dim is None else target_dim
    n_samples = ds_config["n_samples"]
    n_features = ds_config["n_features"]
    _validate_integer("target_dim", dimension)
    _validate_integer("n_samples", n_samples)
    _validate_integer("n_features", n_features)
    if dimension != ds_config["target_dim"]:
        raise ValueError("target_dim does not match the synthetic dataset config")
    name = str(ds_config["name"])
    path = _artifact_path(name, dimension, n_samples)
    generate_command = (f"PYTHONPATH=. python scripts/generate_synthetic.py "
                        f"--target-dim {dimension} --sample-size {n_samples}")

    if not path.exists():
        raise FileNotFoundError(
            f"Synthetic artifact '{name}' was not found at:\n    {path}\n"
            f"The synthetic source does NOT generate data on the fly. Run:\n"
            f"    {generate_command}\n"
            f"Use --force to explicitly replace an incomplete frozen set."
        )

    manifest = _load_manifest(dimension, n_samples)
    if not manifest:
        raise FileNotFoundError(
            f"Synthetic manifest is missing for '{name}'. Regenerate explicitly:\n"
            f"    {generate_command} --force"
        )
    metadata = manifest.get("_meta", {})
    if (metadata.get("target_dim") != dimension
            or metadata.get("sample_size") != n_samples):
        raise ValueError("synthetic manifest shape does not match the requested dataset")
    entry = manifest.get(name)
    if not isinstance(entry, dict) or "sha256" not in entry or "vine" not in entry:
        raise ValueError(f"synthetic manifest has no complete entry for '{name}'")
    if entry.get("config") != ds_config:
        raise ValueError(f"synthetic manifest config mismatch for '{name}'")
    if _sha256(path) != entry["sha256"]:
        raise ValueError(f"synthetic artifact '{name}' sha256 mismatch; refusing modified data")

    frame = pd.read_parquet(path)
    expected_columns = ([f"{_FEATURE_PREFIX}{index}" for index in range(n_features)]
                        + [f"{_TARGET_PREFIX}{index}" for index in range(dimension)])
    if frame.shape != (n_samples, n_features + dimension) or list(frame.columns) != expected_columns:
        raise ValueError(f"synthetic artifact '{name}' shape or columns do not match its config")
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError(f"synthetic artifact '{name}' contains non-finite values")
    return _split_xy(frame, dimension)
