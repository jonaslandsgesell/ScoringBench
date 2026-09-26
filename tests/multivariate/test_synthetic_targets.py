"""Check the randomized DGP, not a preferred ordering of fitted estimators."""

from __future__ import annotations

from collections import Counter
import json
import sys

import numpy as np
import pandas as pd
import pyvinecopulib as pv
import pytest
from scipy.stats import kendalltau
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from scripts import generate_synthetic
from scoringbench.multivariate import config as cfg
from scoringbench.multivariate import synthetic_targets as st
from scoringbench.multivariate.metrics import compute_scoring_rules
from scoringbench.multivariate.prediction import MultivariateSamplePrediction
from scoringbench.multivariate.runner import run_benchmark
from scoringbench.multivariate.sources import get_source
from scoringbench.multivariate.wrappers import (
    ChainedMultiOutputWrapper,
    CopulaMultiOutputWrapper,
    IndependentMultiOutputWrapper,
    TabPFNSampler,
)


def _tabpfn_sampler() -> TabPFNSampler:
    """Real TabPFN per-dimension sampler on the fast v3.5 model.

    Emits TabPFN's native per-row bar distribution, so the guard reflects real
    benchmark performance rather than a stand-in. Uses the ``v3.5-fast`` model
    version to keep the (chained) guard cheap on GPU. ``TabPFNSampler`` picks
    the ``cpu`` device automatically when CUDA is unavailable.
    """
    return TabPFNSampler(
        model_version="v3.5-fast",
        ignore_pretraining_limits=True,
    )


@pytest.mark.parametrize("dependence_type", ["tail", "no_tail", "mixed"])
def test_fitted_chain_learns_feature_dependent_joint_law(dependence_type):
    # Small, low-dimensional problem keeps the (chained) guard fast on GPU: the
    # chained wrapper costs ``n_orders x d`` forward passes per draw, so d=3 with
    # a modest train/test split and a light draw budget runs in ~10-16s per
    # dependence type while the chaining edge stays comfortably above threshold.
    config = next(entry for entry in st.enumerate_synthetic(3, 400)
                  if entry["dependence_type"] == dependence_type)
    features, targets = st._generate(config)
    train, test = train_test_split(np.arange(len(features)), test_size=0.3, random_state=0)
    scores = {}
    with threadpool_limits(limits=1):
        for label, wrapper in (
            ("independent", IndependentMultiOutputWrapper),
            ("copula", CopulaMultiOutputWrapper),
            ("chained", ChainedMultiOutputWrapper),
        ):
            model = wrapper(sampler_factory=_tabpfn_sampler, n_draws=40, seed=42)
            model.fit(features.iloc[train], targets.iloc[train])
            predictions = model.predict_ensemble(features.iloc[test])
            scores[label] = compute_scoring_rules(predictions, targets.iloc[test].to_numpy())
    # The energy score is structurally near-insensitive to dependence (an oracle
    # forecast that knows the true conditional law only earns single-digit gains
    # over the independent baseline for these bimodal latents), so we require a
    # modest 5% edge on it. The variogram score is dependence-sensitive and must
    # beat both baselines by 10%. Chaining must win on both metrics against both
    # baselines.
    metric_thresholds = {"energy_score_beta_1": 0.95, "variogram_score_p_1": 0.90}
    for baseline in ("independent", "copula"):
        for metric, factor in metric_thresholds.items():
            assert scores["chained"][metric] < factor * scores[baseline][metric], (
                baseline, metric, scores["chained"][metric], scores[baseline][metric],
            )


def test_enumerate_is_deterministic_and_balanced():
    configs = st.enumerate_synthetic(target_dim=3, sample_size=500)
    assert configs == st.enumerate_synthetic(target_dim=3, sample_size=500)
    assert len(configs) == cfg.SYNTHETIC_N_DATASETS
    per_cell = Counter((entry["dependence_type"], entry["dependence_strength"])
                       for entry in configs)
    assert len(per_cell) == 3
    assert {entry["dependence_strength"] for entry in configs} == {"strong"}
    assert max(per_cell.values()) - min(per_cell.values()) <= 1
    assert len({entry["name"] for entry in configs}) == len(configs)
    assert len({entry["seed"] for entry in configs}) == len(configs)
    assert all(entry["source"] == st.SOURCE_NAME for entry in configs)
    assert all(not {"family", "tau", "mean_scale", "noise_scale"}.intersection(entry)
               for entry in configs)


def test_suite_growth_preserves_existing_model_identities(monkeypatch):
    original = st.enumerate_synthetic(3, 100)
    monkeypatch.setattr(cfg, "SYNTHETIC_N_DATASETS", cfg.SYNTHETIC_N_DATASETS + 12)
    expanded = {entry["name"]: entry for entry in st.enumerate_synthetic(3, 100)}
    assert all(expanded[entry["name"]] == entry for entry in original)


def test_scenario_reordering_preserves_model_seeds(monkeypatch):
    original = st.enumerate_synthetic(3, 100)
    monkeypatch.setattr(cfg, "SYNTHETIC_DEPENDENCE_TYPES", ("mixed", "no_tail", "tail"))
    reordered = {entry["name"]: entry for entry in st.enumerate_synthetic(3, 100)}
    for entry in original:
        if entry["name"] in reordered:
            assert reordered[entry["name"]] == entry


def test_sample_size_does_not_change_model_identity():
    small = st.enumerate_synthetic(3, 100)[0]
    large = st.enumerate_synthetic(3, 200)[0]
    assert small["name"] == large["name"]
    assert small["seed"] == large["seed"]
    _, _, small_vine = st._generate_with_vine(small)
    _, _, large_vine = st._generate_with_vine(large)
    assert small_vine.to_json() == large_vine.to_json()


@pytest.mark.parametrize("setting,value", [
    ("SYNTHETIC_DECAY", 0.6), ("SYNTHETIC_TAU_UPPER", 0.7),
    ("SYNTHETIC_N_FEATURES", 3), ("SEED", 100),
])
def test_model_changes_get_new_identities(monkeypatch, setting, value):
    original = st.enumerate_synthetic(3, 100)[0]
    monkeypatch.setattr(cfg, setting, value)
    assert st.enumerate_synthetic(3, 100)[0]["name"] != original["name"]


@pytest.mark.parametrize("setting,value", [
    ("_MEAN_N_FEATURES", 2), ("_MEAN_N_CALIBRATION", 1024),
    ("_MEAN_N_RIDGE", 8), ("_MEAN_N_INTERACTIONS", 4),
    ("_MEAN_SIGNAL_FRACTION", 0.5), ("_MEAN_RIDGE_ACTIVATIONS", ("square",)),
])
def test_mean_changes_get_new_identities(monkeypatch, setting, value):
    original = st.enumerate_synthetic(3, 100)[0]
    monkeypatch.setattr(st, setting, value)
    assert st.enumerate_synthetic(3, 100)[0]["name"] != original["name"]


def test_residual_design_cannot_reuse_joint_vine_dataset_identity():
    config = st.enumerate_synthetic(3, 100)[0]
    old_parameters = {key: value for key, value in config.items()
                      if key not in {"name", "id", "sampling", "nonlinear_mean"}}
    assert st._dataset_name(old_parameters) != config["name"]


@pytest.mark.parametrize("dimension", [1, 2, 3, 8])
def test_generate_shape_and_determinism(dimension):
    config = st.enumerate_synthetic(dimension, 300)[0]
    features, targets = st._generate(config, target_dim=dimension)
    repeated_features, repeated_targets = st._generate(config)
    assert features.shape == (300, cfg.SYNTHETIC_N_FEATURES)
    assert targets.shape == (300, dimension)
    assert list(features.columns) == [f"feature_{index}" for index in range(features.shape[1])]
    assert list(targets.columns) == [f"target_{index}" for index in range(dimension)]
    assert np.isfinite(features.values).all()
    assert np.isfinite(targets.values).all()
    np.testing.assert_array_equal(features.values, repeated_features.values)
    np.testing.assert_array_equal(targets.values, repeated_targets.values)


def test_replicates_randomize_structure_and_observations():
    configs = st.enumerate_synthetic(8, 100)
    first_features, first_targets, first_vine = st._generate_with_vine(configs[0])
    second_features, second_targets, second_vine = st._generate_with_vine(configs[1])
    assert first_vine.structure.to_json() != second_vine.structure.to_json()
    assert not np.array_equal(first_features.values, second_features.values)
    assert not np.array_equal(first_targets.values, second_targets.values)


@pytest.mark.parametrize("dependence_type,allowed", [
    ("tail", {"student", "clayton", "gumbel"}),
    ("no_tail", {"gaussian", "frank"}),
    ("mixed", {"student", "clayton", "gumbel", "gaussian", "frank"}),
])
@pytest.mark.parametrize("strength", ["weak", "strong"])
def test_random_vine_families_signs_and_tree_decay(dependence_type, allowed, strength):
    vine = st.simulate_random_vine(10, dependence_type=dependence_type,
                                  dependence_strength=strength)
    pairs = [pair for tree in vine.pair_copulas for pair in tree]
    assert {pair.family.name for pair in pairs} == allowed
    assert min(pair.tau for pair in pairs) < 0 < max(pair.tau for pair in pairs)
    assert len({round(pair.tau, 8) for pair in pairs}) == len(pairs)
    for level, tree in enumerate(vine.pair_copulas, start=1):
        assert len(tree) == 10 - level
        assert all(abs(pair.tau) <= 0.8 ** level + 1e-10 for pair in tree)
        for pair in tree:
            if pair.family.name == "student":
                assert pair.parameters[1, 0] == 4.0
            if pair.family.name in ("clayton", "gumbel"):
                assert (pair.tau < 0) == (pair.rotation in (90, 270))


@pytest.mark.parametrize("strength,expected_mean", [("weak", 0.2), ("strong", 0.5)])
def test_tau_distribution_matches_paper(strength, expected_mean):
    normalized = []
    for seed in range(40):
        vine = st.simulate_random_vine(8, seed=seed, dependence_strength=strength)
        normalized.extend(abs(pair.tau) / 0.8 ** level
                          for level, tree in enumerate(vine.pair_copulas, start=1)
                          for pair in tree)
    assert np.mean(normalized) == pytest.approx(expected_mean, abs=0.02)
    assert min(normalized) < 0.2


def test_mixed_regime_balances_groups_not_individual_families():
    counts = Counter()
    rotations = Counter()
    signs = Counter()
    for seed in range(40):
        vine = st.simulate_random_vine(8, seed=seed)
        for tree in vine.pair_copulas:
            for pair in tree:
                counts[pair.family.name] += 1
                signs[pair.tau > 0] += 1
                if pair.family.name in ("clayton", "gumbel"):
                    rotations[pair.rotation] += 1
    total = sum(counts.values())
    for family in ("student", "clayton", "gumbel"):
        assert counts[family] / total == pytest.approx(1 / 6, abs=0.04)
    for family in ("gaussian", "frank"):
        assert counts[family] / total == pytest.approx(1 / 4, abs=0.04)
    assert signs[True] / total == pytest.approx(0.5, abs=0.05)
    assert set(rotations) == {0, 90, 180, 270}


def test_zero_decay_is_independence_without_a_tau_floor():
    vine = st.simulate_random_vine(4, decay=0)
    assert all(pair.family.name == "indep" for tree in vine.pair_copulas for pair in tree)
    small = st.simulate_random_vine(4, upper=0.01)
    assert all(abs(pair.tau) <= 0.008 + 1e-10
               for tree in small.pair_copulas for pair in tree)


@pytest.mark.parametrize("dimension", [1, 2, 8])
def test_standalone_samples_have_normal_margins(dimension):
    values = st.simulate_synthetic_data(6000, dimension)
    assert values.shape == (6000, dimension)
    assert values.dtype == np.float64
    assert np.isfinite(values).all()
    np.testing.assert_allclose(values.mean(axis=0), 0, atol=0.06)
    np.testing.assert_allclose(values.std(axis=0), 1, atol=0.05)
    np.testing.assert_array_equal(values, st.simulate_synthetic_data(6000, dimension))


@pytest.mark.parametrize("dependence_type", ["tail", "no_tail", "mixed"])
def test_dataset_is_built_from_recorded_streams(dependence_type):
    config = next(entry for entry in st.enumerate_synthetic(3, 256)
                  if entry["dependence_type"] == dependence_type)
    features, targets, vine = st._generate_with_vine(config)
    assert vine.dim == config["target_dim"]
    streams = st._spawn_streams(config["seed"])
    expected_features = streams["features"].standard_normal(
        (config["n_samples"], config["n_features"]),
    )
    innovations = st._sample_normal_data(vine, config["n_samples"], streams["copula_sim"])
    shared_noise = st._sample_shared_noise(config["n_samples"], streams["shared_noise"])
    expected_targets = st._generate_targets(
        expected_features, innovations, shared_noise, config["seed"],
    )
    np.testing.assert_array_equal(features.values, expected_features)
    np.testing.assert_array_equal(targets.values, expected_targets)


def test_nonlinear_mean_is_even_and_independent_of_evaluation_batch():
    features = np.random.default_rng(31).standard_normal((120, 20))
    means = st._nonlinear_mean(features, np.random.default_rng(42))
    reflected = st._nonlinear_mean(-features, np.random.default_rng(42))
    partitioned = np.concatenate([
        st._nonlinear_mean(batch, np.random.default_rng(42))
        for batch in np.array_split(features, 3)
    ])
    np.testing.assert_allclose(means, reflected, atol=1e-12)
    np.testing.assert_allclose(means, partitioned, atol=1e-12)
    assert np.std(means) > 0.5


@pytest.mark.parametrize("dependence_type", ["tail", "no_tail", "mixed"])
def test_nonlinear_targets_are_learnable_but_not_by_linear_models(dependence_type):
    config = next(entry for entry in st.enumerate_synthetic(3, 1000)
                  if entry["dependence_type"] == dependence_type)
    features, targets = st._generate(config)
    train, test = train_test_split(np.arange(len(features)), random_state=0)
    with threadpool_limits(limits=1):
        linear = LinearRegression().fit(features.iloc[train], targets.iloc[train])
        nonlinear_predictions = np.column_stack([
            HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=15, random_state=0)
            .fit(features.iloc[train], targets.iloc[train, column])
            .predict(features.iloc[test])
            for column in range(targets.shape[1])
        ])
        linear_r2 = r2_score(targets.iloc[test], linear.predict(features.iloc[test]))
    nonlinear_r2 = r2_score(targets.iloc[test], nonlinear_predictions)
    assert linear_r2 < 0.05
    assert nonlinear_r2 > 0.3
    assert nonlinear_r2 - linear_r2 > 0.25


@pytest.mark.parametrize("dimension", [3, 8])
def test_shared_noise_dependence_switches_with_features(dimension):
    config = st.enumerate_synthetic(dimension, 4000)[0]
    features, _, vine = st._generate_with_vine(config)
    features = features.to_numpy()
    rng = np.random.default_rng(5)
    innovations = st._sample_normal_data(vine, len(features), rng)
    shared_noise = st._sample_shared_noise(len(features), rng)
    shared_part = (st._generate_targets(features, innovations, shared_noise, config["seed"])
                   - st._generate_targets(features, innovations, np.zeros_like(shared_noise),
                                          config["seed"]))
    magnitudes = np.abs(shared_part)
    np.testing.assert_allclose(magnitudes, np.repeat(magnitudes[:, :1], dimension, axis=1),
                               atol=1e-12)
    pooled = np.corrcoef(shared_part, rowvar=False)[np.triu_indices(dimension, 1)]
    assert np.max(np.abs(pooled)) < 0.1
    signs = np.sign(shared_part[:, 1:] * shared_part[:, :1])
    assert all(0.3 < np.mean(column > 0) < 0.7 for column in signs.T)


@pytest.mark.parametrize("dimension", [3, 8])
@pytest.mark.parametrize("dependence_type", ["tail", "no_tail", "mixed"])
def test_oracle_joint_beats_equal_marginal_independence(dimension, dependence_type):
    config = next(entry for entry in st.enumerate_synthetic(dimension, 640)
                  if entry["dependence_type"] == dependence_type)
    features, targets, vine = st._generate_with_vine(config)
    n_draws = 128
    rng = np.random.default_rng(2026)
    repeated = np.repeat(features.to_numpy(), n_draws, axis=0)
    joint = st._generate_targets(
        repeated, st._sample_normal_data(vine, len(repeated), rng),
        st._sample_shared_noise(len(repeated), rng), config["seed"],
    ).reshape(len(features), n_draws, dimension)
    independent = np.empty_like(joint)
    for column in range(dimension):
        independent[:, :, column] = joint[:, rng.permutation(n_draws), column]
    np.testing.assert_array_equal(np.sort(joint, axis=1), np.sort(independent, axis=1))
    scores = {}
    with threadpool_limits(limits=1):
        for label, samples in (("joint", joint), ("independent", independent)):
            scores[label] = pd.DataFrame([
                compute_scoring_rules(
                    MultivariateSamplePrediction(samples=samples[start:start + 32]),
                    targets.iloc[start:start + 32].to_numpy(),
                )
                for start in range(0, len(features), 32)
            ]).mean()
    for metric in ("energy_score_beta_1", "variogram_score_p_1"):
        assert scores["joint"][metric] < 0.90 * scores["independent"][metric], (metric, scores)
    assert scores["joint"]["dawid_sebastiani"] < scores["independent"]["dawid_sebastiani"]


@pytest.mark.parametrize("dependence_type", ["tail", "no_tail", "mixed"])
@pytest.mark.parametrize("strength", ["weak", "strong"])
def test_sample_tau_matches_generating_pair(dependence_type, strength):
    vine = st.simulate_random_vine(2, dependence_type=dependence_type,
                                  dependence_strength=strength)
    observations = st.simulate_synthetic_data(6000, 2, dependence_type=dependence_type,
                                             dependence_strength=strength)
    observed_tau = kendalltau(observations[:, 0], observations[:, 1]).statistic
    assert observed_tau == pytest.approx(vine.pair_copulas[0][0].tau, abs=0.035)


def test_sampler_leaves_global_numpy_rng_untouched():
    before = np.random.get_state()
    st.simulate_synthetic_data(30, 4)
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


@pytest.mark.parametrize("kwargs", [
    {"dim": 0}, {"dim": 1.5}, {"dim": True}, {"seed": -1}, {"seed": 1.5},
    {"decay": -0.1}, {"decay": 1.1}, {"decay": np.nan}, {"decay": np.inf},
    {"upper": 0}, {"upper": 1.1}, {"upper": np.nan}, {"upper": np.inf},
    {"dependence_type": "unknown"}, {"dependence_strength": "unknown"},
])
def test_invalid_vine_parameters_raise(kwargs):
    with pytest.raises(ValueError):
        st.simulate_random_vine(**dict({"dim": 3}, **kwargs))


@pytest.mark.parametrize("n_samples", [0, -1, 1.5, True])
def test_invalid_sample_counts_raise(n_samples):
    with pytest.raises(ValueError, match="n_samples"):
        st.simulate_synthetic_data(n_samples, 2)


@pytest.mark.parametrize("field,value", [
    ("n_samples", 0), ("n_features", 0),
])
def test_invalid_dataset_parameters_raise(field, value):
    config = st.enumerate_synthetic(2, 100)[0]
    config[field] = value
    with pytest.raises(ValueError, match=field):
        st._generate(config)


def test_target_dimension_override_cannot_mislabel_dataset():
    config = st.enumerate_synthetic(2, 100)[0]
    with pytest.raises(ValueError, match="target_dim"):
        st._generate(config, target_dim=3)


@pytest.fixture
def frozen_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "SYNTHETIC_DIR", tmp_path)
    monkeypatch.setattr(cfg, "SYNTHETIC_N_DATASETS", 1)
    monkeypatch.setattr(sys, "argv", ["generate_synthetic.py", "--target-dim", "3",
                                    "--sample-size", "250"])
    generate_synthetic.main()
    return st.enumerate_synthetic(3, 250)[0]


def test_cache_round_trip(frozen_dataset):
    features, targets = st._generate(frozen_dataset)
    loaded_features, loaded_targets = st.load_synthetic(frozen_dataset)
    np.testing.assert_array_equal(features.values, loaded_features.values)
    np.testing.assert_array_equal(targets.values, loaded_targets.values)


def test_synthetic_source_runs_through_benchmark(frozen_dataset, tmp_path):
    source = get_source("synthetic")
    datasets = source.enumerate_datasets(3, 250)
    assert datasets == [frozen_dataset]
    output_dir = tmp_path / "results"
    results = run_benchmark(
        datasets_config=datasets,
        model_factories={
            "tabpfn": lambda: IndependentMultiOutputWrapper(
                sampler_factory=_tabpfn_sampler, n_draws=30, seed=42,
            ),
        },
        output_dir=output_dir, n_folds=2, n_repeats_cv=1,
        seed=42, sample_size=250, target_dim=3, load_fn=source.load,
    )
    assert len(results) == 2
    assert set(results["dataset_source"]) == {"synthetic"}
    assert set(results["n_targets"]) == {3}
    scores = ["energy_score_beta_1", "variogram_score_p_0.5", "dawid_sebastiani"]
    assert np.isfinite(results[scores].to_numpy()).all()
    stored = pd.read_parquet(
        output_dir / "raw" / "tabpfn" / f"{frozen_dataset['name']}.parquet",
    )
    assert set(stored["fold"]) == {0, 1}
    assert np.isfinite(stored[scores].to_numpy()).all()


@pytest.mark.parametrize("dimension", [1, 4])
def test_generator_script_records_realized_models(tmp_path, monkeypatch, dimension):
    monkeypatch.setattr(st, "SYNTHETIC_DIR", tmp_path)
    monkeypatch.setattr(cfg, "SYNTHETIC_N_DATASETS", 6)
    monkeypatch.setattr(sys, "argv", ["generate_synthetic.py", "--target-dim", str(dimension),
                                    "--sample-size", "32"])
    generate_synthetic.main()
    manifest = st._load_manifest(dimension, 32)
    assert manifest["_meta"]["n_datasets"] == 6
    assert manifest["_meta"]["tree_level_start"] == 1
    assert manifest["_meta"]["sampling"] == "feature_gated_shared_latent_with_vine_innovations"
    assert manifest["_meta"]["vine_role"] == "target_specific_innovations"
    assert manifest["_meta"]["target_dependence"]["regimes"] == "hadamard_rows_by_feature_signs"
    assert manifest["_meta"]["nonlinear_mean"]["enabled"] is True
    for config in st.enumerate_synthetic(dimension, 32):
        entry = manifest[config["name"]]
        assert entry["config"] == config
        restored = pv.Vinecop.from_json(json.dumps(entry["vine"]))
        assert restored.dim == dimension
        original = st._generate_with_vine(config)[2]
        assert restored.structure.to_json() == original.structure.to_json()
        for restored_tree, original_tree in zip(restored.pair_copulas, original.pair_copulas):
            for restored_pair, original_pair in zip(restored_tree, original_tree):
                assert restored_pair.family == original_pair.family
                assert restored_pair.rotation == original_pair.rotation
                np.testing.assert_allclose(restored_pair.parameters, original_pair.parameters)
        features, targets = st.load_synthetic(config)
        assert features.shape == (32, cfg.SYNTHETIC_N_FEATURES)
        assert targets.shape == (32, dimension)


def test_generator_requires_explicit_overwrite(frozen_dataset, monkeypatch):
    path = st._artifact_path(frozen_dataset["name"], 3, 250)
    original_sha = st._sha256(path)
    original_manifest = st._manifest_path(3, 250).read_bytes()
    with pytest.raises(FileExistsError, match="--force"):
        generate_synthetic.main()
    assert st._sha256(path) == original_sha
    assert st._manifest_path(3, 250).read_bytes() == original_manifest
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--force"])
    generate_synthetic.main()
    assert st._sha256(path) == original_sha
    st.load_synthetic(frozen_dataset)


def test_load_rejects_missing_manifest(frozen_dataset):
    st._manifest_path(3, 250).unlink()
    with pytest.raises(FileNotFoundError, match="manifest"):
        st.load_synthetic(frozen_dataset)


@pytest.mark.parametrize("damage,match", [
    ("shape", "shape"),
    ("entry", "complete entry"), ("vine", "complete entry"),
    ("config", "config mismatch"), ("checksum", "sha256 mismatch"),
])
def test_load_rejects_invalid_manifest(frozen_dataset, damage, match):
    manifest = st._load_manifest(3, 250)
    name = frozen_dataset["name"]
    if damage == "shape":
        manifest["_meta"]["target_dim"] = 2
    elif damage == "entry":
        del manifest[name]
    elif damage == "vine":
        del manifest[name]["vine"]
    elif damage == "config":
        manifest[name]["config"]["decay"] = 0.5
    else:
        manifest[name]["sha256"] = "not-the-artifact-checksum"
    st._manifest_path(3, 250).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=match):
        st.load_synthetic(frozen_dataset)


@pytest.mark.parametrize("damage,match", [
    ("bytes", "sha256 mismatch"), ("rows", "shape or columns"),
    ("columns", "shape or columns"), ("nan", "non-finite"),
])
def test_load_rejects_altered_artifacts(frozen_dataset, damage, match):
    path = st._artifact_path(frozen_dataset["name"], 3, 250)
    frame = pd.read_parquet(path)
    if damage in ("bytes", "rows"):
        frame = frame.iloc[:-1]
    elif damage == "columns":
        frame = frame.rename(columns={"feature_0": "extra_target"})
    else:
        frame.iloc[0, 0] = np.nan
    frame.to_parquet(path, index=False)
    if damage != "bytes":
        manifest = st._load_manifest(3, 250)
        manifest[frozen_dataset["name"]]["sha256"] = st._sha256(path)
        st._manifest_path(3, 250).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=match):
        st.load_synthetic(frozen_dataset)


def test_load_rejects_mismatched_request(frozen_dataset):
    with pytest.raises(ValueError, match="config mismatch"):
        st.load_synthetic(dict(frozen_dataset, decay=0.5))
    with pytest.raises(ValueError, match="target_dim"):
        st.load_synthetic(frozen_dataset, target_dim=2)


def test_load_raises_on_cache_miss(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "SYNTHETIC_DIR", tmp_path)
    config = st.enumerate_synthetic(2, 120)[0]
    with pytest.raises(FileNotFoundError, match="generate_synthetic.py"):
        st.load_synthetic(config)


def test_artifact_namespace_is_shape_specific(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "SYNTHETIC_DIR", tmp_path)
    config = st.enumerate_synthetic(2, 120)[0]
    features, targets = st._generate(config)
    directory = st._shape_subdir(2, 120)
    assert directory == tmp_path / "d2_n120"
    directory.mkdir(parents=True)
    features.join(targets).to_parquet(st._artifact_path(config["name"], 2, 120), index=False)
    with pytest.raises(FileNotFoundError):
        st.load_synthetic(dict(config, n_samples=250))
