"""Integration tests for screened pools and disjoint curation/evaluation."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from multivariate_screening import screening, decisions as pools
from multivariate_screening.ci_screen import curation_split
from scoringbench.multivariate.sources import get_source


def test_screening_only_sees_curation_and_its_imputation(monkeypatch):
    features = np.arange(600, dtype=float).reshape(300, 2)
    targets = features.copy()
    curation = curation_split(300)
    features[curation[0], 0] = np.nan
    seen = []
    monkeypatch.setattr(screening, "leaky_features", lambda features, targets: np.zeros(features.shape[1], bool))
    targets[:, 1] = np.sin(np.arange(300))
    monkeypatch.setattr(screening, "screen_kci",
                        lambda features, targets, **kw: seen.append((features, targets)) or {})
    result, benchmark = screening.screen_dataset(features, targets)
    assert len(benchmark) == 240 and not np.intersect1d(curation, benchmark).size
    np.testing.assert_array_equal(seen[0][1], targets[curation])
    assert seen[0][0][0, 0] == np.nanmedian(features[curation, 0])
    assert result["n_curation"] == 60
    changed = features.copy()
    changed[benchmark] = 1e12
    screening.screen_dataset(changed, targets)
    np.testing.assert_array_equal(seen[0][0], seen[1][0])


@pytest.mark.parametrize("dim,count", [(2, 51), (3, 44), (5, 28), (10, 7)])
def test_native_enumeration(dim, count):
    datasets = get_source("scoringbench").enumerate_datasets(dim, 3000)
    assert len(datasets) == count


def test_screen_candidate_records_and_persists_decision(monkeypatch, tmp_path):
    features = pd.DataFrame(np.arange(200).reshape(100, 2))
    targets = features.copy()
    decision = dict(selected=False, scale_ablation=True, reason="scale_ratio", scale_ratio=501.0)
    seen = {}

    def load(config, **kw):
        seen.update(config)
        return features, targets

    # screen_candidate resolves the loader through the source registry
    # (get_source(...).load), so patch the registry lookup in decisions' namespace.
    monkeypatch.setattr(pools, "get_source", lambda name: SimpleNamespace(load=load))
    monkeypatch.setattr(pools, "screen_dataset", lambda *args, **kwargs: (decision.copy(), np.arange(20, 100)))
    entry = get_source("scoringbench").enumerate_datasets(2, 3000)[0]
    result = pools.screen_candidate(entry, target_dim=2)
    assert seen["screen_protocol"] == pools.PROTOCOL  # loader skips feature imputation
    assert entry["screen"] is result and result["reason"] == "scale_ratio"
    assert result["protocol"] == pools.PROTOCOL
    pools.save_screen(entry, tmp_path)
    saved = json.loads(next((tmp_path / "screening").glob("*.json")).read_text())
    assert saved["screen"]["selected"] is False


def test_missing_promoted_targets_do_not_move_the_split(monkeypatch):
    features = np.arange(600, dtype=float).reshape(300, 2)
    targets = features.copy()
    curation = curation_split(300)
    missing = curation[0]
    targets[missing, 0] = np.nan
    seen = []
    monkeypatch.setattr(screening, "leaky_features", lambda features, targets: np.zeros(features.shape[1], bool))
    targets[:, 1] = np.sin(np.arange(300))
    monkeypatch.setattr(screening, "screen_kci",
                        lambda features, targets, **kw: seen.append(targets) or {})
    result, benchmark = screening.screen_dataset(features, targets)
    assert result["n_missing_targets"] == 1
    assert result["n_curation"] == 59
    assert not np.intersect1d(curation, benchmark).size
    np.testing.assert_array_equal(seen[0], targets[curation[1:]])


def test_duplicate_rows_never_cross_curation_and_benchmark(monkeypatch):
    rng = np.random.default_rng(23)
    features, targets = rng.normal(size=(300, 3)), rng.normal(size=(300, 2))
    features = np.r_[features, features[:100]]
    targets = np.r_[targets, targets[:100]]
    monkeypatch.setattr(screening, "screen_kci", lambda *args, **kwargs: {})
    result, benchmark = screening.screen_dataset(features, targets)
    assert not set(result["curation_indices"]) & set(benchmark)
    assert len(benchmark) == 320
    assert len(result["curation_indices"]) == 80
    assert len(benchmark) + len(result["curation_indices"]) == 400


@pytest.mark.parametrize("kind", ["ordinal", "categorical"])
def test_noncontinuous_targets_are_rejected_before_kci(monkeypatch, kind):
    rng = np.random.default_rng(29)
    targets = pd.DataFrame(rng.normal(size=(300, 2)))
    if kind == "ordinal":
        targets[0] = np.arange(300) % 5
    elif kind == "categorical":
        targets[0] = pd.Categorical(np.arange(300))
    monkeypatch.setattr(screening, "screen_kci", lambda *args, **kwargs: pytest.fail("KCI called"))
    result, _ = screening.screen_dataset(rng.normal(size=(300, 2)), targets)
    assert result["reason"] == "noncontinuous_target"


def test_atom_mass_targets_are_permitted_and_proceed_to_kci(monkeypatch):
    rng = np.random.default_rng(29)
    targets = pd.DataFrame(rng.normal(size=(300, 2)))
    targets.loc[:100, 0] = 0
    kci_called = False
    def mock_kci(*args, **kwargs):
        nonlocal kci_called
        kci_called = True
        return dict(selected=True, scale_ablation=False, kci_p=0.001, residual_dcor=0.5, scale_ratio=1.0)
    monkeypatch.setattr(screening, "screen_kci", mock_kci)
    result, _ = screening.screen_dataset(rng.normal(size=(300, 2)), targets)
    assert kci_called
    assert result["has_atoms"]
    assert result["selected"]


@pytest.mark.parametrize("kind", ["copy", "affine", "sum"])
def test_target_leakage_is_rejected_before_kci(monkeypatch, kind):
    rng = np.random.default_rng(31)
    targets = rng.normal(size=(300, 2))
    leaked = {"copy": targets[:, 0], "affine": 3 * targets[:, 0] + 4,
              "sum": targets.sum(axis=1)}[kind]
    features = pd.DataFrame({"safe": rng.normal(size=300), "leaked": leaked})
    monkeypatch.setattr(screening, "screen_kci", lambda *args, **kwargs: pytest.fail("KCI called"))
    result, _ = screening.screen_dataset(features, targets)
    assert result["reason"] == "target_leakage"
    assert result["leaky_features"] == ["leaked"]


def test_duplicate_targets_are_rejected(monkeypatch):
    rng = np.random.default_rng(37)
    target = rng.normal(size=300)
    monkeypatch.setattr(screening, "screen_kci", lambda *args, **kwargs: pytest.fail("KCI called"))
    result, _ = screening.screen_dataset(rng.normal(size=(300, 3)), np.c_[target, target])
    assert result["reason"] == "redundant_targets"


def test_holm_corrects_across_tasks_and_dimensions():
    assert pools.holm_pvalues([0.01, 0.04, 0.03, 1.0]) == pytest.approx([0.04, 0.09, 0.09, 1.0])


def test_freezing_excludes_duplicates_and_sets_selected():
    entries = [dict(name=f"toy_{index}", target_dim=2, screen=dict(
        kci_p=0.001, selected=True, scale_ablation=False, fingerprint=f"data_{index}",
        reason="selected")) for index in range(2)]
    entries.append(dict(name="toy_dup", target_dim=2, screen=dict(
        kci_p=0.001, selected=True, scale_ablation=False, fingerprint="data_0",
        reason="selected")))
    payload = pools.finalize_suite(entries)
    assert [record["screen"]["reason"] for record in payload["datasets"]] == [
        "selected", "selected", "duplicate_dataset"]


def test_categorical_target_metadata_survives_float_coercion(monkeypatch):
    rng = np.random.default_rng(38)
    targets = pd.DataFrame(rng.normal(size=(300, 2)))
    targets.attrs["categorical_targets"] = ["original_category"]
    monkeypatch.setattr(screening, "screen_kci", lambda *args, **kwargs: pytest.fail("KCI called"))
    result, _ = screening.screen_dataset(rng.normal(size=(300, 3)), targets)
    assert result["reason"] == "noncontinuous_target"
