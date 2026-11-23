"""Frozen, pool-corrected KCI suite and explicit candidate diagnostics."""

from pathlib import Path
import json
import hashlib
from importlib.metadata import version

import numpy as np

from . import dheur_targets, native_scoringbench
from .ci_screen import ALPHA, EFFECT_MIN, SCALE_RATIO_MAX
from .screening import screen_dataset

PROTOCOL = "kci_continuous_audited_v3"
TARGET_DIMS = (2, 3, 5, 10)
SUITE_PATH = Path(__file__).with_name("multivariate_suite.json")
KNOWN_NONCONTINUOUS = {
    4531: "motor_UPDRS and total_UPDRS are clinical rating scores",
    41492: "personality targets are questionnaire ratings",
    41487: "views, votes and comments are counts",
    42123: "citations and papers are counts",
    43642: "magNst and nst are station counts",
    46123: "pedestrian targets are counts",
    46241: "pedestrian targets are counts",
}
DUPLICATE_FAMILIES = {41484: 41483, 45016: 42105, 46241: 46123}


def definition(config: dict) -> dict:
    """Data-selection fields that must agree with a frozen decision."""
    return {key: config.get(key) for key in (
        "source", "id", "target_dim", "target_cols", "all_target_cols", "feature_cols", "drop_cols")}


def holm_pvalues(pvalues) -> list[float]:
    values = np.asarray(pvalues, float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be finite and between zero and one")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(len(values))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(
        values[order] * np.arange(len(values), 0, -1)))
    return adjusted.tolist()


def finalize_suite(entries: list[dict]) -> dict:
    """Holm over candidates within each target dimension family."""
    seen = {}
    dims = sorted(set(entry["target_dim"] for entry in entries))
    for dimension in dims:
        dim_entries = [e for e in entries if e["target_dim"] == dimension]
        adjusted = holm_pvalues([entry["screen"].get("kci_p")
                                if entry["screen"].get("kci_p") is not None else 1.0
                                for entry in dim_entries])
        for entry, pvalue in zip(dim_entries, adjusted):
            result = entry["screen"]
            result.update(holm_p=pvalue, family_size=len(dim_entries))
            fingerprint = result.get("fingerprint")
            key = (entry["target_dim"], fingerprint)
            duplicate = seen.get(key) if fingerprint else None
            if fingerprint and not duplicate:
                seen[key] = entry["name"]
            if duplicate:
                result.update(selected=False, scale_ablation=False,
                              reason="duplicate_dataset", duplicate_of=duplicate)
    records = [dict(name=entry["name"], definition=definition(entry), screen=entry["screen"])
               for entry in entries]
    payload = dict(protocol=PROTOCOL, target_dims=list(TARGET_DIMS), alpha=ALPHA,
                   effect_min=EFFECT_MIN,
                   scale_ratio_max=SCALE_RATIO_MAX, correction="holm_per_dimension",
                   packages={name: version(name) for name in ("causal-learn", "numpy", "pandas", "scipy")},
                   datasets=records)
    payload["suite_id"] = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return payload

def enumerate_screened(target_dim: int, sample_size: int) -> list[dict]:
    """Candidate configurations for the screened scoringbench suite."""
    datasets = native_scoringbench.enumerate_native_scoringbench(target_dim, sample_size)
    for entry in datasets:
        entry.update(selection="selected", screen_protocol=PROTOCOL)
    return datasets


def load_screened(ds_config: dict, target_dim=None):
    """Replay a frozen decision, or explicitly run candidate curation."""
    loaders = {
        "scoringbench": native_scoringbench.load_native_scoringbench,
        "dheur": dheur_targets.load_dheur,
    }
    ds_config.pop("screen", None)
    features, targets = loaders[ds_config.get("source", "scoringbench")](ds_config, target_dim=target_dim)
    if ds_config.get("screen_only"):
        dataset_id = ds_config.get("openml_id")
        decision, benchmark = screen_dataset(features, targets,
            name=ds_config.get("name"),
            run_test=not (ds_config.get("quality_only", False)
                          or dataset_id in KNOWN_NONCONTINUOUS
                          or dataset_id in DUPLICATE_FAMILIES))
        if dataset_id in KNOWN_NONCONTINUOUS:
            decision.update(selected=False, scale_ablation=False, kci_p=None,
                            reason="noncontinuous_target", is_ordinal=True, semantic_note=KNOWN_NONCONTINUOUS[dataset_id])
        if dataset_id in DUPLICATE_FAMILIES:
            decision.update(selected=False, scale_ablation=False, kci_p=None,
                            reason="duplicate_family", is_duplicate=True, duplicate_of=DUPLICATE_FAMILIES[dataset_id])
    else:
        path = Path(ds_config.get("suite_path", SUITE_PATH))
        if not path.exists():
            raise ValueError("Frozen suite missing; run scripts/screen_multivariate.py --freeze first")
        payload = json.loads(path.read_text())
        if payload.get("protocol") != PROTOCOL:
            raise ValueError("Frozen suite uses a different screening protocol")
        record = next((item for item in payload["datasets"]
                       if item["definition"] == definition(ds_config)), None)
        if record is None:
            raise ValueError("Dataset definition is absent from the frozen suite")
        decision = record["screen"].copy()
        decision["suite_id"] = payload["suite_id"]
        ds_config["screen"] = decision
        if not decision.get(ds_config["selection"]):
            raise ValueError(f"{ds_config['name']}: excluded from frozen suite ({decision['reason']})")
        audit, benchmark = screen_dataset(features, targets, name=ds_config.get("name"), run_test=False)
        if (audit["reason"] != "quality_checked"
                or any(audit[key] != decision[key] for key in (
                    "fingerprint", "curation_fingerprint", "n_benchmark"))):
            raise ValueError("Dataset or curation split changed after freezing; refuse evaluation")
    decision["protocol"] = PROTOCOL
    decision["thresholds"] = dict(alpha=ALPHA, effect_min=EFFECT_MIN,
                                  scale_ratio_max=SCALE_RATIO_MAX)
    ds_config["screen"] = decision
    selection = ds_config.get("selection", "selected")
    if not decision.get(selection, False):
        raise ValueError(f"{ds_config['name']}: excluded from {selection} "
                         f"({decision['reason']}, scale_ratio={decision['scale_ratio']})")
    return (features.iloc[benchmark].reset_index(drop=True),
            targets.iloc[benchmark].reset_index(drop=True))


def save_screen(ds_config: dict, output_dir: Path) -> None:
    """Persist accepted and rejected decisions independently of model results."""
    if not ds_config.get("screen"):
        return
    directory = Path(output_dir) / "screening"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / (ds_config["name"].replace(" ", "_") + ".json")
    payload = dict(dataset=ds_config["name"], source=ds_config["source"],
                   target_dim=ds_config["target_dim"], selection=ds_config.get("selection", "selected"),
                   dataset_id=ds_config.get("id"),
                   target_cols=ds_config.get("target_cols"),
                   screen=ds_config["screen"])
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(destination)