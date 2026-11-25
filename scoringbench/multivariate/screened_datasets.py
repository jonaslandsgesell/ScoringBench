"""Model-free KCI screening of candidate multivariate datasets.

Used by ``scripts/screen_multivariate.py`` to (re-)derive which manifest entries
pass the dependence screen. Benchmark runs read ``multivariate_datasets.json``
directly and do not call into this module.
"""

from pathlib import Path
import json
import hashlib
from importlib.metadata import version

import numpy as np

from . import native_scoringbench
from .ci_screen import ALPHA, EFFECT_MIN, SCALE_RATIO_MAX
from .screening import screen_dataset

PROTOCOL = "kci_continuous_audited_v3"
TARGET_DIMS = (2, 3, 5, 10)
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

def screen_candidate(ds_config: dict, target_dim=None, quality_only: bool = False) -> dict:
    """Run the model-free KCI screen on one candidate and record the decision.

    The data are loaded without feature imputation (``screen_protocol``):
    :func:`screen_dataset` imputes on the curation split only, so the benchmark
    rows never influence the decision. The decision is stored in
    ``ds_config["screen"]`` and returned.
    """
    ds_config.pop("screen", None)
    ds_config["screen_protocol"] = PROTOCOL
    features, targets = native_scoringbench.load_native_scoringbench(ds_config, target_dim=target_dim)
    dataset_id = ds_config.get("openml_id")
    decision, _ = screen_dataset(features, targets, name=ds_config.get("name"),
                                 run_test=not (quality_only
                                               or dataset_id in KNOWN_NONCONTINUOUS
                                               or dataset_id in DUPLICATE_FAMILIES))
    if dataset_id in KNOWN_NONCONTINUOUS:
        decision.update(selected=False, scale_ablation=False, kci_p=None,
                        reason="noncontinuous_target", is_ordinal=True,
                        semantic_note=KNOWN_NONCONTINUOUS[dataset_id])
    if dataset_id in DUPLICATE_FAMILIES:
        decision.update(selected=False, scale_ablation=False, kci_p=None,
                        reason="duplicate_family", is_duplicate=True,
                        duplicate_of=DUPLICATE_FAMILIES[dataset_id])
    decision["protocol"] = PROTOCOL
    decision["thresholds"] = dict(alpha=ALPHA, effect_min=EFFECT_MIN,
                                  scale_ratio_max=SCALE_RATIO_MAX)
    ds_config["screen"] = decision
    return decision


def save_screen(ds_config: dict, output_dir: Path) -> None:
    """Persist accepted and rejected decisions independently of model results."""
    if not ds_config.get("screen"):
        return
    directory = Path(output_dir) / "screening"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / (ds_config["name"].replace(" ", "_") + ".json")
    payload = dict(dataset=ds_config["name"], source=ds_config["source"],
                   target_dim=ds_config["target_dim"],
                   dataset_id=ds_config.get("id"),
                   target_cols=ds_config.get("target_cols"),
                   screen=ds_config["screen"])
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(destination)