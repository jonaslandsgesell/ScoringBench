"""Re-run the model-free KCI screen over a manifest (default: the shipped suite). No benchmark scores."""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import sys

_HERE = Path(__file__).resolve()
sys.path[:0] = [str(_HERE.parents[2]), str(_HERE.parents[1])]  # repo root, scripts/
RESULTS = _HERE.parent / "results"

from multivariate_screening.decisions import (
    PROTOCOL, TARGET_DIMS, finalize_suite, save_screen, screen_candidate,
)
from scoringbench.multivariate.sources import SOURCES, get_source


def run_screen(manifest: Path | None, target_dims, output_dir: Path,
               quality_only: bool = False, dataset_index: int | None = None,
               source: str = "scoringbench") -> int:
    """Screen every manifest entry at each target dimension; write decisions + report.

    ``source`` selects which registered multivariate source provides the datasets
    (``scoringbench``/OpenML by default). ``manifest`` overrides that source's
    default manifest when given.

    Returns the number of candidates that errored while loading or testing.
    """
    enumerate_datasets = get_source(source).enumerate_datasets
    errors = 0
    entries = []
    for dimension in target_dims:
        datasets = (enumerate_datasets(dimension, 3000, manifest_path=manifest)
                    if manifest is not None else enumerate_datasets(dimension, 3000))
        if dataset_index is not None:
            if not 0 <= dataset_index < len(datasets):
                raise ValueError("dataset_index is outside the candidate list")
            datasets = [datasets[dataset_index]]
        counts = Counter()
        for entry in datasets:
            try:
                screen_candidate(entry, target_dim=dimension, quality_only=quality_only)
            except Exception as exc:
                if not entry.get("screen"):
                    errors += 1
                    entry["screen"] = dict(selected=False, scale_ablation=False,
                                           reason="error", error=f"{type(exc).__name__}: {exc}")
            gc.collect()
            entries.append(entry)
            decision = entry["screen"]
            counts[decision["reason"]] += 1
            print(f"{entry['name']}: {decision['reason']}; "
                  f"n={decision.get('n_curation')} pX={decision.get('n_features')} "
                  f"p={decision.get('kci_p')} ordinal={decision.get('is_ordinal', False)}", flush=True)
        print(f"d={dimension}: {dict(counts)}", flush=True)
    payload = finalize_suite(entries)  # Holm per dimension + duplicate detection
    payload["quality_only"] = quality_only
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "suite_draft.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    lines = [f"# Multivariate Suite: {PROTOCOL}", "", "No benchmark-model scores enter selection.", "",
             "| d | Candidates | Selected (Core) | Scale Ablation | Rejected |", "| --- | --- | --- | --- | --- |"]
    for dimension in target_dims:
        decisions = [entry["screen"] for entry in entries if entry["target_dim"] == dimension]
        n_cand = len(decisions)
        n_sel = sum(bool(item.get("selected")) for item in decisions)
        n_scale = sum(bool(item.get("scale_ablation")) for item in decisions)
        lines.append(f"| {dimension} | {n_cand} | {n_sel} | {n_scale} | {n_cand - n_sel - n_scale} |")
    lines += ["", "## Decisions", "",
              "| Dataset | d | Decision | KCI p | Holm p | Residual dcorr | Scale Ratio | Ordinal | Duplicate Of |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for entry in entries:
        decision = entry["screen"]
        lines.append(f"| {entry['name']} | {entry['target_dim']} | {decision['reason']} | "
                     f"{decision.get('kci_p')} | {decision.get('holm_p')} | {decision.get('residual_dcor')} | "
                     f"{decision.get('scale_ratio')} | {decision.get('is_ordinal', False)} | {decision.get('duplicate_of', '-')} |")
        # Saved after finalize_suite so the per-dataset file carries Holm p / duplicate status.
        save_screen(entry, output_dir / f"d{entry['target_dim']}")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=sorted(SOURCES), default="scoringbench",
                        help="Multivariate source to screen (default: 'scoringbench', OpenML). "
                             "Each registered source uses its own default manifest unless "
                             "--manifest is given.")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Manifest to screen (schema of the source's dataset manifest). "
                             "Default: the source's shipped manifest, i.e. re-check its suite.")
    parser.add_argument("--target_dims", nargs="+", type=int, choices=TARGET_DIMS, default=[2],
                        help="Target dimensions to KCI-screen. Default: d=2 only. "
                             "Pass e.g. --target_dims 2 3 5 10 to screen higher d.")
    parser.add_argument("--dataset_index", type=int)
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="Default: results/<protocol>/manifest (scoringbench) "
                             "or results/<protocol>/<source> for other sources.")
    parser.add_argument("--quality_only", action="store_true",
                        help="Only run the data-quality gates, skip the KCI test.")
    args = parser.parse_args()
    output_dir = args.output_dir or (
        RESULTS / PROTOCOL / ("manifest" if args.source == "scoringbench" else args.source))
    errors = run_screen(args.manifest, args.target_dims, output_dir,
                        args.quality_only, args.dataset_index, args.source)
    return int(errors > 0)


if __name__ == "__main__":
    raise SystemExit(main())