"""Audit and freeze the continuous-target KCI suite without benchmark scores."""

import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scoringbench.multivariate.screened_datasets import (
    PROTOCOL, TARGET_DIMS, SUITE_PATH, finalize_suite, save_screen,
)
from scoringbench.multivariate.sources import get_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("scoringbench", "variogram_screened"),
                        default="scoringbench")
    parser.add_argument("--target_dims", nargs="+", type=int, choices=TARGET_DIMS, default=list(TARGET_DIMS))
    parser.add_argument("--dataset_index", type=int)
    parser.add_argument("--output_dir", type=Path, default=Path("results") / PROTOCOL)
    parser.add_argument("--quality_only", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--suite_path", type=Path, default=SUITE_PATH)
    args = parser.parse_args()
    if args.freeze and (args.source != "scoringbench" or args.dataset_index is not None
                        or sorted(args.target_dims) != list(TARGET_DIMS) or args.quality_only):
        parser.error("--freeze requires the complete native family at d=2,3,5,10 with KCI")
    if args.freeze and args.suite_path.exists():
        parser.error("refusing to overwrite a frozen suite; use a new --suite_path")
    source = get_source(args.source)
    errors = 0
    entries = []
    for dimension in args.target_dims:
        datasets = source.enumerate_datasets(dimension, 3000)
        if args.dataset_index is not None:
            if not 0 <= args.dataset_index < len(datasets):
                parser.error("dataset_index is outside the candidate list")
            datasets = [datasets[args.dataset_index]]
        counts = Counter()
        for entry in datasets:
            entry.update(screen_only=True, quality_only=args.quality_only)
            entry.pop("screen", None)
            try:
                source.load(entry, target_dim=dimension)
            except Exception as exc:
                if not entry.get("screen"):
                    errors += 1
                    entry["screen"] = dict(selected=False, scale_ablation=False,
                                           reason="error", error=f"{type(exc).__name__}: {exc}")
            gc.collect()
            save_screen(entry, args.output_dir / args.source / f"d{dimension}")
            entries.append(entry)
            decision = entry["screen"]
            counts[decision["reason"]] += 1
            print(f"{entry['name']}: {decision['reason']}; "
                  f"n={decision.get('n_curation')} pX={decision.get('n_features')} "
                  f"p={decision.get('kci_p')} ordinal={decision.get('is_ordinal', False)}", flush=True)
        print(f"{args.source} d={dimension}: {dict(counts)}", flush=True)
    payload = finalize_suite(entries)
    payload["quality_only"] = args.quality_only
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "suite_draft.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    lines = [f"# Multivariate Suite: {PROTOCOL}", "", "No benchmark-model scores enter selection.", "",
             "| d | Candidates | Selected (Core) | Scale Ablation | Rejected |", "| --- | --- | --- | --- | --- |"]
    for dimension in args.target_dims:
        decisions = [entry["screen"] for entry in entries if entry["target_dim"] == dimension]
        n_cand = len(decisions)
        n_sel = sum(bool(item.get("selected")) for item in decisions)
        n_scale = sum(bool(item.get("scale_ablation")) for item in decisions)
        n_rej = n_cand - n_sel - n_scale
        lines.append(f"| {dimension} | {n_cand} | {n_sel} | {n_scale} | {n_rej} |")
    lines += ["", "## Decisions", "",
              "| Dataset | d | Decision | KCI p | Holm p | Residual dcorr | Scale Ratio | Ordinal | Duplicate Of |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for entry in entries:
        decision = entry["screen"]
        lines.append(f"| {entry['name']} | {entry['target_dim']} | {decision['reason']} | "
                     f"{decision.get('kci_p')} | {decision.get('holm_p')} | {decision.get('residual_dcor')} | "
                     f"{decision.get('scale_ratio')} | {decision.get('is_ordinal', False)} | {decision.get('duplicate_of', '-')} |")
        save_screen(entry, args.output_dir / args.source / f"d{entry['target_dim']}")
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")
    if args.freeze:
        if errors:
            raise SystemExit("Cannot freeze with unresolved loading or test errors; see report.md")
        args.suite_path.parent.mkdir(parents=True, exist_ok=True)
        args.suite_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        print(f"Frozen suite: {args.suite_path}", flush=True)
    return int(errors > 0)


if __name__ == "__main__":
    raise SystemExit(main())