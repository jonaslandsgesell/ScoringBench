"""Generate nonlinear regression datasets with feature-gated shared-latent target dependence.

Each manifest records the mean and target-dependence specification, the complete
realized innovation vine,
signed edge taus, artifact checksum and dependency versions. Artifacts live under
``datasets/synthetic/d{d}_n{n}``.
An existing set is never overwritten unless --force is supplied.

Usage
-----
    PYTHONPATH=. python scripts/generate_synthetic.py --target-dim 3 --sample-size 1000

After regenerating, commit the parquet files and manifest. If you track the
artifacts with git-LFS, ensure the LFS pattern covers
``datasets/synthetic/**/*.parquet`` (keep ``manifest.json`` off LFS so diffs stay
readable).
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyvinecopulib as pv
import scipy

from scoringbench.multivariate import config as cfg
from scoringbench.multivariate import synthetic_targets as st


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dim", type=int, default=int(cfg.TARGET_DIM),
                        help="Number of targets d per dataset.")
    parser.add_argument("--sample-size", type=int, default=int(cfg.SAMPLE_SIZE),
                        help="Number of rows n per dataset.")
    parser.add_argument("--force", action="store_true",
                        help="Explicitly replace an existing frozen set.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    configs = st.enumerate_synthetic(
        target_dim=args.target_dim, sample_size=args.sample_size
    )
    out_dir = st._shape_subdir(args.target_dim, args.sample_size)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        raise FileExistsError(f"Frozen synthetic set already exists at {out_dir}; use --force to replace it")
    out_dir.mkdir(parents=True, exist_ok=True)
    # A clean regeneration: remove any stale parquet artifacts from a previous
    # enumeration (e.g. dropped dependence-strength cells) so the frozen set
    # contains exactly the datasets the current config enumerates.
    for stale in out_dir.glob("*.parquet"):
        stale.unlink()
    print(f"Generating {len(configs)} synthetic datasets "
          f"(d={args.target_dim}, n={args.sample_size}) "
          f"into {out_dir}")

    manifest: dict[str, object] = {
        "_meta": {
            "design_reference": "https://arxiv.org/html/1701.00845v2#S4.SS1",
            "structure_sampling": "uniform random R-vine",
            "sampling": configs[0]["sampling"],
            "features": "independent_standard_normal",
            "vine_role": "target_specific_innovations",
            "margins": "conditional_normal",
            "residual_margins": "standard_normal",
            "tau_beta_parameters": {
                strength: st._TAU_BETA_PARAMETERS[strength]
                for strength in cfg.SYNTHETIC_DEPENDENCE_STRENGTHS
            },
            "tree_level_start": 1,
            "student_df": 4.0,
            "normal_quantile_clip": [1e-12, 1 - 1e-12],
            "nonlinear_mean": {
                "enabled": True,
                **configs[0]["nonlinear_mean"],
            },
            "target_dependence": configs[0]["target_dependence"],
            "rng_streams": list(st._STREAM_NAMES),
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "pandas_version": pd.__version__,
            "pyvinecopulib_version": pv.__version__,
            "target_dim": args.target_dim,
            "sample_size": args.sample_size,
            "n_datasets": len(configs),
        }
    }

    for dataset_config in configs:
        features, targets, vine = st._generate_with_vine(dataset_config)
        frame = features.join(targets)
        path = st._artifact_path(dataset_config["name"], args.target_dim, args.sample_size)
        frame.to_parquet(path, index=False)
        sha = st._sha256(path)
        manifest[dataset_config["name"]] = {
            "config": dataset_config,
            "vine": json.loads(vine.to_json()),
            "pair_copula_taus": [[float(pair.tau) for pair in tree]
                                 for tree in vine.pair_copulas],
            "sha256": sha,
        }
        print(f"  wrote {path.name}  ({frame.shape[0]}x{frame.shape[1]})  "
              f"sha256={sha[:12]}..")

    manifest_path = st._manifest_path(args.target_dim, args.sample_size)
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    with open(temporary_manifest, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
    temporary_manifest.replace(manifest_path)
    print(f"Wrote manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
