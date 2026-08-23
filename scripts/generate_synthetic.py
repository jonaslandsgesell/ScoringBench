"""(Re)generate the frozen synthetic dataset artifacts.

The synthetic source (:mod:`scoringbench.multivariate.synthetic_targets`) loads
committed parquet artifacts so results are reproducible across numpy /
pyvinecopulib versions. This script regenerates that frozen set: for every
config produced by ``enumerate_synthetic`` it runs ``_generate`` and writes one
``{name}.parquet`` plus a ``manifest.json`` recording the generating parameters,
the artifact sha256, and the numpy / pyvinecopulib versions used.

Usage
-----
    python scripts/generate_synthetic.py --target-dim 3 --sample-size 1000

After regenerating, commit the parquet files and manifest. If you track the
artifacts with git-LFS, ensure the LFS pattern covers
``datasets/synthetic/*.parquet`` (keep ``manifest.json`` off LFS so diffs stay
readable).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pyvinecopulib as pv

from scoringbench.multivariate import config as cfg
from scoringbench.multivariate import synthetic_targets as st


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-dim", type=int, default=int(cfg.TARGET_DIM),
                   help="Number of targets d per dataset.")
    p.add_argument("--sample-size", type=int, default=int(cfg.SAMPLE_SIZE),
                   help="Number of rows n per dataset.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    # Artifacts are scoped by (target_dim, sample_size) into their own subfolder
    # so different shapes never collide and each carries its own manifest.
    out_dir = st._shape_subdir(args.target_dim, args.sample_size)
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = st.enumerate_synthetic(
        target_dim=args.target_dim, sample_size=args.sample_size
    )
    print(f"Generating {len(configs)} synthetic datasets "
          f"(d={args.target_dim}, n={args.sample_size}) "
          f"into {out_dir}")

    manifest: dict[str, object] = {
        "_meta": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "numpy_version": np.__version__,
            "pyvinecopulib_version": pv.__version__,
            "target_dim": args.target_dim,
            "sample_size": args.sample_size,
        }
    }

    for ds in configs:
        X, Y = st._generate(ds, target_dim=args.target_dim)
        # Store features + targets in one frame; loader splits by prefix.
        frame = X.join(Y)
        path = st._artifact_path(ds["name"], args.target_dim, args.sample_size)
        frame.to_parquet(path, index=False)
        sha = st._sha256(path)
        manifest[ds["name"]] = {
            "family": ds["family"],
            "tau": ds["tau"],
            "replicate": ds["replicate"],
            "seed": ds["seed"],
            "n_samples": ds["n_samples"],
            "n_features": ds["n_features"],
            "target_dim": ds["target_dim"],
            "noise_scale": ds["noise_scale"],
            "mean_scale": ds["mean_scale"],
            "sha256": sha,
        }
        print(f"  wrote {path.name}  ({frame.shape[0]}x{frame.shape[1]})  "
              f"sha256={sha[:12]}..")

    manifest_path = st._manifest_path(args.target_dim, args.sample_size)
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    print(f"Wrote manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
