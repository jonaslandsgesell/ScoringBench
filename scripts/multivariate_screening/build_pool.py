"""Describe the full OpenML candidate pool for the multivariate benchmark.

The pool is *every active OpenML dataset*; nothing is hand-picked. Whether a
dataset is multivariate is decided only by OpenML's own metadata: the uploader's
``default_target_attribute`` (a comma-separated list for multi-target uploads).
We never choose target columns ourselves.

Output: one parquet row per active dataset, with

    openml_id, name, version, upload_date, n_rows, d_x, d_y,
    d_y_numeric, d_x_numeric, n_missing_values,
    target_cols, numeric_target_cols, excluded_cols

``d_y`` counts the declared default targets. ``d_x`` is every remaining column
except OpenML's declared row-id/ignore attributes. The per-column type counts
(``d_y_numeric``, ``d_x_numeric``) come from the features endpoint and are only
fetched when ``d_y > 1``; they are null otherwise. ``target_cols`` is the
declared target list in upload order, ``numeric_target_cols`` its numeric subset
(``d_y > 1`` only), ``excluded_cols`` OpenML's declared row-id/ignore columns.

By default only declared multi-target datasets (``d_y > 1``) are written, because
the whole downstream suite needs at least two jointly-measured targets and those
are ~2% of OpenML. OpenML's dataset-list API cannot filter on the number of
targets (its only metadata filters are ``number_instances``, ``number_features``,
``number_classes`` and ``number_missing_values``), so ``d_y`` is read from each
dataset's ``default_target_attribute`` and the ``d_y > 1`` filter is applied here,
client-side, while the pool is built. Pass ``--include_singletarget`` to keep the
unfiltered population instead. The remaining selection filters (numeric targets,
size, deduplication) stay downstream as explicit, logged steps.

API responses are cached per dataset under ``--cache_dir``, so a rerun only
fetches datasets it has not seen. ``--max_pool`` stops once that many datasets
have entered the pool, for a quick end-to-end smoke test before a full run.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path

import pandas as pd

API = "https://www.openml.org/api/v1/json"
RESULTS = Path(__file__).resolve().parent / "results"


def _get_json(url: str, cache: Path, retries: int = 4) -> dict:
    if cache.exists():
        return json.loads(cache.read_text())
    error = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                payload = json.load(response)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(payload))
            return payload
        except Exception as exc:  # network hiccups, HTTP 5xx, rate limits
            error = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{url}: {error}")


def _split(value) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [v for item in value for v in _split(item)]
    return [v.strip() for v in str(value).split(",") if v.strip()]


def describe(did: int, n_features: int, cache_dir: Path) -> dict:
    """Pool row for one dataset (description always; feature types only if multi-target)."""
    desc = _get_json(f"{API}/data/{did}", cache_dir / "data" / f"{did}.json")["data_set_description"]
    targets = _split(desc.get("default_target_attribute"))
    excluded = set(_split(desc.get("row_id_attribute"))) | set(_split(desc.get("ignore_attribute")))
    row = dict(openml_id=did, name=desc.get("name"), version=int(desc.get("version", 0)),
               upload_date=desc.get("upload_date"), d_y=len(targets),
               d_x=int(n_features) - len(targets) - len(excluded - set(targets)),
               d_y_numeric=None, d_x_numeric=None, target_cols=targets,
               numeric_target_cols=None, excluded_cols=sorted(excluded - set(targets)))
    if len(targets) > 1:
        feats = _get_json(f"{API}/data/features/{did}",
                          cache_dir / "features" / f"{did}.json")["data_features"]["feature"]
        is_y = lambda f: f["name"] in targets
        is_x = lambda f: not is_y(f) and f["name"] not in excluded
        numeric_targets = [t for t in targets if any(
            f["name"] == t and f["data_type"] == "numeric" for f in feats)]
        row.update(d_y_numeric=len(numeric_targets), numeric_target_cols=numeric_targets,
                   d_x_numeric=sum(f["data_type"] == "numeric" for f in feats if is_x(f)),
                   d_x=sum(is_x(f) for f in feats))
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=RESULTS / "openml_pool.parquet")
    parser.add_argument("--cache_dir", type=Path, default=RESULTS / "openml_pool_cache")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--include_singletarget", action="store_true",
                        help="Keep every dataset; by default only d_y > 1 datasets are written.")
    parser.add_argument("--max_pool", type=int, default=None,
                        help="Stop once this many datasets are in the pool (quick smoke test).")
    args = parser.parse_args()
    multitarget_only = not args.include_singletarget

    import openml

    listing = openml.datasets.list_datasets(output_format="dataframe", status="active")
    listing = listing[["did", "NumberOfInstances", "NumberOfFeatures", "NumberOfMissingValues"]]
    print(f"active OpenML datasets: {len(listing)}", flush=True)

    def task(item):
        did, n_features = item
        try:
            return describe(int(did), 0 if pd.isna(n_features) else n_features, args.cache_dir)
        except Exception as exc:
            return dict(openml_id=int(did), error=str(exc))

    def keep(row) -> bool:
        return args.include_singletarget or "error" in row or int(row.get("d_y", 0)) > 1

    # Scan in order, in worker-sized chunks, so --max_pool can stop early instead
    # of describing every one of the ~7,000 datasets.
    items = iter(zip(listing["did"], listing["NumberOfFeatures"]))
    rows = []
    scanned = 0
    with ThreadPoolExecutor(args.workers) as pool:
        while batch := list(islice(items, args.workers * 8)):
            for row in pool.map(task, batch):
                scanned += 1
                if keep(row):
                    rows.append(row)
                if scanned % 500 == 0:
                    print(f"  scanned {scanned}, pool {len([r for r in rows if 'error' not in r])}", flush=True)
            if args.max_pool is not None and len([r for r in rows if "error" not in r]) >= args.max_pool:
                break
    print(f"scanned {scanned} datasets; pool has "
          f"{len([r for r in rows if 'error' not in r])} "
          f"{'(all targets)' if args.include_singletarget else '(d_y > 1)'}", flush=True)

    failed = [r for r in rows if "error" in r]
    if failed:
        print(f"{len(failed)} datasets failed to fetch (rerun to retry, cache keeps the rest):")
        for r in failed[:20]:
            print(f"  {r['openml_id']}: {r['error']}")

    pool = pd.DataFrame([r for r in rows if "error" not in r])
    pool = pool.merge(listing.rename(columns={"did": "openml_id", "NumberOfInstances": "n_rows",
                                              "NumberOfMissingValues": "n_missing_values"})
                      [["openml_id", "n_rows", "n_missing_values"]], on="openml_id", how="left")
    columns = ["openml_id", "name", "version", "upload_date", "n_rows", "d_x", "d_y",
               "d_y_numeric", "d_x_numeric", "n_missing_values",
               "target_cols", "numeric_target_cols", "excluded_cols"]
    pool = pool[columns].sort_values("openml_id").reset_index(drop=True)
    for col in ("n_rows", "d_y_numeric", "d_x_numeric", "n_missing_values"):
        pool[col] = pool[col].astype("Int64")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(args.output, index=False)

    multi = pool[pool.d_y > 1]
    print(f"wrote {args.output}: {len(pool)} datasets, {len(multi)} with d_y > 1, "
          f"{int((multi.d_y_numeric > 1).sum())} with >1 numeric target", flush=True)
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
