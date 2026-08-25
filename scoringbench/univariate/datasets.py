from __future__ import annotations

import os
import io
import re
import zipfile
import urllib.request
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_iris, load_diabetes, load_breast_cancer, load_wine, load_digits,
    fetch_openml,
)
import openml

# ---------------------------------------------------------------------------
# Cache directory for downloaded datasets
# ---------------------------------------------------------------------------
CACHE_DIR = Path.home() / '.cache' / 'scoringbench' / 'datasets'


# ---------------------------------------------------------------------------
# Permanently skipped datasets (OpenML dataset ids)
# ---------------------------------------------------------------------------
# Datasets listed here are dropped from DATASETS_CONFIG unconditionally, so
# they never reach the univariate runner nor the multivariate runner (which
# reuses this same config via scoringbench.multivariate.datasets).
#
# 42572 Santander_transaction_value: ~4.5k rows x 4991 mostly-sparse/constant
#   columns. RealMLP (TabArena HPO) hangs indefinitely on it, and several other
#   models blow up on the feature count. Not worth the wall-clock.
SKIPPED_OPENML_IDS = {
    42572,  # Santander_transaction_value
}


def _ensure_cached(name: str, url: str, filename: str | None = None) -> Path:
    """Download a file and cache it locally.  Return path to cached file."""
    cache_dir = CACHE_DIR / name
    cache_dir.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = url.split('/')[-1]
    filepath = cache_dir / filename
    if not filepath.exists():
        print(f"  Downloading {name} from {url} ...")
        urllib.request.urlretrieve(url, filepath)
    return filepath


# ---------------------------------------------------------------------------
# Name normalisation & deduplication helpers
# ---------------------------------------------------------------------------
def _normalize_name(name: str) -> str:
    """Normalise a dataset name for deduplication comparison.
    
    - Lowercase all text (case-insensitive)
    - Strip leading numeric prefixes
    - Remove all special characters
    - Collapse multiple spaces/underscores
    """
    name = name.lower()  # Case-insensitive
    # Strip leading numeric prefix
    name = re.sub(r'^\d+[_\-\s]?', '', name)
    # Replace common separators/spaces with nothing
    name = re.sub(r'[\s_\-]+', '', name)
    # Remove all non-alphanumeric
    name = re.sub(r'[^a-z0-9]', '', name)
    # Remove duplicate consecutive chars (e.g., "dataa" -> "data")
    name = re.sub(r'([a-z0-9])\1{2,}', r'\1', name)
    return name


def _similarity_ratio(str1: str, str2: str) -> float:
    """Calculate similarity ratio between two strings (0-1).
    
    Uses string matching to catch close but not identical duplicates.
    """
    from difflib import SequenceMatcher
    return SequenceMatcher(None, str1, str2).ratio()


def _is_duplicate(new_name: str, existing_names: set[str],
                  dedup_keys: list[str] | None = None,
                  verbose: bool = False) -> bool:
    """Return True if *new_name* (or any of its *dedup_keys*) already exists.
    
    Performs multiple levels of checking:
    1. Exact normalized match
    2. Substring match (both directions) for names > 3 chars
    3. Fuzzy match (>85% similarity)
    4. Explicit dedup keys with all above checks
    """
    norm_new = _normalize_name(new_name)
    
    reasons = []
    
    # --- Check 1: Exact normalised match ---
    if norm_new in existing_names:
        reasons.append("exact normalized match")
        if verbose:
            print(f"    DUPLICATE: {new_name} - {reasons[-1]} with {norm_new}")
        return True
    
    # --- Check 2: Substring match (both directions) for names > 3 chars ---
    for existing in existing_names:
        if len(norm_new) > 3 and len(existing) > 3:
            if norm_new in existing or existing in norm_new:
                reasons.append(f"substring match with {existing}")
                if verbose:
                    print(f"    DUPLICATE: {new_name} - {reasons[-1]}")
                return True
        
        # --- Check 3: Fuzzy match (high similarity) ---
        similarity = _similarity_ratio(norm_new, existing)
        if similarity >= 0.85:  # 85% or higher is likely a duplicate
            reasons.append(f"fuzzy match ({similarity:.1%}) with {existing}")
            if verbose:
                print(f"    DUPLICATE: {new_name} - {reasons[-1]}")
            return True
    
    # --- Check 4: Explicit dedup keys (apply all checks above) ---
    if dedup_keys:
        for key in dedup_keys:
            norm_key = _normalize_name(key)
            
            # Exact match on dedup key
            if norm_key in existing_names:
                reasons.append(f"dedup_key exact match: {key}")
                if verbose:
                    print(f"    DUPLICATE: {new_name} - {reasons[-1]}")
                return True
            
            # Substring match on dedup key
            for existing in existing_names:
                if len(norm_key) > 3 and len(existing) > 3:
                    if norm_key in existing or existing in norm_key:
                        reasons.append(f"dedup_key substring match: {key} with {existing}")
                        if verbose:
                            print(f"    DUPLICATE: {new_name} - {reasons[-1]}")
                        return True
                
                # Fuzzy match on dedup key
                similarity = _similarity_ratio(norm_key, existing)
                if similarity >= 0.85:
                    reasons.append(f"dedup_key fuzzy match: {key} ({similarity:.1%}) with {existing}")
                    if verbose:
                        print(f"    DUPLICATE: {new_name} - {reasons[-1]}")
                    return True
    
    return False


def _is_classification(y: pd.Series) -> bool:
    """Check if target variable indicates binary or few-class classification (not regression).
    
    Returns True if y has fewer than 4 unique values (binary or few-class classification).
    These datasets should be skipped from a regression benchmark.
    """
    # Handle NaN values
    y_clean = y.dropna()
    n_unique = y_clean.nunique()
    return n_unique < 4


# ---------------------------------------------------------------------------
# OpenML suite loader (existing)
# ---------------------------------------------------------------------------
# The suite -> [dataset config] mapping is cached to disk, one JSON file per
# suite id, in ``SUITE_CACHE_DIR``. OpenML (0.15.x) does NOT cache
# ``get_suite``/``get_study`` calls, so without this every run (and every SLURM
# array task) re-queries ``openml.org/api/v1/xml/study/{id}`` and dies with a
# 504 when the OpenML server is flaky. Once a suite has been resolved once, we
# read straight from the local cache and never touch the network again.
#
# On a transient failure we fall back to the last-known-good cached copy so a
# single 504 no longer silently drops an entire suite's datasets.
#
# Set ``SCORINGBENCH_NO_CACHE=1`` to bypass the cache (force a fresh fetch).
SUITE_CACHE_DIR = CACHE_DIR / 'suites'


def _read_suite_cache(suite_id) -> list | None:
    """Return cached dataset configs for ``suite_id`` or ``None`` if absent."""
    path = SUITE_CACHE_DIR / f"suite_{suite_id}.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        # Corrupt/partial cache file -> ignore and re-fetch.
        return None


def _write_suite_cache(suite_id, configs: list) -> None:
    """Atomically persist a suite's resolved dataset configs to disk."""
    try:
        SUITE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = SUITE_CACHE_DIR / f"suite_{suite_id}.json"
        tmp = path.with_suffix('.json.tmp')
        with open(tmp, 'w') as f:
            json.dump(configs, f, indent=2)
        os.replace(tmp, path)
    except Exception as e:
        # Caching is best-effort; never fail over a cache write.
        print(f"  [suite-cache] warning: could not cache suite "
              f"{suite_id}: {str(e)[:80]}")


def _fetch_suite_configs(suite_id) -> list:
    """Fetch and resolve a single OpenML suite into a list of dataset configs.

    Returns ``(configs, complete)`` where ``complete`` is ``True`` only if every
    task in the suite resolved without error. A suite is cached only when it is
    complete, so a partial fetch (e.g. a 504 on one task) is retried on the next
    run instead of being frozen into the cache.

    Raises on network/server errors from ``get_suite`` itself so the caller can
    decide whether to fall back to a cached copy.
    """
    suite = openml.study.get_suite(suite_id)
    print(f"  Processing OpenML Suite {suite_id} ('{suite.name}')...")
    configs = []
    complete = True
    for task_id in suite.tasks:
        try:
            task = openml.tasks.get_task(task_id)
            dataset_id = task.dataset_id
            dataset = openml.datasets.get_dataset(
                dataset_id, download_data=False
            )
            dataset_name = dataset.name
            abbr = ''.join(
                [word[0] for word in dataset_name.split('_') if word]
            ).upper()[:3]

            configs.append({
                'name': dataset_name,
                'abbr': abbr,
                'source': 'openml',
                'id': dataset_id,
            })
        except Exception as e:
            complete = False
            print(f"    Error processing task {task_id} "
                  f"in suite {suite_id}: {e}")
    return configs, complete


def generate_datasets_config_from_openml_suites(suite_ids):
    """Generate dataset configs from OpenML benchmark suites.

    Uses an on-disk per-suite cache (``SUITE_CACHE_DIR``) so suites are resolved
    against the OpenML servers at most once per machine. Set the env var
    ``SCORINGBENCH_NO_CACHE=1`` to bypass the cache and force a fresh fetch.
    """
    datasets_config = []
    seen_dataset_ids = set()
    no_cache = bool(os.environ.get('SCORINGBENCH_NO_CACHE'))

    print("Fetching datasets from OpenML suites...")

    for suite_id in suite_ids:
        configs = None

        # --- Cache hit: reuse the previously resolved suite ---
        if not no_cache:
            configs = _read_suite_cache(suite_id)
            if configs is not None:
                print(f"  [suite-cache] hit for suite {suite_id} "
                      f"({len(configs)} datasets)")

        # --- Cache miss (or bypass): fetch from OpenML ---
        if configs is None:
            try:
                configs, complete = _fetch_suite_configs(suite_id)
                # Only cache a fully-resolved suite so a partial fetch (e.g. a
                # 504 on a single task) is retried next run instead of frozen.
                if complete and not no_cache:
                    _write_suite_cache(suite_id, configs)
                elif not complete:
                    print(f"  [suite-cache] suite {suite_id} incomplete "
                          f"(some tasks failed) - NOT caching, will retry "
                          f"next run")
            except Exception as e:
                print(f"  Error retrieving suite {suite_id}: {e}")
                # Fall back to any stale-but-usable cached copy.
                stale = _read_suite_cache(suite_id)
                if stale is not None:
                    print(f"  [suite-cache] falling back to cached suite "
                          f"{suite_id} ({len(stale)} datasets)")
                    configs = stale
                else:
                    configs = []

        # --- Merge, deduplicating by dataset id ---
        for cfg in configs:
            dataset_id = cfg.get('id')
            if dataset_id not in seen_dataset_ids:
                datasets_config.append(cfg)
                seen_dataset_ids.add(dataset_id)

    print("\nFinished populating DATASETS_CONFIG from OpenML suites.")
    return datasets_config


# ---------------------------------------------------------------------------
# Source-specific loaders
# ---------------------------------------------------------------------------
def _load_pmlb(config: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Load a PMLB dataset from its GitHub TSV.gz URL."""
    url = config['url']
    name = config['name']
    cache_path = _ensure_cached(name, url, f"{name}.tsv.gz")
    df = pd.read_csv(cache_path, sep='\t', compression='gzip')
    target_col = config.get('target_col', 'target')
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y





def _load_keel(config: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Load a dataset from the KEEL repository (.dat format inside .zip)."""
    url = config['url']
    name = config['name']

    cache_path = _ensure_cached(name, url, f"{name}.zip")

    extract_dir = CACHE_DIR / name / 'extracted'
    if not extract_dir.exists():
        extract_dir.mkdir(parents=True)
        with zipfile.ZipFile(cache_path, 'r') as z:
            z.extractall(extract_dir)

    # Find .dat files and combine them (train + test)
    dat_files = list(extract_dir.rglob('*.dat'))
    if not dat_files:
        raise FileNotFoundError(f"No .dat files found in {extract_dir}")

    all_dfs: list[pd.DataFrame] = []
    attr_names: list[str] | None = None

    for dat_file in dat_files:
        attrs: list[str] = []
        data_lines: list[str] = []
        in_data = False

        with open(dat_file, 'r') as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith('%'):
                    continue
                if line.lower().startswith('@data'):
                    in_data = True
                    continue
                if not in_data:
                    if line.lower().startswith('@attribute'):
                        parts = line.split()
                        attrs.append(parts[1])
                else:
                    if line and not line.startswith('@'):
                        data_lines.append(line)

        if data_lines:
            if attr_names is None and attrs:
                attr_names = attrs
            data_str = '\n'.join(data_lines)
            df = pd.read_csv(
                io.StringIO(data_str), header=None, names=attr_names
            )
            all_dfs.append(df)

    if not all_dfs:
        raise ValueError(f"Could not parse any KEEL data files for {name}")

    df = pd.concat(all_dfs, ignore_index=True)
    df = df.replace('<null>', np.nan)

    # Last column is target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y


# ---------------------------------------------------------------------------
# TabRegSet-101 benchmark datasets
# ---------------------------------------------------------------------------
TABREGSET_DATASETS = [
    # ---- UCI datasets converted to OpenML (no ucimlrepo dependency) ----
    {'name': 'Abalone', 'source': 'openml', 'id': 183,
     'dedup_keys': ['abalone']},
    {'name': 'Student_Performance', 'source': 'openml', 'id': 42352},
    {'name': 'Infrared_Thermography_Temperature', 'source': 'openml',
     'id': 46613},
    {'name': 'Parkinsons_Telemonitoring', 'source': 'openml', 'id': 4531},
    {'name': 'Energy_Efficiency', 'source': 'openml', 'id': 44960,
     'dedup_keys': ['energy-efficiency', 'energy_efficiency']},
    {'name': 'QsarFishToxicity', 'source': 'openml', 'id': 44970},
    {'name': 'concrete_compressive_strength', 'source': 'openml', 'id': 44959},
    {'name': 'PRODUCTIVITY', 'source': 'openml', 'id': 42989},
    {'name': 'AIRFOIL', 'source': 'openml', 'id': 44957,
     'dedup_keys': ['airfoil_self_noise', 'airfoilselfnoise']},
    {'name': 'BIAS_CORRECTION', 'source': 'openml', 'id': 42897},
    
    # ---- PMLB datasets (TSV.gz from GitHub, target col = 'target') ----
    {'name': '1027_ESL', 'source': 'pmlb',
     'url': 'https://github.com/EpistasisLab/penn-ml-benchmarks/raw/master/datasets/1027_ESL/1027_ESL.tsv.gz'},
    {'name': '1028_SWD', 'source': 'pmlb',
     'url': 'https://github.com/EpistasisLab/penn-ml-benchmarks/raw/master/datasets/1028_SWD/1028_SWD.tsv.gz'},
    {'name': '1029_LEV', 'source': 'pmlb',
     'url': 'https://github.com/EpistasisLab/penn-ml-benchmarks/raw/master/datasets/1029_LEV/1029_LEV.tsv.gz'},
    {'name': '1030_ERA', 'source': 'pmlb',
     'url': 'https://github.com/EpistasisLab/penn-ml-benchmarks/raw/master/datasets/1030_ERA/1030_ERA.tsv.gz'},
    {'name': '1199_BNG_echoMonths', 'source': 'pmlb',
     'url': 'https://github.com/EpistasisLab/penn-ml-benchmarks/raw/master/datasets/1199_BNG_echoMonths/1199_BNG_echoMonths.tsv.gz'},
    {'name': '197_cpu_act', 'source': 'pmlb',
     'url': 'https://github.com/EpistasisLab/penn-ml-benchmarks/raw/master/datasets/197_cpu_act/197_cpu_act.tsv.gz',
     'dedup_keys': ['cpu_act', 'cpuact']},
    {'name': '225_puma8NH', 'source': 'pmlb',
     'url': 'https://github.com/EpistasisLab/penn-ml-benchmarks/raw/master/datasets/225_puma8NH/225_puma8NH.tsv.gz'},
    {'name': '227_cpu_small', 'source': 'pmlb',
     'url': 'https://github.com/EpistasisLab/penn-ml-benchmarks/raw/master/datasets/227_cpu_small/227_cpu_small.tsv.gz',
     'dedup_keys': ['cpu_small', 'cpusmall']},
    {'name': '294_satellite_image', 'source': 'pmlb',
     'url': 'https://github.com/EpistasisLab/penn-ml-benchmarks/raw/master/datasets/294_satellite_image/294_satellite_image.tsv.gz'},
    {'name': '344_mv', 'source': 'pmlb',
     'url': 'https://github.com/EpistasisLab/penn-ml-benchmarks/raw/master/datasets/344_mv/344_mv.tsv.gz'},
    {'name': '503_wind', 'source': 'pmlb',
     'url': 'https://github.com/EpistasisLab/penn-ml-benchmarks/raw/master/datasets/503_wind/503_wind.tsv.gz'},
    {'name': '529_pollen', 'source': 'pmlb',
     'url': 'https://github.com/EpistasisLab/penn-ml-benchmarks/raw/master/datasets/529_pollen/529_pollen.tsv.gz'},
    {'name': '537_houses', 'source': 'pmlb',
     'url': 'https://github.com/EpistasisLab/penn-ml-benchmarks/raw/master/datasets/537_houses/537_houses.tsv.gz'},
    {'name': '547_no2', 'source': 'pmlb',
     'url': 'https://github.com/EpistasisLab/penn-ml-benchmarks/raw/master/datasets/547_no2/547_no2.tsv.gz'},
    {'name': '564_fried', 'source': 'pmlb',
     'url': 'https://github.com/EpistasisLab/penn-ml-benchmarks/raw/master/datasets/564_fried/564_fried.tsv.gz'},
    {'name': '595_fri_c0_1000_10', 'source': 'pmlb',
     'url': 'https://github.com/EpistasisLab/penn-ml-benchmarks/raw/master/datasets/595_fri_c0_1000_10/595_fri_c0_1000_10.tsv.gz'},
    {'name': '593_fri_c1_1000_10', 'source': 'pmlb',
     'url': 'https://github.com/EpistasisLab/penn-ml-benchmarks/raw/master/datasets/593_fri_c1_1000_10/593_fri_c1_1000_10.tsv.gz'},
    {'name': '1193_BNG_lowbwt', 'source': 'pmlb',
     'url': 'https://github.com/EpistasisLab/penn-ml-benchmarks/raw/master/datasets/1193_BNG_lowbwt/1193_BNG_lowbwt.tsv.gz'},
    {'name': '1201_BNG_breastTumor', 'source': 'pmlb',
     'url': 'https://github.com/EpistasisLab/penn-ml-benchmarks/raw/master/datasets/1201_BNG_breastTumor/1201_BNG_breastTumor.tsv.gz'},
    {'name': '1203_BNG_pwLinear', 'source': 'pmlb',
     'url': 'https://github.com/EpistasisLab/penn-ml-benchmarks/raw/master/datasets/1203_BNG_pwLinear/1203_BNG_pwLinear.tsv.gz'},
    {'name': '215_2dplanes', 'source': 'pmlb',
     'url': 'https://github.com/EpistasisLab/penn-ml-benchmarks/raw/master/datasets/215_2dplanes/215_2dplanes.tsv.gz'},
    {'name': '218_house_8L', 'source': 'pmlb',
     'url': 'https://github.com/EpistasisLab/penn-ml-benchmarks/raw/master/datasets/218_house_8L/218_house_8L.tsv.gz'},

    # ---- KEEL datasets ----
    {'name': 'Wizmir', 'source': 'keel',
     'url': 'https://sci2s.ugr.es/keel/dataset/data/regression/wizmir.zip'},
    {'name': 'Ele2', 'source': 'keel',
     'url': 'https://sci2s.ugr.es/keel/dataset/data/regression/ele-2.zip'},
    {'name': 'Treasury', 'source': 'keel',
     'url': 'https://sci2s.ugr.es/keel/dataset/data/regression/treasury.zip'},
    {'name': 'Mortgage', 'source': 'keel',
     'url': 'https://sci2s.ugr.es/keel/dataset/data/regression/mortgage.zip'},
    {'name': 'Laser', 'source': 'keel',
     'url': 'https://sci2s.ugr.es/keel/dataset/data/regression/laser.zip'},
]

TALENT_OPENML_REGRESSION_DATASETS = [
        # ---- OpenML verified regression datasets ----
    {'name': '1000-Cameras-Dataset', 'source': 'openml', 'id': 43714},
    {'name': '2dplanes', 'source': 'openml', 'id': 215,
     'dedup_keys': ['2dplanes', 'fri_c0_1000_10']},
    {'name': '3D_Estimation_using_RSSI_of_WLAN_dataset_complete_1_target', 'source': 'openml', 'id': 45720},
    {'name': 'Ailerons', 'source': 'openml', 'id': 296,
     'dedup_keys': ['ailerons']},
    {'name': 'Another-Dataset-on-used-Fiat-500-(1538-rows)', 'source': 'openml', 'id': 43828},
    {'name': 'BNG(echoMonths)', 'source': 'openml', 'id': 1199,
     'dedup_keys': ['bng_echomonths', 'echomonths']},
    {'name': 'BNG(lowbwt)', 'source': 'openml', 'id': 1193,
     'dedup_keys': ['bng_lowbwt', 'lowbwt']},
    {'name': 'BNG(mv)', 'source': 'openml', 'id': 1213,
     'dedup_keys': ['bng_mv']},
    {'name': 'BNG(stock)', 'source': 'openml', 'id': 1200,
     'dedup_keys': ['bng_stock']},
    {'name': 'Brazilian_houses_reproduced', 'source': 'openml', 'id': 44152,
     'dedup_keys': ['brazilian_houses']},
    {'name': 'CPMP-2015-regression', 'source': 'openml', 'id': 41700},
    {'name': 'CPS1988', 'source': 'openml', 'id': 43963},
    {'name': 'CookbookReviews', 'source': 'openml', 'id': 45744},
    {'name': 'Goodreads-Computer-Books', 'source': 'openml', 'id': 43785},
    {'name': 'IEEE80211aa-GATS', 'source': 'openml', 'id': 43180},
    {'name': 'Job_Profitability', 'source': 'openml', 'id': 44311},
    {'name': 'Kaggle_bike_sharing_demand_challange', 'source': 'openml', 'id': 1414,
     'dedup_keys': ['bike_sharing_demand']},
    {'name': 'MIP-2016-regression', 'source': 'openml', 'id': 43071},
    {'name': 'MiamiHousing2016', 'source': 'openml', 'id': 43093},
    {'name': 'Moneyball', 'source': 'openml', 'id': 41021},
    {'name': 'NASA_PHM2008', 'source': 'openml', 'id': 42821},
    {'name': 'OnlineNewsPopularity', 'source': 'openml', 'id': 4545,
     'dedup_keys': ['onlinenewspopularity']},
    {'name': 'SAT11-HAND-runtime-regression', 'source': 'openml', 'id': 41980},
    {'name': 'analcatdata_supreme', 'source': 'openml', 'id': 504},
    {'name': 'avocado_sales', 'source': 'openml', 'id': 43927},
    {'name': 'bank32nh', 'source': 'openml', 'id': 558},
    {'name': 'bank8FM', 'source': 'openml', 'id': 572},
    {'name': 'boston', 'source': 'openml', 'id': 531},
    {'name': 'chscase_foot', 'source': 'openml', 'id': 703},
    {'name': 'colleges', 'source': 'openml', 'id': 42727},
    {'name': 'cpu_act', 'source': 'openml', 'id': 197,
     'dedup_keys': ['cpu_act', 'cpuact']},
    {'name': 'cpu_small', 'source': 'openml', 'id': 562,
     'dedup_keys': ['cpu_small', 'cpusmall']},
    {'name': 'dataset_sales', 'source': 'openml', 'id': 42183},
    {'name': 'debutanizer', 'source': 'openml', 'id': 23516},
    {'name': 'delta_elevators', 'source': 'openml', 'id': 198,
     'dedup_keys': ['delta_elevators', 'elevators', 'deltaelevatores']},
    {'name': 'elevators', 'source': 'openml', 'id': 216},
    {'name': 'fifa', 'source': 'openml', 'id': 44026},
    {'name': 'fried', 'source': 'openml', 'id': 564,
     'dedup_keys': ['fried', 'fri_c0_1000_10', 'fric01010']},
    {'name': 'house_16H_reg', 'source': 'openml', 'id': 574,
     'dedup_keys': ['house_16h', 'house_16H']},
    {'name': 'house_8L', 'source': 'openml', 'id': 218,
     'dedup_keys': ['house_8l', 'house_8L']},
    {'name': 'house_prices_nominal', 'source': 'openml', 'id': 42563},
    {'name': 'house_sales_reduced', 'source': 'openml', 'id': 42635,
     'dedup_keys': ['house_sales']},
    {'name': 'houses', 'source': 'openml', 'id': 537,
     'dedup_keys': ['houses', 'calhousing', 'california_housing']},
    {'name': 'kin8nm', 'source': 'openml', 'id': 189},
    {'name': 'mauna-loa-atmospheric-co2', 'source': 'openml', 'id': 41187},
    {'name': 'mv', 'source': 'openml', 'id': 344},
    {'name': 'pol_reg', 'source': 'openml', 'id': 201,
     'dedup_keys': ['pol']},
    {'name': 'puma32H', 'source': 'openml', 'id': 308},
    {'name': 'puma8NH', 'source': 'openml', 'id': 225},
    {'name': 'satellite_image', 'source': 'openml', 'id': 294,
     'dedup_keys': ['satellite_image']},
    {'name': 'sensory', 'source': 'openml', 'id': 546},
    {'name': 'socmob', 'source': 'openml', 'id': 541},
    {'name': 'space_ga', 'source': 'openml', 'id': 507},
    {'name': 'stock', 'source': 'openml', 'id': 223},
    {'name': 'stock_fardamento02', 'source': 'openml', 'id': 42545},
    {'name': 'sulfur', 'source': 'openml', 'id': 23515},
    {'name': 'topo_2_1', 'source': 'openml', 'id': 422},
    {'name': 'treasury', 'source': 'openml', 'id': 42367},
    {'name': 'us_crime', 'source': 'openml', 'id': 42730},
    {'name': 'weather_izmir', 'source': 'openml', 'id': 42369},
    {'name': 'wind', 'source': 'openml', 'id': 503,
     'dedup_keys': ['wind', 'wind_data']},
    {'name': 'yprop_4_1', 'source': 'openml', 'id': 416},
]

# ---------------------------------------------------------------------------
# Lazy initialization: DATASETS_CONFIG is built only when first accessed
# ---------------------------------------------------------------------------

# Global cache for DATASETS_CONFIG (built lazily on first access)
_DATASETS_CONFIG_CACHE = None
_DATASETS_CONFIG_INITIALIZED = False


def _build_datasets_config():
    """Build and deduplicate DATASETS_CONFIG. Called lazily on first access.
    
    This is deferred from module import to avoid fetching/validating datasets
    immediately. Only called when benchmark is actually run.
    """
    global _DATASETS_CONFIG_CACHE, _DATASETS_CONFIG_INITIALIZED
    
    if _DATASETS_CONFIG_INITIALIZED:
        return _DATASETS_CONFIG_CACHE
    
    print("Building datasets configuration...")
    
    # Start with OpenML suite datasets
    config = generate_datasets_config_from_openml_suites(
        suite_ids=[297, 299, 269]
    )

    # Merge TabRegSet-101 datasets, skipping duplicates
    _existing_names = {_normalize_name(d['name']) for d in config}
    _added = 0
    _skipped = []
    openml_datasets = TABREGSET_DATASETS + TALENT_OPENML_REGRESSION_DATASETS
    for _ds in openml_datasets:
        _dedup_keys = _ds.get('dedup_keys')
        if not _is_duplicate(_ds['name'], _existing_names, _dedup_keys, verbose=False):
            _abbr = ''.join(
                [w[0] for w in _ds['name'].split('_') if w]
            ).upper()[:3]
            _ds_config = {**_ds, 'abbr': _abbr}
            _ds_config.pop('dedup_keys', None)
            config.append(_ds_config)
            _existing_names.add(_normalize_name(_ds['name']))
            _added += 1
        else:
            _skipped.append(_ds['name'])

    print(f"Added {_added} OpenML datasets (skipped {len(openml_datasets) - _added} duplicates)")

    # --- Final deduplication pass ---
    _priority = {'openml': 4, 'pmlb': 3, 'keel': 2, 'sklearn': 1}
    _dedup_map: dict[str, dict] = {}
    _removed_dups: list[tuple[str, str]] = []

    for _ds in config:
        _norm = _normalize_name(_ds['name'])
        if _norm not in _dedup_map:
            _dedup_map[_norm] = _ds
        else:
            existing = _dedup_map[_norm]
            existing_prio = _priority.get(existing.get('source', ''), 0)
            new_prio = _priority.get(_ds.get('source', ''), 0)
            if new_prio > existing_prio:
                _removed_dups.append((_normalize_name(_ds['name']), existing['name']))
                _dedup_map[_norm] = _ds
            else:
                _removed_dups.append((_ds['name'], existing['name']))

    config = list(_dedup_map.values())

    if _removed_dups:
        print(f"⚠️  Found and removed {len(_removed_dups)} duplicate datasets (kept higher-priority sources)")

    # --- Drop permanently skipped OpenML datasets ---
    _skipped_ids = [
        _ds for _ds in config
        if _ds.get('source') == 'openml' and _ds.get('id') in SKIPPED_OPENML_IDS
    ]
    if _skipped_ids:
        config = [_ds for _ds in config if _ds not in _skipped_ids]
        for _ds in _skipped_ids:
            print(f"⊘ Skipping {_ds['name']} (openml id {_ds['id']}) "
                  f"- in SKIPPED_OPENML_IDS")

    print(f"✓ Datasets config ready: {len(config)} datasets (before validation)")

    # Export to JSON for reproducibility
    _export_list = []
    for _ds in config:
        entry = {'name': _ds['name'], 'source': _ds.get('source', 'openml')}
        if entry['source'] == 'openml':
            entry['id'] = _ds.get('id')
        elif entry['source'] in ('pmlb', 'keel'):
            entry['url'] = _ds.get('url')
        elif entry['source'] == 'sklearn':
            entry['loader'] = _ds.get('loader')
        if 'abbr' in _ds:
            entry['abbr'] = _ds['abbr']
        _export_list.append(entry)

    _datasets_json_path = Path.cwd() / 'datasets.json'
    try:
        with open(_datasets_json_path, 'w', encoding='utf-8') as _fh:
            json.dump(_export_list, _fh, indent=2, ensure_ascii=False)
        print(f"✓ Exported datasets list to: {_datasets_json_path}")
    except Exception as _e:
        print(f"⚠️  Could not write datasets.json: {_e}")
    
    _DATASETS_CONFIG_CACHE = config
    _DATASETS_CONFIG_INITIALIZED = True
    return config


def get_DATASETS_CONFIG():
    """Get DATASETS_CONFIG, building it lazily on first access."""
    global _DATASETS_CONFIG_CACHE, _DATASETS_CONFIG_INITIALIZED
    if not _DATASETS_CONFIG_INITIALIZED:
        _build_datasets_config()
    return _DATASETS_CONFIG_CACHE


# Initialize DATASETS_CONFIG as empty; will be lazily populated
DATASETS_CONFIG = []


# ---------------------------------------------------------------------------
# Sklearn loaders
# ---------------------------------------------------------------------------
SKLEARN_LOADERS = {
    'load_iris': load_iris,
    'load_diabetes': load_diabetes,
    'load_breast_cancer': load_breast_cancer,
    'load_wine': load_wine,
    'load_digits': load_digits,
}


# ---------------------------------------------------------------------------
# Main dataset loading function
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Processed-dataset cache
# ---------------------------------------------------------------------------
# The fully preprocessed (X, y) for each dataset is cached to disk as a single
# parquet file, keyed by a hash of the dataset config. This means each dataset
# is downloaded AND preprocessed at most once per machine; subsequent runs
# (including the full validate_datasets sweep and every SLURM array task) read
# straight from the local cache and never touch OpenML / PMLB / KEEL servers.
#
# Set the env var SCORINGBENCH_NO_CACHE=1 to bypass the cache entirely (e.g. to
# force a fresh download).
PROCESSED_CACHE_DIR = CACHE_DIR / 'processed'


def _dataset_cache_key(dataset_config: dict) -> str:
    """Stable hash of the fields that determine a dataset's processed (X, y)."""
    import hashlib

    key_fields = {
        'source': dataset_config.get('source', 'openml'),
        'id': dataset_config.get('id'),
        'loader': dataset_config.get('loader'),
        'url': dataset_config.get('url'),
        'target_col': dataset_config.get('target_col'),
        'name': dataset_config.get('name'),
    }
    blob = json.dumps(key_fields, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode('utf-8')).hexdigest()[:16]


def _read_processed_cache(key: str):
    """Return cached (X, y) for ``key`` or ``None`` if absent/unreadable."""
    path = PROCESSED_CACHE_DIR / f"{key}.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        y = df['__target__']
        X = df.drop(columns=['__target__'])
        return X, y
    except Exception:
        # Corrupt/partial cache file -> ignore and re-download.
        return None


def _write_processed_cache(key: str, X: pd.DataFrame, y: pd.Series) -> None:
    """Atomically persist the processed (X, y) to the parquet cache."""
    try:
        PROCESSED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df = X.copy()
        df['__target__'] = y.values
        # Parquet requires string column names.
        df.columns = [str(c) for c in df.columns]
        path = PROCESSED_CACHE_DIR / f"{key}.parquet"
        tmp = path.with_suffix('.parquet.tmp')
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    except Exception as e:
        # Caching is best-effort; never fail the benchmark over a cache write.
        print(f"  [cache] warning: could not cache dataset: {str(e)[:80]}")


def load_dataset(dataset_config: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Load and preprocess a dataset, using an on-disk cache when available.

    On a cache miss the dataset is downloaded and preprocessed via
    :func:`_load_dataset_uncached`, then the processed (X, y) is written to
    ``PROCESSED_CACHE_DIR`` so future runs (and other SLURM tasks) never
    re-download it. Set ``SCORINGBENCH_NO_CACHE=1`` to bypass the cache.

    Supported sources: sklearn, openml, pmlb, keel.
    """
    if os.environ.get('SCORINGBENCH_NO_CACHE'):
        return _load_dataset_uncached(dataset_config)

    key = _dataset_cache_key(dataset_config)
    cached = _read_processed_cache(key)
    if cached is not None:
        X, y = cached
        return pd.DataFrame(X), pd.Series(y)

    X, y = _load_dataset_uncached(dataset_config)
    _write_processed_cache(key, X, y)
    return X, y


def _load_dataset_uncached(dataset_config: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Load and preprocess a dataset from any supported source (no cache).

    Supported sources: sklearn, openml, pmlb, keel.
    """
    source = dataset_config.get('source', 'openml')

    # ----- source-specific loading -----
    if source == 'sklearn':
        loader_func = SKLEARN_LOADERS[dataset_config['loader']]
        data_sklearn = loader_func()
        X = pd.DataFrame(
            data_sklearn.data,
            columns=(data_sklearn.feature_names
                     if hasattr(data_sklearn, 'feature_names')
                     else [f"feature_{i}"
                           for i in range(data_sklearn.data.shape[1])]),
        )
        y = pd.Series(data_sklearn.target)

    elif source == 'openml':
        dataset_id = dataset_config['id']

        def _fetch_openml_robust(did):
            """Robust OpenML fetch: handles sparse ARFF, missing default target,
            and corrupted-cache md5 mismatches."""
            import shutil, numpy as _np
            from sklearn.datasets import get_data_home
            for attempt in range(2):
                try:
                    try:
                        d = fetch_openml(data_id=did, as_frame=True, parser='auto')
                    except ValueError as e:
                        if 'Sparse ARFF' not in str(e):
                            raise
                        # sparse ARFF: load dense-from-array then frame it
                        d = fetch_openml(data_id=did, as_frame=False, parser='auto')
                        Xa = d.data
                        if hasattr(Xa, 'toarray'):
                            Xa = Xa.toarray()
                        cols = list(getattr(d, 'feature_names', None) or
                                    [f"feature_{i}" for i in range(Xa.shape[1])])
                        d.data = pd.DataFrame(Xa, columns=cols)
                        if d.target is not None:
                            d.target = pd.Series(_np.asarray(d.target).reshape(-1))
                    return d
                except ValueError as e:
                    # corrupted local cache (md5 mismatch) -> purge openml cache and retry once
                    if 'md5 checksum' in str(e) and attempt == 0:
                        cache = os.path.join(get_data_home(), 'openml')
                        shutil.rmtree(cache, ignore_errors=True)
                        continue
                    # persistent md5 mismatch = server-side checksum bug -> use the
                    # openml library's download path, which doesn't enforce it.
                    if 'md5 checksum' in str(e):
                        import openml as _oml
                        _ds = _oml.datasets.get_dataset(did, download_data=True)
                        _X, _y, _, _ = _ds.get_data(target=_ds.default_target_attribute)
                        class _D:  # mimic the sklearn Bunch fields used below
                            pass
                        d = _D(); d.data = pd.DataFrame(_X)
                        d.target = (pd.Series(_np.asarray(_y).reshape(-1)) if _y is not None else None)
                        return d
                    raise

        data = _fetch_openml_robust(dataset_id)
        X = data.data
        y = data.target
        # Some OpenML datasets have no default target attribute -> y is None.
        # Fall back to the last column of X as the regression target.
        if y is None:
            X = pd.DataFrame(X)
            y = X.iloc[:, -1]
            X = X.iloc[:, :-1]
        # Convert target to numeric if needed
        y = pd.Series(y)
        if y.dtype == 'object' or isinstance(y.dtype, pd.CategoricalDtype):
            y = pd.to_numeric(y, errors='coerce')

    elif source == 'pmlb':
        X, y = _load_pmlb(dataset_config)

    elif source == 'keel':
        X, y = _load_keel(dataset_config)

    else:
        raise ValueError(f"Unknown dataset source: {source}")

    # ----- common preprocessing -----
    X = pd.DataFrame(X)
    y = pd.Series(y)

    # Convert target to numeric if needed
    if y.dtype == 'object' or isinstance(y.dtype, pd.CategoricalDtype):
        y = pd.to_numeric(y, errors='coerce')

    # Remove rows where target is NaN
    target_valid_mask = ~y.isna()
    X = X[target_valid_mask].reset_index(drop=True)
    y = y[target_valid_mask].reset_index(drop=True)


    # Convert categorical columns to numeric using label encoding
    for col in X.select_dtypes(exclude=[np.number]).columns:
        X[col] = pd.Categorical(X[col]).codes

    return X, y


def validate_datasets(datasets_config):
    """Filter out binary or few-class classification datasets (not regression).
    
    Called only when benchmark actually runs, not at import time.
    
    Parameters
    ----------
    datasets_config : list[dict]
        Unvalidated datasets from get_DATASETS_CONFIG()
    
    Returns
    -------
    list[dict]
        Filtered datasets suitable for regression benchmarking
    """
    print("\n" + "="*70)
    print("VALIDATING DATASETS (checking for binary or few-class classification)...")
    print("="*70)

    _classification_removed = []
    _validated_datasets = []

    for _ds in datasets_config:
        try:
            print(f"  Checking {_ds['name']:40s}...", end=" ", flush=True)
            _, _y = load_dataset(_ds)
            _n_unique = _y.nunique()
            if _is_classification(_y):
                print("❌ REMOVED (binary or few-class classification - less than 4 labels)")
                _classification_removed.append(_ds['name'])
            else:
                print(f"✓ OK ({_n_unique} unique values)")
                _validated_datasets.append(_ds)
        except Exception as e:
            print(f"⚠️  KEPT (validation error: {str(e)[:40]})")
            _validated_datasets.append(_ds)

    print("\n" + "="*70)
    if _classification_removed:
        print(f"⚠️  REMOVED {len(_classification_removed)} BINARY OR FEW-CLASS CLASSIFICATION DATASETS:")
        for _name in _classification_removed:
            print(f"    • {_name} (less than 4 y labels - not suitable for regression)")
    print("="*70)

    print(f"\n✓ FINAL CLEAN LIST: {len(_validated_datasets)} regression datasets")
    print("\nAll validated datasets in benchmark:")
    for _i, _ds in enumerate(_validated_datasets, 1):
        print(f"  {_i:3d}. {_ds['name']}")
    
    return _validated_datasets
