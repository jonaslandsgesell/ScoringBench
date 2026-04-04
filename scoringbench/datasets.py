from __future__ import annotations

import io
import re
import zipfile
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_iris, load_diabetes, load_breast_cancer, load_wine, load_digits,
    fetch_openml,
)
from sklearn.impute import SimpleImputer
import openml

# ---------------------------------------------------------------------------
# Cache directory for downloaded datasets
# ---------------------------------------------------------------------------
CACHE_DIR = Path.home() / '.cache' / 'scoringbench' / 'datasets'


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
    """Normalise a dataset name for deduplication comparison."""
    name = re.sub(r'^\d+[_\-]?', '', name)          # strip leading numeric prefix
    return re.sub(r'[^a-z0-9]', '', name.lower())    # lowercase, alphanum only


def _is_duplicate(new_name: str, existing_names: set[str],
                  dedup_keys: list[str] | None = None) -> bool:
    """Return True if *new_name* (or any of its *dedup_keys*) already exists."""
    norm_new = _normalize_name(new_name)

    # Exact normalised match
    if norm_new in existing_names:
        return True

    # Substring match (both directions) for names > 4 chars
    for existing in existing_names:
        if len(norm_new) > 4 and len(existing) > 4:
            if norm_new in existing or existing in norm_new:
                return True

    # Check explicit dedup keys
    if dedup_keys:
        for key in dedup_keys:
            norm_key = re.sub(r'[^a-z0-9]', '', key.lower())
            if norm_key in existing_names:
                return True
            for existing in existing_names:
                if len(norm_key) > 4 and len(existing) > 4:
                    if norm_key in existing or existing in norm_key:
                        return True

    return False


# ---------------------------------------------------------------------------
# OpenML suite loader (existing)
# ---------------------------------------------------------------------------
def generate_datasets_config_from_openml_suites(suite_ids):
    """Generate dataset configs from OpenML benchmark suites."""
    datasets_config = []
    seen_dataset_ids = set()

    print("Fetching datasets from OpenML suites...")

    for suite_id in suite_ids:
        try:
            suite = openml.study.get_suite(suite_id)
            print(f"  Processing OpenML Suite {suite_id} ('{suite.name}')...")
            for task_id in suite.tasks:
                try:
                    task = openml.tasks.get_task(task_id)
                    dataset_id = task.dataset_id

                    if dataset_id not in seen_dataset_ids:
                        dataset = openml.datasets.get_dataset(
                            dataset_id, download_data=False
                        )
                        dataset_name = dataset.name
                        abbr = ''.join(
                            [word[0] for word in dataset_name.split('_') if word]
                        ).upper()[:3]

                        datasets_config.append({
                            'name': dataset_name,
                            'abbr': abbr,
                            'source': 'openml',
                            'id': dataset_id,
                        })
                        seen_dataset_ids.add(dataset_id)
                except Exception as e:
                    print(f"    Error processing task {task_id} "
                          f"in suite {suite_id}: {e}")
        except Exception as e:
            print(f"  Error retrieving suite {suite_id}: {e}")

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


def _load_uci(config: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Load a UCI dataset via the *ucimlrepo* package."""
    from ucimlrepo import fetch_ucirepo          # soft dependency

    uci_id = config['uci_id']
    print(f"  Fetching UCI dataset {uci_id} via ucimlrepo ...")
    dataset = fetch_ucirepo(id=uci_id)
    X = dataset.data.features
    y = dataset.data.targets

    # Handle multi-target datasets
    if isinstance(y, pd.DataFrame) and y.shape[1] > 1:
        target_col = config.get('target_col')
        if target_col and target_col in y.columns:
            y = y[target_col]
        else:
            y = y.iloc[:, 0]
    elif isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    return X, pd.Series(y, name='target')


def _load_kaggle(config: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Load a Kaggle dataset.  Requires *kaggle* CLI to be installed."""
    import subprocess

    name = config['name']
    kaggle_dataset = config['kaggle_dataset']
    target_col = config['target_col']

    cache_dir = CACHE_DIR / name
    cache_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(cache_dir.glob('*.csv'))
    if not csv_files:
        print(f"  Downloading {name} from Kaggle ({kaggle_dataset}) ...")
        try:
            subprocess.run(
                ['kaggle', 'datasets', 'download', '-d', kaggle_dataset,
                 '-p', str(cache_dir), '--unzip'],
                check=True, capture_output=True, text=True,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "kaggle CLI not found. Install with: pip install kaggle\n"
                "Then configure API credentials: "
                "https://github.com/Kaggle/kaggle-api#api-credentials"
            ) from None
        csv_files = sorted(cache_dir.glob('*.csv'))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found for {name} after download"
        )

    # Use specified csv_file or the largest CSV available
    csv_file = config.get('csv_file')
    if csv_file:
        df = pd.read_csv(cache_dir / csv_file)
    else:
        csv_files.sort(key=lambda f: f.stat().st_size, reverse=True)
        df = pd.read_csv(csv_files[0])

    # Strip whitespace from column names (common in Kaggle CSVs)
    df.columns = df.columns.str.strip()
    target_col = target_col.strip()

    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def _load_dcc(config: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Load a dataset from DCC / Luis Torgo's regression collection."""
    url = config['url']
    name = config['name']

    ext = '.tar.gz' if url.endswith('.tar.gz') else '.tgz'
    cache_path = _ensure_cached(name, url, f"{name}{ext}")

    extract_dir = CACHE_DIR / name / 'extracted'
    if not extract_dir.exists():
        extract_dir.mkdir(parents=True)
        with tarfile.open(cache_path, 'r:gz') as tar:
            tar.extractall(extract_dir)

    # Collect all data / test files (skip metadata)
    skip_suffixes = {'.names', '.domain', '.gz', '.tgz'}
    skip_names = {'README', 'readme'}
    all_files = [
        f for f in extract_dir.rglob('*')
        if f.is_file()
        and f.suffix not in skip_suffixes
        and f.name not in skip_names
        and f.suffix in ('.data', '.test', '.csv', '')
    ]

    if not all_files:
        raise FileNotFoundError(f"No data files found in {extract_dir}")

    dfs: list[pd.DataFrame] = []
    for data_file in all_files:
        try:
            df = pd.read_csv(data_file, sep=r'\s+', header=None, engine='python')
            dfs.append(df)
        except Exception:
            try:
                df = pd.read_csv(data_file, header=None)
                dfs.append(df)
            except Exception:
                continue

    if not dfs:
        raise ValueError(f"Could not parse any data files for {name}")

    df = pd.concat(dfs, ignore_index=True)

    # Last column is always the target for DCC regression datasets
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X.columns = [f"feature_{i}" for i in range(X.shape[1])]
    y.name = 'target'
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
    # ---- UCI datasets (loaded via ucimlrepo, ID from URL path) ----
    {'name': 'Abalone', 'source': 'uci', 'uci_id': 1,
     'dedup_keys': ['abalone']},
    {'name': 'Student_Performance', 'source': 'uci', 'uci_id': 320,
     'target_col': 'G3'},
    {'name': 'Infrared_Thermography_Temperature', 'source': 'uci',
     'uci_id': 925},
    {'name': 'Parkinsons_Telemonitoring', 'source': 'uci', 'uci_id': 189,
     'target_col': 'total_UPDRS'},
    {'name': 'Energy_Efficiency', 'source': 'uci', 'uci_id': 242,
     'target_col': 'Y1',
     'dedup_keys': ['energy-efficiency', 'energy_efficiency']},
    {'name': 'QsarFishToxicity', 'source': 'uci', 'uci_id': 504},
    {'name': 'concrete_compressive_strength', 'source': 'uci', 'uci_id': 165},
    {'name': 'PRODUCTIVITY', 'source': 'uci', 'uci_id': 597,
     'target_col': 'actual_productivity'},
    {'name': 'CCPP', 'source': 'uci', 'uci_id': 294,
     'target_col': 'PE'},
    {'name': 'AIRFOIL', 'source': 'uci', 'uci_id': 291,
     'dedup_keys': ['airfoil_self_noise', 'airfoilselfnoise']},
    {'name': 'TETOUAN', 'source': 'uci', 'uci_id': 849},
    {'name': 'BIAS_CORRECTION', 'source': 'uci', 'uci_id': 514,
     'target_col': 'Next_Tmax'},
    {'name': 'APARTMENTS', 'source': 'uci', 'uci_id': 555},
    {'name': 'NewsPopularity', 'source': 'uci', 'uci_id': 332,
     'target_col': 'shares',
     'dedup_keys': ['onlinenewspopularity', 'OnlineNewsPopularity']},

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

    # ---- Kaggle datasets (requires kaggle CLI + API key) ----
    {'name': 'MedicalCost', 'source': 'kaggle',
     'kaggle_dataset': 'mirichoi0218/insurance',
     'target_col': 'charges'},
    {'name': 'Vehicle', 'source': 'kaggle',
     'kaggle_dataset': 'nehalbirla/vehicle-dataset-from-cardekho',
     'target_col': 'selling_price'},
    {'name': 'LifeExpectancy', 'source': 'kaggle',
     'kaggle_dataset': 'kumarajarshi/life-expectancy-who',
     'target_col': 'Life expectancy'},
    {'name': 'BigMartSales', 'source': 'kaggle',
     'kaggle_dataset': 'brijbhushannanda1979/bigmart-sales-data',
     'target_col': 'Item_Outlet_Sales'},
    {'name': 'VideoGameSales', 'source': 'kaggle',
     'kaggle_dataset': 'gregorut/videogamesales',
     'target_col': 'Global_Sales'},

    # ---- DCC / Luis Torgo regression datasets ----
    {'name': 'CalHousing', 'source': 'dcc',
     'url': 'https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.tgz',
     'dedup_keys': ['californiahousing', 'california:housing', 'calhousing']},
    {'name': 'Ailerons', 'source': 'dcc',
     'url': 'https://www.dcc.fc.up.pt/~ltorgo/Regression/ailerons.tgz'},
    {'name': 'DeltaElevators', 'source': 'dcc',
     'url': 'https://www.dcc.fc.up.pt/~ltorgo/Regression/delta_elevators.tgz'},
    {'name': 'Pole', 'source': 'dcc',
     'url': 'https://www.dcc.fc.up.pt/~ltorgo/Regression/pol.tgz'},
    {'name': 'Kinematics', 'source': 'dcc',
     'url': 'https://www.dcc.fc.up.pt/~ltorgo/Regression/kinematics.tar.gz'},

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


# ---------------------------------------------------------------------------
# Build final DATASETS_CONFIG
# ---------------------------------------------------------------------------
# Start with OpenML suite datasets
DATASETS_CONFIG = generate_datasets_config_from_openml_suites(
    suite_ids=[297, 299, 269]
)

# Merge TabRegSet-101 datasets, skipping duplicates
_existing_names = {_normalize_name(d['name']) for d in DATASETS_CONFIG}
_added = 0
for _ds in TABREGSET_DATASETS:
    _dedup_keys = _ds.get('dedup_keys')
    if not _is_duplicate(_ds['name'], _existing_names, _dedup_keys):
        _abbr = ''.join(
            [w[0] for w in _ds['name'].split('_') if w]
        ).upper()[:3]
        _ds_config = {**_ds, 'abbr': _abbr}
        _ds_config.pop('dedup_keys', None)
        DATASETS_CONFIG.append(_ds_config)
        _existing_names.add(_normalize_name(_ds['name']))
        _added += 1
    else:
        print(f"  Skipping {_ds['name']} "
              f"(already in benchmark via OpenML suites)")

print(f"\nAdded {_added} TabRegSet-101 datasets "
      f"(skipped {len(TABREGSET_DATASETS) - _added} duplicates)")
print(f"Total datasets in benchmark: {len(DATASETS_CONFIG)}")


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
def load_dataset(dataset_config: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Load and preprocess a dataset from any supported source.

    Supported sources: sklearn, openml, pmlb, uci, kaggle, dcc, keel.
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
        data = fetch_openml(data_id=dataset_id, as_frame=True, parser='auto')
        X = data.data
        y = data.target
        # Convert target to numeric if needed
        if y.dtype == 'object' or isinstance(y.dtype, pd.CategoricalDtype):
            y = pd.to_numeric(y, errors='coerce')

    elif source == 'pmlb':
        X, y = _load_pmlb(dataset_config)

    elif source == 'uci':
        X, y = _load_uci(dataset_config)

    elif source == 'kaggle':
        X, y = _load_kaggle(dataset_config)

    elif source == 'dcc':
        X, y = _load_dcc(dataset_config)

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

    # Impute missing feature values
    if X.isna().sum().sum() > 0:
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns

        if len(numeric_cols) > 0:
            numeric_imputer = SimpleImputer(strategy='median')
            X[numeric_cols] = numeric_imputer.fit_transform(X[numeric_cols])

        if len(categorical_cols) > 0:
            categorical_imputer = SimpleImputer(strategy='most_frequent')
            X[categorical_cols] = categorical_imputer.fit_transform(
                X[categorical_cols]
            )

    # Convert categorical columns to numeric using label encoding
    for col in X.select_dtypes(exclude=[np.number]).columns:
        X[col] = pd.Categorical(X[col]).codes

    # Apply per-dataset sample size limit
    sample_size = dataset_config.get('sample_size')
    if sample_size and len(X) > sample_size:
        indices = np.random.choice(len(X), size=sample_size, replace=False)
        X = X.iloc[indices].reset_index(drop=True)
        y = y.iloc[indices].reset_index(drop=True)

    return X, y
