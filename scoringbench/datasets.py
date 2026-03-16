import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_diabetes, load_breast_cancer,load_wine, load_digits
from sklearn.impute import SimpleImputer

from sklearn.datasets import fetch_openml
# Dataset configurations
import openml

def generate_datasets_config_from_openml_suites(suite_ids):
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
                        dataset = openml.datasets.get_dataset(dataset_id, download_data=False)
                        dataset_name = dataset.name
                        # Generate a simple abbreviation. This might not be unique for all datasets.
                        abbr = ''.join([word[0] for word in dataset_name.split('_') if word]).upper()[:3]

                        datasets_config.append({
                            'name': dataset_name,
                            'abbr': abbr,
                            'source': 'openml',
                            'id': dataset_id
                        })
                        seen_dataset_ids.add(dataset_id)
                except Exception as e:
                    print(f"    Error processing task {task_id} in suite {suite_id}: {e}")
        except Exception as e:
            print(f"  Error retrieving suite {suite_id}: {e}")

    print("\nFinished populating DATASETS_CONFIG from OpenML suites.")
    return datasets_config

# Call the function to generate DATASETS_CONFIG
DATASETS_CONFIG = generate_datasets_config_from_openml_suites(suite_ids=[297, 299, 269])


# DATASETS_CONFIG = [
#     {'name': 'cpu', 'source': 'openml', 'id': 561},
#     {'name': 'auction_verification', 'source': 'openml', 'id': 44958},
#     {'name': 'autoMpg', 'source': 'openml', 'id': 196},
#     {'name': 'energy-efficiency', 'source': 'openml', 'id': 1472},
#     {'name': 'nyc-taxi-green-dec-2016', 'source': 'openml', 'id': 44065},
#     {'name': 'forest_fires', 'source': 'openml', 'id': 44962},
#     {'name': 'physiochemical_protein', 'source': 'openml', 'id': 44963},
#     {'name': 'airfoil_self_noise', 'source': 'openml', 'id': 44957},
#     {'name': 'kin8nm', 'source': 'openml', 'id': 189},
#     {'name': 'superconductivity', 'source': 'openml', 'id': 44964},
#     {'name': 'naval_propulsion_plant', 'source': 'openml', 'id': 44969},
#     {'name': 'synchronous_machine', 'source': 'openml', 'id': 44968},
#     {'name': 'wine_quality', 'source': 'openml', 'id': 43994},
#     {'name': 'isolet', 'source': 'openml', 'id': 300},
#     {'name': 'cpu_act', 'source': 'openml', 'id': 197},
#     {'name': 'black_friday', 'source': 'openml', 'id': 44057},
#     {'name': 'Brazilian_houses', 'source': 'openml', 'id': 44062},
#     {'name': 'Allstate_Claims_Severity', 'source': 'openml', 'id': 44060},
#     {'name': 'diamonds', 'source': 'openml', 'id': 44059},
#     {'name': 'Mercedes_Benz_Greener_Manufacturing', 'source': 'openml', 'id': 44061},
#     {'name': 'Bike_Sharing_Demand', 'source': 'openml', 'id': 44063},
#     {'name': 'house_sales', 'source': 'openml', 'id': 44066},
#     {'name': 'OnlineNewsPopularity', 'source': 'openml', 'id': 44064},
#     {'name': 'LoanDefaultPrediction', 'source': 'openml', 'id': 44067},
#     {'name': 'particulate-matter-ukair-2017', 'source': 'openml', 'id': 44068},
#     {'name': 'nyc-taxi-green-dec-2016', 'source': 'openml', 'id': 44143},
#     {'name': 'california:housing', 'source': 'openml', 'id': 43939},
#     {'name': 'tecator', 'source': 'openml', 'id': 505},
#     {'name': 'boston', 'source': 'openml', 'id': 531},
#     {'name': 'Moneyball', 'source': 'openml', 'id': 41021},
#     {'name': 'house_prices', 'source': 'openml', 'id': 42165},
#     {'name': 'cloud', 'source': 'openml', 'id': 210},
#     {'name': 'balloon', 'source': 'openml', 'id': 512},
#     {'name': 'abalone', 'source': 'openml', 'id': 183},
#     {'name': 'Buzzinsocialmedia_Twitter', 'source': 'openml', 'id': 4549},
#     {'name': 'cpu_small', 'source': 'openml', 'id': 227},
#     {'name': 'puma32H', 'source': 'openml', 'id': 308},
# ]

# Sklearn loader mapping
SKLEARN_LOADERS = {
    'load_iris': load_iris,
    'load_diabetes': load_diabetes,
    'load_breast_cancer': load_breast_cancer,
    'load_wine': load_wine,
    'load_digits': load_digits
}


def load_dataset(dataset_config: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Load and preprocess a dataset."""
    if dataset_config.get('source') == 'sklearn':
        loader_func = SKLEARN_LOADERS[dataset_config['loader']]
        data_sklearn = loader_func()
        X = pd.DataFrame(
            data_sklearn.data, 
            columns=data_sklearn.feature_names if hasattr(data_sklearn, 'feature_names') 
                    else [f"feature_{i}" for i in range(data_sklearn.data.shape[1])]
        )
        y = pd.Series(data_sklearn.target)
    else:
        dataset_id = dataset_config['id']
        data = fetch_openml(data_id=dataset_id, as_frame=True, parser='auto')
        X = data.data
        y = data.target
        
        # Convert target to numeric if needed
        if y.dtype == 'object' or pd.api.types.is_categorical_dtype(y):
            y = pd.to_numeric(y, errors='coerce')
    
    # Remove rows where target is NaN
    target_valid_mask = ~y.isna()
    X = X[target_valid_mask].reset_index(drop=True)
    y = y[target_valid_mask].reset_index(drop=True)
    
    # Impute missing values in features
    if X.isna().sum().sum() > 0:
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns
        
        if len(numeric_cols) > 0:
            numeric_imputer = SimpleImputer(strategy='median')
            X[numeric_cols] = numeric_imputer.fit_transform(X[numeric_cols])
        
        if len(categorical_cols) > 0:
            categorical_imputer = SimpleImputer(strategy='most_frequent')
            X[categorical_cols] = categorical_imputer.fit_transform(X[categorical_cols])
    
    # Convert categorical columns to numeric using label encoding
    for col in X.select_dtypes(exclude=[np.number]).columns:
        X[col] = pd.Categorical(X[col]).codes
    
    # Apply sample size limit if specified
    sample_size = dataset_config.get('sample_size')
    if sample_size and len(X) > sample_size:
        indices = np.random.choice(len(X), size=sample_size, replace=False)
        X = X.iloc[indices].reset_index(drop=True)
        y = y.iloc[indices].reset_index(drop=True)
    
    return X, y
