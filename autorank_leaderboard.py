#!/usr/bin/env python3
"""
Leaderboard (Fold-Level Version) - Statistical Ranking
Two ranking approaches:
  1. rank_with_autorank   — Autorank-based statistical comparison with CD diagrams.
  2. rank_with_ranx — Ranx-based ranking using magnitude-based comparison with statistical significance testing.
"""
import argparse
import importlib.util
import io
import json
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from autorank import autorank, plot_stats, create_report, latex_table
from scipy import stats
from ranx import Run, compare


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_write_latex_writer():
    try:
        base = os.path.dirname(__file__)
        path = os.path.join(base, 'scoringbench', 'latex_tables.py')
        spec = importlib.util.spec_from_file_location('scoringbench_latex_tables', path)
        if spec is None or spec.loader is None: return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, 'write_latex_tables', None)
    except Exception:
        return None


def _collect_all_rows(root):
    rows = []
    if not os.path.exists(root): return rows
    for entry in os.listdir(root):
        entry_path = os.path.join(root, entry)
        if os.path.isfile(entry_path) and entry.endswith('.parquet'):
            model_name = entry[:-8]
            try:
                df = pd.read_parquet(entry_path)
                if df.empty: continue
                if 'model' not in df.columns: df['model'] = model_name
                for _, r in df.iterrows():
                    row = r.to_dict()
                    if 'fold' in row and isinstance(row['fold'], (int, float)):
                        row['fold'] = f"fold_{int(row['fold'])}"
                    rows.append(row)
            except Exception: continue
    return rows


def load_metric_matrix(root, metric):
    """
    Load metric data and return a pivot table (rows=datasets, columns=models)
    where each cell is the mean score across folds for that (dataset, model) pair.
    
    Robust handling: 
    - Only includes models that have at least 90% dataset coverage for this metric.
    - Models with < 90% coverage are excluded with a warning.
    - Datasets where any of the remaining models is missing are dropped.
    
    Returns: (pivot_table, included_models) or (None, None) if no data available.
    """
    data = []
    for entry in os.listdir(root):
        entry_path = os.path.join(root, entry)
        if not os.path.isfile(entry_path) or not entry.endswith('.parquet'):
            continue
        model = entry[:-8]
        try:
            df = pd.read_parquet(entry_path)
            if df.empty or metric not in df.columns: continue
            for _, row in df.iterrows():
                data.append({
                    "dataset": row.get('dataset', 'unknown'),
                    "fold": row.get('fold', 0),
                    "model": model,
                    "score": float(row[metric]),
                })
        except Exception: continue

    if not data: return None, None

    df_metric = pd.DataFrame(data)
    # Step 1: aggregate across folds → (dataset, model, avg_score)
    df_agg = df_metric.groupby(['dataset', 'model'])['score'].mean().reset_index()
    pivot = df_agg.pivot(index='dataset', columns='model', values='score')
    
    # Step 2: Identify models with any data for this metric
    models_with_data = pivot.columns[pivot.notna().any()].tolist()
    if not models_with_data:
        return None, None
    
    # Step 3: Filter models by 90% dataset coverage threshold
    total_datasets = len(pivot)
    coverage_threshold = 0.90
    models_sufficient_coverage = []
    models_dropped = []
    
    for model in models_with_data:
        coverage_count = pivot[model].notna().sum()
        coverage_pct = coverage_count / total_datasets if total_datasets > 0 else 0.0
        
        if coverage_pct >= coverage_threshold:
            models_sufficient_coverage.append(model)
        else:
            models_dropped.append((model, coverage_count, total_datasets, coverage_pct * 100))
    
    # Print warnings for dropped models
    if models_dropped:
        print(f"\n⚠️  WARNING: Dropping models with < 90% dataset coverage on metric '{metric}':")
        for model, covered, total, pct in models_dropped:
            print(f"   • {model}: {covered}/{total} datasets ({pct:.1f}%)")
    
    if not models_sufficient_coverage:
        return None, None
    
    # Step 4: Keep only models with sufficient coverage, then remove datasets with any NaN
    pivot_filtered = pivot[models_sufficient_coverage].dropna(how='any')
    
    if pivot_filtered.empty:
        return None, None
    
    return pivot_filtered, models_sufficient_coverage


def load_metric_long_format(root, metric):
    """
    Load metric data in long format (rows=fold-level for each dataset-model pair).
    Returns DataFrame with columns: ['dataset', 'model', 'fold', 'score']
    
    Robust handling: Includes all folds from models that have any data for this metric.
    
    Returns: (df_long, models_list) or (None, None) if no data available.
    """
    data = []
    models_seen = set()
    for entry in os.listdir(root):
        entry_path = os.path.join(root, entry)
        if not os.path.isfile(entry_path) or not entry.endswith('.parquet'):
            continue
        model = entry[:-8]
        try:
            df = pd.read_parquet(entry_path)
            if df.empty or metric not in df.columns: continue
            models_seen.add(model)
            for _, row in df.iterrows():
                data.append({
                    "dataset": row.get('dataset', 'unknown'),
                    "fold": row.get('fold', 0),
                    "model": model,
                    "score": float(row[metric]),
                })
        except Exception: continue

    if not data: return None, None
    return pd.DataFrame(data), list(models_seen)


# ---------------------------------------------------------------------------
# Ranking approach 1: Autorank
# ---------------------------------------------------------------------------

def rank_with_autorank(pivot, metric, order, hib, alpha):
    """
    Run autorank on the pivot table and return (rankedDF, result).

    rankedDF columns: rank, model, meanrank, + any columns autorank provides.
    Returns None on failure.
    """
    result = autorank(pivot, alpha=alpha, order=order)

    if not hasattr(result, 'rankdf') or result.rankdf is None:
        return None, None

    rankedDF = result.rankdf.copy()
    rankedDF.index.name = 'model'
    rankedDF = rankedDF.reset_index()
    # autorank assigns lower meanrank to better models regardless of order
    rankedDF = rankedDF.sort_values('meanrank', ascending=True).reset_index(drop=True)
    rankedDF.insert(0, 'rank', rankedDF.index + 1)
    return rankedDF, result


# ---------------------------------------------------------------------------
# Ranking approach 2: Ranx magnitude-based ranking
# ---------------------------------------------------------------------------

def rank_with_ranx_aggregated(df_long, hib=True):
    """
    Ranks models by first aggregating folds (to avoid pseudoreplication)
    and then performing a cross-dataset magnitude-stable comparison.
    
    Stability improvements:
    - Filters models by 90% dataset coverage (matches autorank's robustness)
    - Filters datasets by complete model coverage (no missing models)
    - Handles NaN values robustly when averaging normalized scores
    """
    try:
        # 1. Aggregate folds to the Dataset level (The "Anti-Pseudoreplication" step)
        df_agg = df_long.groupby(['model', 'dataset'])['score'].mean().reset_index()

        # 2. STABILITY FILTER: Apply coverage thresholds (matching autorank approach)
        pivot_temp = df_agg.pivot(index='dataset', columns='model', values='score')
        total_datasets = len(pivot_temp)
        coverage_threshold = 0.90
        
        # Identify models with 90%+ dataset coverage
        models_sufficient_coverage = []
        for model in pivot_temp.columns:
            coverage_count = pivot_temp[model].notna().sum()
            coverage_pct = coverage_count / total_datasets if total_datasets > 0 else 0.0
            if coverage_pct >= coverage_threshold:
                models_sufficient_coverage.append(model)
        
        if not models_sufficient_coverage:
            print("Warning: No models have 90%+ dataset coverage for ranx ranking")
            return pd.DataFrame(columns=["rank", "model", "mean_std_diff", "n_datasets"])
        
        # Filter data to only include models with sufficient coverage
        df_agg = df_agg[df_agg['model'].isin(models_sufficient_coverage)]
        
        # Filter to only datasets where ALL included models have data (complete case analysis)
        pivot_filtered = pivot_temp[models_sufficient_coverage].dropna(how='any')
        valid_datasets = set(pivot_filtered.index)
        df_agg = df_agg[df_agg['dataset'].isin(valid_datasets)]
        
        if df_agg.empty:
            print("Warning: No complete dataset-model pairs after filtering")
            return pd.DataFrame(columns=["rank", "model", "mean_std_diff", "n_datasets"])

        # 3. Normalize Magnitude per dataset
        def standardize(x):
            return (x - x.mean()) / (x.std() + 1e-9)

        df_agg['norm_score'] = df_agg.groupby('dataset')['score'].transform(standardize)

        # If lower is better, negate so higher normalized scores are better
        if not hib:
            df_agg['norm_score'] = -df_agg['norm_score']

        # 4. Create ranx Runs
        runs = []
        for model_name in models_sufficient_coverage:
            model_data = df_agg[df_agg['model'] == model_name]
            
            # ranx format: { dataset_id: { model_id: normalized_score } }
            run_dict = {
                str(row.dataset): {model_name: float(row.norm_score)} 
                for _, row in model_data.iterrows()
                if pd.notna(row.norm_score)  # Skip NaN scores
            }
            
            if run_dict:  # Only include model if it has valid scores
                runs.append(Run(run_dict, name=model_name))

        if not runs:
            print("Warning: No valid runs after filtering NaNs")
            return pd.DataFrame(columns=["rank", "model", "mean_std_diff", "n_datasets"])

        # 5. Extract Leaderboard
        leaderboard = []
        for r in runs:
            run_dict = r.to_dict()
            if run_dict:
                # Average normalized scores, skipping NaNs
                scores = [float(next(iter(v.values()))) for v in run_dict.values()]
                valid_scores = [s for s in scores if not np.isnan(s)]
                avg_norm_score = np.mean(valid_scores) if valid_scores else 0.0
                n_datasets_valid = len(valid_scores)
            else:
                avg_norm_score = 0.0
                n_datasets_valid = 0
            
            leaderboard.append({
                'model': r.name,
                'mean_std_diff': float(avg_norm_score),
                'n_datasets': n_datasets_valid
            })

        results = pd.DataFrame(leaderboard).sort_values('mean_std_diff', ascending=False).reset_index(drop=True)
        results.insert(0, 'rank', results.index + 1)
        
        return results
    
    except Exception as e:
        print(f"Ranx aggregated ranking error: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def _rank_correlation(autorank_rankedDF, ranx_rankedDF):
    """
    Compute Pearson correlation between the rank vectors of both methods.
    Models are aligned by name; only models present in both are used.
    Returns (pearson_r, p_value, n_models) or (None, None, 0) on failure.
    """
    try:
        # Handle empty dataframes
        if autorank_rankedDF.empty or ranx_rankedDF.empty:
            print("Warning: One or both ranking dataframes are empty, skipping correlation")
            return None, None, 0
        
        # Check if required columns exist
        if 'model' not in autorank_rankedDF.columns or 'rank' not in autorank_rankedDF.columns:
            print("Warning: Autorank dataframe missing required columns")
            return None, None, 0
        
        if 'model' not in ranx_rankedDF.columns or 'rank' not in ranx_rankedDF.columns:
            print("Warning: Ranx dataframe missing required columns")
            return None, None, 0
        
        ar = autorank_rankedDF[['model', 'rank']].rename(columns={'rank': 'rank_autorank'})
        rx = ranx_rankedDF[['model', 'rank']].rename(columns={'rank': 'rank_ranx'})
        merged = ar.merge(rx, on='model')
        n = len(merged)
        
        if n < 3:
            print(f"Warning: Insufficient overlapping models ({n}) for correlation")
            return None, None, n
        
        r, p = stats.pearsonr(merged['rank_autorank'], merged['rank_ranx'])
        print(f"Rank correlation: r={r:.3f}, p={p:.4f}, n_models={n}")
        return float(r), float(p), n
    except Exception as e:
        print(f"Error computing rank correlation: {e}")
        return None, None, 0


def save_merged_cd_data(out_dir, metric, autorank_rankedDF, autorank_result, ranx_rankedDF, order, hib):
    """
    Write a single JSON file with two top-level sections:
      - "autorank": statistical results from autorank
      - "ranx": ranx-based ranking with magnitude scores
    Also includes Pearson correlation between both ranking methods.
    """

    # --- autorank section ---
    autorank_section = {
        "alpha": autorank_result.alpha,
        "cd": float(autorank_result.cd) if autorank_result.cd is not None else None,
        "pvalue": float(autorank_result.pvalue) if autorank_result.pvalue is not None else None,
        "omnibus_test": autorank_result.omnibus,
        "posthoc_test": autorank_result.posthoc,
        "models": [],
    }
    for _, row in autorank_rankedDF.iterrows():
        def _f(key):
            v = row.get(key)
            return float(v) if v is not None and pd.notna(v) else None

        autorank_section["models"].append({
            "rank": int(row['rank']),
            "name": row['model'],
            "meanrank": float(row['meanrank']),
            "mean": _f('mean'),
            "median": _f('median'),
            "std": _f('std'),
            "mad": _f('mad'),
            "ci_lower": _f('ci_lower'),
            "ci_upper": _f('ci_upper'),
            "effect_size": _f('effect_size'),
            "magnitude": row.get('magnitude', 'unknown'),
            "effect_size_above": _f('effect_size_above'),
            "magnitude_above": row.get('magnitude_above', 'unknown'),
        })

    # --- Ranx section ---
    ranx_section = {
        "description": (
            "Ranx-based magnitude ranking with dataset normalization. "
            "Averages performance across datasets for stable model comparison. "
            "Includes standard deviation and dataset coverage metrics."
        ),
        "models": [],
    }
    for _, row in ranx_rankedDF.iterrows():
        ranx_section["models"].append({
            "rank": int(row['rank']),
            "name": row['model'],
            "mean_std_diff": float(row.get('mean_std_diff', 0)),
            "n_datasets": int(row.get('n_datasets', 0)),
        })

    pearson_r, pearson_p, n_models_corr = _rank_correlation(autorank_rankedDF, ranx_rankedDF)

    n_datasets = None  # Not directly available in this scope

    cd_data = {
        "metric": metric,
        "order": order,
        "higher_is_better": hib,
        "n_datasets": n_datasets,
        "rank_correlation": {
            "method": "pearson",
            "between": ["rank_autorank", "rank_ranx"],
            "r": pearson_r,
            "pvalue": pearson_p,
            "n_models": n_models_corr,
        },
        "autorank": autorank_section,
        "ranx": ranx_section,
    }

    json_path = os.path.join(out_dir, f"cd_data_{metric}.json")
    with open(json_path, 'w') as f:
        json.dump(cd_data, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="output")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level for statistical tests")
    args = parser.parse_args()

    root = args.output
    if not os.path.exists(root): return

    # Attempt to aggregate output/raw → output/ before analysis.
    # This is best-effort: failures are printed but never stop the ranking.
    raw_dir = os.path.join(root, "raw")
    if os.path.isdir(raw_dir):
        try:
            _script_dir = os.path.dirname(os.path.abspath(__file__))
            if _script_dir not in sys.path:
                sys.path.insert(0, _script_dir)
            from aggregate_datasets import aggregate as _aggregate
            print(f"Aggregating raw parquets from {raw_dir} …")
            _aggregate(raw_dir=Path(raw_dir), out_dir=Path(root))
        except Exception as _exc:
            print(f"Warning: aggregation step failed ({_exc}), continuing with existing output/ files.")

    rows_all = _collect_all_rows(root)
    writer = _load_write_latex_writer()
    if writer: writer(root, rows_all)

    # Discover metrics
    discovered_metrics = set()
    for entry in os.listdir(root):
        if entry.endswith('.parquet'):
            try:
                df = pd.read_parquet(os.path.join(root, entry))
                for k in df.select_dtypes(include=['number']).columns:
                    if k not in ('fold', 'index'): discovered_metrics.add(k)
            except Exception: continue

    for metric in sorted(discovered_metrics):
        pivot, models_in_metric = load_metric_matrix(root, metric)
        if pivot is None: continue

        # Determine metric ordering and apply any score transformation
        is_coverage = metric.startswith("coverage_")
        hib = metric in ("r2", "dispersion")

        if is_coverage:
            # Transform to absolute distance from nominal level: lower = better
            try:
                target = int(metric.split("_")[1]) / 100.0
            except Exception:
                target = 0.5
            pivot = (pivot - target).abs()
            hib = False

        order = 'descending' if hib else 'ascending'
        out_dir = os.path.join(root, "figures", "leaderboard")
        os.makedirs(out_dir, exist_ok=True)

        # --- Approach 1: Autorank ---
        try:
            autorank_rankedDF, autorank_result = rank_with_autorank(pivot, metric, order, hib, args.alpha)

            if autorank_rankedDF is None:
                print(f"Skipping {metric}: autorank returned no rankdf")
                continue

            print(f"\n--- {metric} (Autorank) ---")
            print(f"  Models: {len(models_in_metric)} | Datasets: {len(pivot)}")
            print(f"Order: {order} | higher_is_better: {hib}")
            cols = [c for c in ("rank", "model", "mean", "std", "meanrank") if c in autorank_rankedDF.columns]
            print(autorank_rankedDF[cols].to_string(index=False))

            print(f"\nStatistical Report for {metric}:")
            create_report(autorank_result)

            # LaTeX table
            old_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()
            latex_table(autorank_result)
            sys.stdout = old_stdout
            latex_str = buffer.getvalue()
            if latex_str.strip():
                with open(os.path.join(out_dir, f"latex_table_{metric}.tex"), 'w') as f:
                    f.write(latex_str)

            # CD diagram
            fig = plot_stats(autorank_result, allow_insignificant=True)
            if fig is None:
                fig, ax = plt.subplots(figsize=(max(6, len(autorank_rankedDF) * 0.6), 4))
                colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(autorank_rankedDF))]
                ax.barh(autorank_rankedDF['model'][::-1], autorank_rankedDF['meanrank'][::-1], color=colors[::-1])
                ax.set_xlabel('Mean Rank (lower = better)')
                ax.set_title(f'{metric} — Mean Ranks')
                plt.tight_layout()
            else:
                if hasattr(fig, 'get_figure'):
                    fig = fig.get_figure()
            if fig is not None:
                plt.savefig(os.path.join(out_dir, f"cd_diagram_{metric}.png"), dpi=150, bbox_inches='tight')
            plt.close('all')

        except Exception as e:
            print(f"Skipping {metric} (autorank error): {e}")
            continue

        # --- Approach 2: Ranx magnitude-based ranking ---
        try:
            df_long, models_in_long = load_metric_long_format(root, metric)
            if df_long is None:
                print(f"Warning: could not load long format data for {metric}")
                ranx_rankedDF = pd.DataFrame(columns=["rank", "model", "mean_std_diff", "n_datasets"])
            else:
                # Apply transformation if needed (for coverage metrics)
                if is_coverage:
                    try:
                        target = int(metric.split("_")[1]) / 100.0
                    except Exception:
                        target = 0.5
                    df_long['score'] = (df_long['score'] - target).abs()
                
                ranx_rankedDF = rank_with_ranx_aggregated(df_long, hib)

                print(f"\n--- {metric} (Ranx Magnitude Ranking) ---")
                print(ranx_rankedDF[["rank", "model", "mean_std_diff"]].to_string(index=False))

        except Exception as e:
            print(f"Warning: Ranx ranking failed for {metric}: {e}")
            ranx_rankedDF = pd.DataFrame(columns=["rank", "model", "mean_std_diff"])

        # --- Save merged JSON ---
        save_merged_cd_data(out_dir, metric, autorank_rankedDF, autorank_result, ranx_rankedDF, order, hib)


if __name__ == "__main__":
    main()