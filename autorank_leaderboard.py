#!/usr/bin/env python3
"""
Leaderboard (Fold-Level Version) - Statistical Ranking
Two ranking approaches:
  1. rank_with_autorank   — Autorank-based statistical comparison with CD diagrams.
  2. rank_with_standardized_scores — Magnitude-based: robust z-score (median/MAD)
     standardization per dataset, then average across datasets.
"""
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from autorank import autorank, plot_stats, create_report, latex_table
from scipy import stats
import importlib.util
import io
import sys
import json


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
    Only rows complete across all models are kept.
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

    if not data: return None

    df_metric = pd.DataFrame(data)
    # Step 1: aggregate across folds → (dataset, model, avg_score)
    df_agg = df_metric.groupby(['dataset', 'model'])['score'].mean().reset_index()
    pivot = df_agg.pivot(index='dataset', columns='model', values='score')
    return pivot.dropna(how='any')


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
# Ranking approach 2: Robust standardized scores (median / MAD)
# ---------------------------------------------------------------------------

def rank_with_standardized_scores(pivot, hib):
    """
    Magnitude-based ranking using robust z-scores per dataset.

        Steps:
            1. pivot already contains (dataset, model, avg_score) — fold aggregation done upstream.
            2. Per dataset, compute median and MAD across models. IMPORTANT: aggregate correlated
                 observations first to avoid pseudoreplication (e.g. group correlated samples by
                 block or fold and compute block means before computing dataset-level statistics).
                 Failure to account for correlation leads to underestimated variance and overly
                 narrow confidence intervals (see Lazic 2010 https://bmcneurosci.biomedcentral.com/articles/10.1186/1471-2202-11-5).
            3. Standardize each model's score: z = (score - dataset_median) / (dataset_MAD + eps).
            4. Average standardized scores across datasets.
            5. Rank: if higher_is_better, higher avg_z → rank 1; else lower avg_z → rank 1.

    Returns a DataFrame with columns: rank, model, avg_z, median_z, std_z, n_datasets.
    """
    eps = 1e-8  # prevent division by zero when MAD == 0

    # Step 2 & 3: per-dataset robust standardization
    dataset_medians = pivot.median(axis=1)       # shape: (n_datasets,)
    dataset_mads = pivot.apply(                   # shape: (n_datasets,)
        lambda row: np.median(np.abs(row - np.median(row))), axis=1
    )

    z_scores = pivot.subtract(dataset_medians, axis=0).divide(
        dataset_mads + eps, axis=0
    )  # shape: (n_datasets, n_models)

    # Step 4: average standardized scores across datasets
    avg_z = z_scores.mean(axis=0)
    std_z = z_scores.std(axis=0)
    median_z = z_scores.median(axis=0)
    n_datasets = z_scores.shape[0]

    df = pd.DataFrame({
        'model': avg_z.index,
        'avg_z': avg_z.values,
        'median_z': median_z.values,
        'std_z': std_z.values,
        'n_datasets': n_datasets,
    })

    # Step 5: rank — higher avg_z is better when hib=True, lower when hib=False
    df = df.sort_values('avg_z', ascending=not hib).reset_index(drop=True)
    df.insert(0, 'rank', df.index + 1)
    return df


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def _rank_correlation(autorank_rankedDF, std_rankedDF):
    """
    Compute Pearson correlation between the rank vectors of both methods.
    Models are aligned by name; only models present in both are used.
    Returns (pearson_r, p_value, n_models) or (None, None, 0) on failure.
    """
    try:
        ar = autorank_rankedDF[['model', 'rank']].rename(columns={'rank': 'rank_autorank'})
        st = std_rankedDF[['model', 'rank']].rename(columns={'rank': 'rank_std'})
        merged = ar.merge(st, on='model')
        n = len(merged)
        if n < 3:
            return None, None, n
        r, p = stats.pearsonr(merged['rank_autorank'], merged['rank_std'])
        return float(r), float(p), n
    except Exception:
        return None, None, 0


def save_merged_cd_data(out_dir, metric, autorank_rankedDF, autorank_result, std_rankedDF, order, hib):
    """
    Write a single JSON file with two top-level sections:
      - "autorank": statistical results from autorank
      - "standardized_scores": magnitude-based robust z-score results
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

    # --- standardized scores section ---
    std_section = {
        "description": (
            "Robust z-score standardization per dataset (median/MAD), "
            "averaged across datasets. Lower avg_z = better when higher_is_better=False."
        ),
        "models": [],
    }
    for _, row in std_rankedDF.iterrows():
        std_section["models"].append({
            "rank": int(row['rank']),
            "name": row['model'],
            "avg_z": float(row['avg_z']),
            "median_z": float(row['median_z']),
            "std_z": float(row['std_z']),
            "n_datasets": int(row['n_datasets']),
        })

    pearson_r, pearson_p, n_models_corr = _rank_correlation(autorank_rankedDF, std_rankedDF)

    n_datasets = int(std_rankedDF['n_datasets'].iloc[0]) if not std_rankedDF.empty else None

    cd_data = {
        "metric": metric,
        "order": order,
        "higher_is_better": hib,
        "n_datasets": n_datasets,
        "rank_correlation": {
            "method": "pearson",
            "between": ["rank_autorank", "rank_standardized_scores"],
            "r": pearson_r,
            "pvalue": pearson_p,
            "n_models": n_models_corr,
        },
        "autorank": autorank_section,
        "standardized_scores": std_section,
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

    # Basic table generation
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
        pivot = load_metric_matrix(root, metric)
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

        # --- Approach 2: Robust standardized scores ---
        try:
            std_rankedDF = rank_with_standardized_scores(pivot, hib)

            print(f"\n--- {metric} (Standardized Scores, median/MAD) ---")
            print(std_rankedDF[["rank", "model", "avg_z", "median_z", "std_z"]].to_string(index=False))

        except Exception as e:
            print(f"Warning: standardized score ranking failed for {metric}: {e}")
            std_rankedDF = pd.DataFrame(columns=["rank", "model", "avg_z", "median_z", "std_z", "n_datasets"])

        # --- Save merged JSON ---
        save_merged_cd_data(out_dir, metric, autorank_rankedDF, autorank_result, std_rankedDF, order, hib)


if __name__ == "__main__":
    main()