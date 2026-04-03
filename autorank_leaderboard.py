#!/usr/bin/env python3
"""
Leaderboard (Fold-Level Version) - Autorank Statistical Ranking
Uses autorank for rigorous statistical comparison of models with 
significance testing and critical difference diagrams.
"""
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from autorank import autorank, plot_stats, create_report, latex_table
import importlib.util
import io
import sys
import json

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

def _save_cd_diagram_data(out_dir, metric, result, rankedDF, order, hib):
    """Export CD diagram data to JSON for JavaScript visualization."""
    cd_data = {
        "metric": metric,
        "order": order,
        "higher_is_better": hib,
        "alpha": result.alpha,
        "cd": float(result.cd) if result.cd is not None else None,
        "pvalue": float(result.pvalue) if result.pvalue is not None else None,
        "omnibus_test": result.omnibus,
        "posthoc_test": result.posthoc,
        "models": []
    }
    
    for idx, row in rankedDF.iterrows():
        model_data = {
            "rank": int(row['rank']),
            "name": row['model'],
            "meanrank": float(row['meanrank']),
            "mean": float(row['mean']) if pd.notna(row.get('mean')) else None,
            "median": float(row.get('median')) if pd.notna(row.get('median')) else None,
            "std": float(row['std']) if pd.notna(row.get('std')) else None,
            "mad": float(row.get('mad')) if pd.notna(row.get('mad')) else None,
            "ci_lower": float(row.get('ci_lower')) if pd.notna(row.get('ci_lower')) else None,
            "ci_upper": float(row.get('ci_upper')) if pd.notna(row.get('ci_upper')) else None,
            "effect_size": float(row.get('effect_size')) if pd.notna(row.get('effect_size')) else None,
            "magnitude": row.get('magnitude', 'unknown'),
            "effect_size_above": float(row.get('effect_size_above')) if pd.notna(row.get('effect_size_above')) else None,
            "magnitude_above": row.get('magnitude_above', 'unknown'),
        }
        cd_data["models"].append(model_data)
    
    json_path = os.path.join(out_dir, f"cd_data_{metric}.json")
    with open(json_path, 'w') as f:
        json.dump(cd_data, f, indent=2)

def load_metric_matrix(root, metric):
    """Load metric data by dataset and model, aggregated across folds."""
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
                dataset = row.get('dataset', 'unknown')
                fold = row.get('fold', 0)
                data.append({
                    "dataset": dataset,
                    "fold": fold,
                    "model": model,
                    "score": float(row[metric])
                })
        except Exception: continue
    
    if not data: return None
    
    df_metric = pd.DataFrame(data)
    # Aggregate across folds: average score per dataset per model avoiding pseudo-replication treat dataset as observation
    df_agg = df_metric.groupby(['dataset', 'model'])['score'].mean().reset_index()
    # Pivot: rows=datasets, columns=models, values=averaged scores
    pivot = df_agg.pivot(index='dataset', columns='model', values='score')
    return pivot.dropna(how='any')

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
        
        # Determine metric ordering and any score transformation
        # 'ascending': lower score is better (loss, error, etc.)
        # 'descending': higher score is better (r2, dispersion, etc.)
        is_coverage = metric.startswith("coverage_")
        hib = metric in ("r2", "dispersion")

        if is_coverage:
            # Transform coverage to absolute distance from nominal level: minimize distance
            try:
                lvl = int(metric.split("_")[1])
                target = lvl / 100.0
            except Exception:
                target = 0.5
            pivot = (pivot - target).abs()
            hib = False  # lower distance = better

        order = 'descending' if hib else 'ascending'

        try:
            # Run autorank: rows=datasets (observations), columns=models (treatments)
            result = autorank(pivot, alpha=args.alpha, order=order)

            if not hasattr(result, 'rankdf') or result.rankdf is None:
                print(f"Skipping {metric}: result has no rankdf")
                continue

            # Extract ranking information (index = model name)
            rankedDF = result.rankdf.copy()
            # autorank always assigns lower meanrank to better models regardless of order,
            # so sort ascending by meanrank → rank 1 = best model
            rankedDF = rankedDF.sort_values('meanrank', ascending=True)
            rankedDF.index.name = 'model'
            rankedDF = rankedDF.reset_index()
            rankedDF.insert(0, 'rank', rankedDF.index + 1)

            # Save CSV
            out_dir = os.path.join(root, "figures", "leaderboard")
            os.makedirs(out_dir, exist_ok=True)
            # rankedDF.to_csv(os.path.join(out_dir, f"autorank_{metric}.csv"), index=False)

            print(f"\n--- {metric} (Autorank Results) ---")
            print(f"Order: {order} (higher better: {hib})")
            cols = [c for c in ("rank", "model", "mean", "std", "meanrank") if c in rankedDF.columns]
            print(rankedDF[cols].to_string(index=False))

            # Generate report
            print(f"\nStatistical Report for {metric}:")
            create_report(result)

            # Save LaTeX table (latex_table prints to stdout, not returns)
            old_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()
            latex_table(result)
            sys.stdout = old_stdout
            latex_str = buffer.getvalue()
            if latex_str.strip():
                latex_path = os.path.join(out_dir, f"latex_table_{metric}.tex")
                with open(latex_path, 'w') as f:
                    f.write(latex_str)

            # Save CD diagram data as JSON for JavaScript rendering
            _save_cd_diagram_data(out_dir, metric, result, rankedDF, order, hib)

            # Save critical difference diagram (or fallback bar chart)
            fig = plot_stats(result, allow_insignificant=True)
            fig_path = os.path.join(out_dir, f"cd_diagram_{metric}.png")
            if fig is None:
                # Fallback: bar chart of mean ranks
                fig, ax = plt.subplots(figsize=(max(6, len(rankedDF) * 0.6), 4))
                colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(rankedDF))]
                ax.barh(rankedDF['model'][::-1], rankedDF['meanrank'][::-1], color=colors[::-1])
                ax.set_xlabel('Mean Rank (lower = better)')
                ax.set_title(f'{metric} — Mean Ranks')
                plt.tight_layout()
            else:
                # plot_stats may return Axes; get figure if needed
                if hasattr(fig, 'get_figure'):
                    fig = fig.get_figure()
            if fig is not None:
                plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close('all')

        except Exception as e:
            print(f"Skipping {metric}: {e}")

if __name__ == "__main__":
    main()