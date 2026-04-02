import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "--relative",
    metavar="MODEL_NAME",
    default="tabpfn",
    help=(
        "Compute metrics relative to this baseline model (default: tabpfn). "
        "Non-R² metrics are reported as percent change vs. baseline; "
        "R² is reported as absolute change × 100 pp. "
        "Relative changes are computed per fold, then averaged."
    ),
)
parser.add_argument(
    "--median",
    action="store_true",
    default=False,
    help="Aggregate across datasets using median instead of mean (more robust to outlier datasets).",
)
parser.add_argument(
    "--output",
    metavar="OUTPUT_ROOT",
    default=None,
    help="Path to the results root (default: run on all subfolders of output/)",
)

args = parser.parse_args()
use_median = args.median
agg_fn = "median" if use_median else "mean"
agg_label = "Median" if use_median else "Average"

# Always run both relative and absolute modes
modes_to_run = ['relative', 'absolute']

# Discover output roots
def get_output_roots():
    # Use the provided output root or the default 'output' directory.
    if args.output:
        return [args.output]
    return ["output"]

output_roots = get_output_roots()

# Process both relative and absolute modes
for mode in modes_to_run:
    # Set mode-specific variables
    relative_baseline = args.relative if mode == 'relative' else None
    mode_label = f"relative to {relative_baseline}" if relative_baseline else "absolute"
    
    print(f"\n{'='*60}")
    print(f"Processing {mode_label} mode")
    print(f"{'='*60}")
    
    def _collect_all_rows(root_path):
        """Collect rows from either per-model parquet files or legacy JSON layout.

        Returns a list of dicts with keys including `dataset`, `fold`, `model` and metrics.
        """
        rows = []
        if not os.path.exists(root_path):
            return rows
        for entry in os.listdir(root_path):
            entry_path = os.path.join(root_path, entry)
            # Per-model parquet file
            if os.path.isfile(entry_path) and entry.endswith('.parquet'):
                model_name = entry[:-8]
                try:
                    df = pd.read_parquet(entry_path)
                except Exception:
                    continue
                if df.empty:
                    continue
                if 'model' not in df.columns:
                    df['model'] = model_name
                for _, r in df.iterrows():
                    row = r.to_dict()
                    # Normalize fold name to 'fold_<N>' if integer
                    if 'fold' in row and isinstance(row['fold'], (int, float)):
                        row['fold'] = f"fold_{int(row['fold'])}"
                    rows.append(row)
                continue

            # Legacy layout: root/<model>/<dataset>/fold_<N>/results.json
            if not os.path.isdir(entry_path):
                continue
            model_name = entry
            model_dir = entry_path
            datasets = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
            for dataset in datasets:
                dataset_dir = os.path.join(model_dir, dataset)
                folds = [f for f in os.listdir(dataset_dir) if f.startswith('fold_') and os.path.isdir(os.path.join(dataset_dir, f))]
                for fold in sorted(folds):
                    path = os.path.join(dataset_dir, fold, 'results.json')
                    if not os.path.exists(path):
                        continue
                    with open(path) as fh:
                        try:
                            res = json.load(fh)
                        except Exception:
                            continue
                    row = dict(dataset=dataset, fold=fold, model=model_name)
                    row.update(res)
                    rows.append(row)

        return rows

    for root in tqdm(output_roots, desc=f"Processing output roots ({mode})"):
        all_rows = _collect_all_rows(root)

        df = pd.DataFrame(all_rows)
        if df.empty:
            print(f"No results found in {root}, skipping.")
            continue

        # --- Relative mode: compute changes per (dataset, fold), then aggregate ---
        if relative_baseline is not None:
            if relative_baseline not in df["model"].values:
                raise ValueError(
                    f"Baseline model '{relative_baseline}' not found in results. "
                    f"Available models: {sorted(df['model'].unique())}"
                )
            numeric_cols = [c for c in df.columns if c not in ("dataset", "fold", "model")
                            and np.issubdtype(df[c].dtype, np.number)]
            baseline_df = (
                df[df["model"] == relative_baseline]
                .set_index(["dataset", "fold"])[numeric_cols]
                .rename(columns=lambda c: c + "_baseline")
            )
            df = df.join(baseline_df, on=["dataset", "fold"])
            for col in numeric_cols:
                bcol = col + "_baseline"
                if col == "r2":
                    # Absolute change × 100 pp; positive = better
                    df[col] = (df[col] - df[bcol]) * 100
                elif col == "dispersion":
                    # Dispersion: higher is favorable (per Tran et al.); flip the formula
                    # positive = better (higher dispersion)
                    df[col] = (df[col] - df[bcol]) / df[bcol].abs() * 100
                elif col == "cov_90":
                    # Coverage metric targeting 0.90; use distance-to-target formula
                    target = 0.90
                    df[col] = ((np.abs(target - df[bcol]) - np.abs(target - df[col])) / 
                               np.abs(target - df[bcol]) * 100)
                elif col == "cov_95":
                    # Coverage metric targeting 0.95; use distance-to-target formula
                    target = 0.95
                    df[col] = ((np.abs(target - df[bcol]) - np.abs(target - df[col])) / 
                               np.abs(target - df[bcol]) * 100)
                else:
                    # Percent improvement vs baseline; positive = better (lower error is better)
                    df[col] = (df[bcol] - df[col]) / df[bcol].abs() * 100
            df = df.drop(columns=[c + "_baseline" for c in numeric_cols])

        # Aggregate over all datasets and folds (per model)
        avg_all = df.groupby("model").agg(agg_fn, numeric_only=True)

        # Aggregate over folds, per dataset (per model, per dataset)
        avg_per_dataset = df.groupby(["dataset", "model"]).agg(agg_fn, numeric_only=True).reset_index()

        # Ensure output dir exists
        csv_mode = f"relative_{relative_baseline}" if relative_baseline else "absolute"
        figures_dir = os.path.join(root, "figures")
        mode_dir = os.path.join(figures_dir, csv_mode)
        os.makedirs(mode_dir, exist_ok=True)

        # --- Table as image (overall average) ---

        # Prepare table data: keep model names as strings, round only numeric columns to 2 digits

        avg_all_reset = avg_all.reset_index()
        table_data = avg_all_reset.copy()
        col_labels = list(avg_all_reset.columns)
        if relative_baseline:
            # Add units to column headers for metrics
            for i, col in enumerate(col_labels):
                if col == "r2":
                    col_labels[i] = f"{col} (pp)"
                elif col != "model" and np.issubdtype(avg_all_reset[col].dtype, np.number):
                    col_labels[i] = f"{col} (%)"
        for col in avg_all_reset.columns:
            if np.issubdtype(avg_all_reset[col].dtype, np.number):
                table_data[col] = np.round(avg_all_reset[col], 2)
        # Convert all values to string for better table formatting
        cell_text = table_data.astype(str).values.tolist()

        # Find best model per metric (min for most, max for r2/dispersion or relative mode, closest-to-target for coverage)
        metrics = [col for col in avg_all.columns if np.issubdtype(avg_all[col].dtype, np.number)]
        best_model_per_metric = {}
        for metric in metrics:
            if metric == "r2" or metric == "dispersion" or relative_baseline:
                # R², dispersion (higher is better), and relative mode: maximize
                best_model = avg_all[metric].idxmax()
            elif metric == "cov_90":
                # Coverage targeting 0.90: minimize distance to target
                best_model = (avg_all[metric] - 0.90).abs().idxmin()
            elif metric == "cov_95":
                # Coverage targeting 0.95: minimize distance to target
                best_model = (avg_all[metric] - 0.95).abs().idxmin()
            else:
                # Error metrics: minimize
                best_model = avg_all[metric].idxmin()
            best_model_per_metric[metric] = best_model


        fig, ax = plt.subplots(figsize=(2 + 2 * len(avg_all_reset.columns), 1.5 + 0.5 * len(avg_all)))
        ax.axis('off')
        table = ax.table(cellText=cell_text,
                         colLabels=col_labels,
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)

        # Bold the winning model per metric
        model_names = avg_all_reset['model'].tolist()
        for j, metric in enumerate(avg_all_reset.columns[1:]):  # skip model col
            if metric in best_model_per_metric:
                best_model = best_model_per_metric[metric]
                if best_model in model_names:
                    i = model_names.index(best_model)
                    cell = table[i+1, j+1]  # +1 for header row/col
                    cell.get_text().set_weight('bold')

        title_suffix = f" (relative to {relative_baseline})" if relative_baseline else ""
        plt.title(f"{agg_label} over all datasets and folds (per model){title_suffix}")
        plt.tight_layout()
        plt.savefig(os.path.join(mode_dir, f"summary_table_{csv_mode}.png"), dpi=200)
        plt.close()

        # --- CSV: aggregated summary ---
        csv_summary = avg_all_reset.copy()
        csv_summary.columns = col_labels
        csv_summary.to_csv(os.path.join(mode_dir, f"summary_table_{csv_mode}.csv"), index=False)

        # --- Table as image (per dataset) ---


        # Prepare table data: keep dataset/model names as strings, round only numeric columns
        table_data_ds = avg_per_dataset.copy()
        col_labels_ds = list(avg_per_dataset.columns)
        if relative_baseline:
            for i, col in enumerate(col_labels_ds):
                if col == "r2":
                    col_labels_ds[i] = f"{col} (pp)"
                elif col not in ("dataset", "model") and np.issubdtype(avg_per_dataset[col].dtype, np.number):
                    col_labels_ds[i] = f"{col} (%)"
        for col in avg_per_dataset.columns:
            if np.issubdtype(avg_per_dataset[col].dtype, np.number):
                table_data_ds[col] = np.round(avg_per_dataset[col], 4)
        cell_text_ds = table_data_ds.values.tolist()

        fig, ax = plt.subplots(figsize=(16, 2 + 0.3 * len(avg_per_dataset)))
        ax.axis('off')
        table = ax.table(cellText=cell_text_ds,
                         colLabels=col_labels_ds,
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        plt.title(f"{agg_label} over folds, per dataset (per model, per dataset){title_suffix}")
        plt.tight_layout()
        plt.savefig(os.path.join(mode_dir, f"per_dataset_table_{csv_mode}.png"), dpi=200)
        plt.close()

        # --- CSV: per-dataset table ---
        csv_per_dataset = table_data_ds.copy()
        csv_per_dataset.columns = col_labels_ds
        csv_per_dataset.to_csv(os.path.join(mode_dir, f"per_dataset_table_{csv_mode}.csv"), index=False)

        # --- Bar plots for key metrics ---
        metrics = ["mae", "rmse", "crps", "crls", "wcrps_left", "wcrps_right", "wcrps_center", "sharpness", "dispersion", "r2", "cov_90", "cov_95", "log_score"]
        for metric in metrics:
            if metric not in avg_all.columns:
                continue
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(avg_all.index, avg_all[metric].values)
            if relative_baseline:
                ylabel = f"{metric} (% improvement vs {relative_baseline})" if metric != "r2" else f"{metric} (pp change vs {relative_baseline})"
                ax.set_title(f"{metric} — % improvement vs {relative_baseline} ({agg_label.lower()}, higher is better)")
                ax.set_ylabel(ylabel)
            else:
                # For log_score and crls, use linear scale (error metrics); others use symlog
                # Trick: log_score and crls are error metrics where lower is better, avoid symlog
                if metric not in ("log_score", "crls"):
                    ax.set_yscale('symlog', linthresh=1e-3)
                # Determine direction for this metric
                if metric == "r2" or metric == "dispersion":
                    direction = "higher is better"
                elif metric in ("cov_90", "cov_95"):
                    direction = "closer to target is better"
                else:
                    direction = "lower is better"
                ax.set_title(f"{metric} ({agg_label.lower()}, {direction})")
                ax.set_ylabel(metric)
            ax.set_xlabel("Model")
            # Rotate x-axis labels by 90 degrees to prevent overlap
            plt.setp(ax.get_xticklabels(), rotation=90)
            # Use tight_layout with extra padding to ensure y-axis is readable
            plt.tight_layout(pad=2.0, rect=[0.08, 0, 1, 1])
            plt.savefig(os.path.join(mode_dir, f"bar_{metric}.png"), dpi=200)
            plt.close()

        # --- Per-dataset bar plots for each metric ---
        for metric in metrics:
            if metric not in avg_per_dataset.columns:
                continue
            fig, ax = plt.subplots(figsize=(16, 6))
            datasets_u = avg_per_dataset['dataset'].unique()
            models_u = avg_per_dataset['model'].unique()
            width = 0.8 / len(models_u)
            x = np.arange(len(datasets_u))
            for i, model in enumerate(models_u):
                vals = avg_per_dataset[avg_per_dataset['model'] == model][metric].values
                ax.bar(x + i * width, vals, width=width, label=model)
            if relative_baseline:
                ylabel = f"{metric} (% improvement vs {relative_baseline})" if metric != "r2" else f"{metric} (pp change vs {relative_baseline})"
                ax.set_title(f"{metric} by dataset — % improvement vs {relative_baseline} ({agg_label.lower()}, higher is better)")
                ax.set_ylabel(ylabel)
            else:
                # For log_score and crls, use linear scale (error metrics); others use symlog
                # Trick: log_score and crls are error metrics where lower is better, avoid symlog
                if metric not in ("log_score", "crls"):
                    ax.set_yscale('symlog', linthresh=1e-3)
                # Determine direction for this metric
                if metric == "r2" or metric == "dispersion":
                    direction = "higher is better"
                elif metric in ("cov_90", "cov_95"):
                    direction = "closer to target is better"
                else:
                    direction = "lower is better"
                ax.set_title(f"{metric} by dataset ({agg_label.lower()}, {direction})")
                ax.set_ylabel(metric)
            ax.set_xlabel("Dataset")
            ax.set_xticks(x + width * (len(models_u) - 1) / 2)
            ax.set_xticklabels(datasets_u, rotation=90)
            ax.legend(title="Model")
            # Use tight_layout with extra padding to ensure y-axis is readable
            plt.tight_layout(pad=2.0, rect=[0.08, 0, 1, 1])
            plt.savefig(os.path.join(mode_dir, f"bar_{metric}_per_dataset.png"), dpi=200)
            plt.close()

        print(f"Plots saved to {mode_dir}/.")

        # --- Spider/Radar plots per metric (PDF exports) ---
        metrics_all = [col for col in avg_all.columns if np.issubdtype(avg_all[col].dtype, np.number)]
        
        # Define formulas for metrics (relative mode shown, absolute will be different)
        relative_formulas = {
            # Relative mode: computed per fold, then averaged
            "mae": r"$\mathbb{E}\left[\frac{B - M}{|B|}\right] \times 100\%$",
            "rmse": r"$\mathbb{E}\left[\frac{B - M}{|B|}\right] \times 100\%$",
            "crps": r"$\mathbb{E}\left[\frac{B - M}{|B|}\right] \times 100\%$",
            "crls": r"$\mathbb{E}\left[\frac{B - M}{|B|}\right] \times 100\%$",
            "sharpness": r"$\mathbb{E}\left[\frac{B - M}{|B|}\right] \times 100\%$",
            "dispersion": r"$\mathbb{E}\left[\frac{M - B}{|B|}\right] \times 100\%$",
            "r2": r"$\mathbb{E}[(R^2_M - R^2_B)] \times 100~\text{pp}$",
            "cov_90": r"$\mathbb{E}\left[\frac{|0.90 - B| - |0.90 - M|}{|0.90 - B|}\right] \times 100\%$",
            "cov_95": r"$\mathbb{E}\left[\frac{|0.95 - B| - |0.95 - M|}{|0.95 - B|}\right] \times 100\%$",
            "wcrps_left": r"$\mathbb{E}\left[\frac{B - M}{|B|}\right] \times 100\%$",
            "wcrps_right": r"$\mathbb{E}\left[\frac{B - M}{|B|}\right] \times 100\%$",
            "wcrps_center": r"$\mathbb{E}\left[\frac{B - M}{|B|}\right] \times 100\%$",
            "log_score": r"$\mathbb{E}\left[\frac{B - M}{|B|}\right] \times 100\%$",
        }
        
        absolute_formulas = {
            # Absolute mode: aggregated values (mean or median across folds/datasets)
            "mae": r"$\mathbb{E}_{i}[\text{MAE}_i]$",
            "rmse": r"$\mathbb{E}_{i}[\text{RMSE}_i]$",
            "crps": r"$\mathbb{E}_{i}[\text{CRPS}_i]$",
            "crls": r"$\mathbb{E}_{i}[\text{CRLS}_i]$",
            "sharpness": r"$\mathbb{E}_{i}[\text{Sharp}_i]$",
            "dispersion": r"$\mathbb{E}_{i}[\text{Disp}_i]$",
            "r2": r"$\mathbb{E}_{i}[R^2_i]$",
            "cov_90": r"$\mathbb{E}_{i}[\text{Cov}_{90,i}]$",
            "cov_95": r"$\mathbb{E}_{i}[\text{Cov}_{95,i}]$",
            "wcrps_left": r"$\mathbb{E}_{i}[\text{wCRPS}_{\text{L},i}]$",
            "wcrps_right": r"$\mathbb{E}_{i}[\text{wCRPS}_{\text{R},i}]$",
            "wcrps_center": r"$\mathbb{E}_{i}[\text{wCRPS}_{\text{C},i}]$",
            "log_score": r"$\mathbb{E}_{i}[\text{LogS}_i]$",
        }
        
        for metric in metrics_all:
            fig, ax = plt.subplots(figsize=(10, 9), subplot_kw=dict(projection='polar'))

            models = list(avg_all.index)
            values = avg_all[metric].values.tolist()

            # In relative mode, exclude the baseline model from the plot
            if relative_baseline:
                baseline_idx = models.index(relative_baseline) if relative_baseline in models else -1
                if baseline_idx >= 0:
                    models = models[:baseline_idx] + models[baseline_idx+1:]
                    values = values[:baseline_idx] + values[baseline_idx+1:]

            # Normalize values for visualization
            if relative_baseline:
                val_min = min(values)
                val_max = max(values)
                val_range = val_max - val_min if val_max > val_min else 1
                normalized_values = [0.1 + 0.9 * (v - val_min) / val_range for v in values]
            else:
                normalized_values = values[:]

            # Set up the radar
            angles = np.linspace(0, 2 * np.pi, len(models), endpoint=False).tolist()
            normalized_values += normalized_values[:1]  # Complete the circle
            angles += angles[:1]

            # --- Overlay gradient (red to green) ---
            # Determine if higher or lower values are better for this metric
            # Trick: log_score and crls are error metrics where lower is better
            higher_is_better = metric == "r2" or metric == "dispersion"  # R² and dispersion: higher is better
            
            from matplotlib.colors import LinearSegmentedColormap
            n_rings = 200
            if relative_baseline:
                # For normalized scale, 0.1 (worst) to 1.0 (best)
                radii = np.linspace(0.1, 1.0, n_rings)
                # In relative mode, higher is always better (it's percent improvement)
                rg_cmap = LinearSegmentedColormap.from_list('rg', ['red', 'yellow', 'green'])
                cmap_values = np.linspace(0, 1, n_rings)
            else:
                # For absolute mode, use the actual radial range
                val_min = min(normalized_values)
                val_max = max(normalized_values)
                radii = np.linspace(val_min, val_max, n_rings)
                
                # Choose colormap direction based on metric
                if higher_is_better:
                    # R²: higher is better, so red at bottom (low), green at top (high)
                    rg_cmap = LinearSegmentedColormap.from_list('rg', ['red', 'yellow', 'green'])
                    cmap_values = np.linspace(0, 1, n_rings)
                else:
                    # Error metrics: lower is better, so red at top (high), green at bottom (low)
                    rg_cmap = LinearSegmentedColormap.from_list('rg', ['green', 'yellow', 'red'])
                    cmap_values = np.linspace(0, 1, n_rings)
            
            theta = np.linspace(0, 2 * np.pi, 500)
            for r_idx, r in enumerate(radii):
                color = rg_cmap(cmap_values[r_idx])
                ax.fill_between(theta, r, r + (radii[1] - radii[0]), color=color, alpha=0.18, zorder=0)

            # Plot the spider net
            ax.plot(angles, normalized_values, 'o-', linewidth=2, label=metric, color='black', zorder=2)
            ax.fill(angles, normalized_values, alpha=0.25, color='black', zorder=2)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(models, size=10)

            # Set radial limits and ticks based on mode
            if relative_baseline:
                ax.set_ylim(0, 1.0)
                ax.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9])
                ax.set_yticklabels(['Worst', '', 'Mid', '', 'Best'], fontsize=8)
            else:
                val_min = min(values)
                val_max = max(values)
                val_range = val_max - val_min if val_max > val_min else 1
                ax.set_ylim(max(0, val_min - 0.1 * val_range), val_max + 0.1 * val_range)
                ax.set_yticks(np.linspace(max(0, val_min - 0.1 * val_range), val_max + 0.1 * val_range, 5))
                # Update axis labels based on metric direction
                if higher_is_better:
                    ax.set_yticklabels(['Worse', '', 'Mid', '', 'Better'], fontsize=8)
                else:
                    ax.set_yticklabels(['Better', '', 'Mid', '', 'Worse'], fontsize=8)

            ax.grid(True)

            # Determine formula based on mode
            if relative_baseline:
                formula = relative_formulas.get(metric, "")
                title_text = f"{metric} (relative to {relative_baseline})\nNormalized scale"
            else:
                formula = absolute_formulas.get(metric, "")
                title_text = f"{metric} (absolute)\nRaw aggregated values"

            ax.set_title(title_text, size=12, pad=20)

            # Add formula as text below the plot
            formula_text = f"Formula: {formula}" if formula else ""
            if relative_baseline:
                mode_note = "Expectation computed per fold before aggregation.\nRadial scale: Min=worst model, Max=best model"
            else:
                if higher_is_better:
                    mode_note = "Aggregated metric (mean/median across all folds and datasets).\nColor scale: Red=low(worse), Green=high(better)"
                else:
                    mode_note = "Aggregated metric (mean/median across all folds and datasets).\nColor scale: Green=low(better), Red=high(worse)"

            fig.text(0.5, 0.02, formula_text + "\n" + mode_note, 
                    ha='center', fontsize=9, style='italic', wrap=True)

            plt.tight_layout(rect=[0, 0.06, 1, 1])
            pdf_filename = os.path.join(mode_dir, f"spider_{metric}_{csv_mode}.pdf")
            plt.savefig(pdf_filename, format='pdf', dpi=150, bbox_inches='tight')
            plt.close()

# Insights
print("\nInsights:")
for metric in metrics:
    if metric not in avg_all.columns:
        continue
    best = avg_all[metric].idxmax() if (metric == "r2" or relative_baseline) else avg_all[metric].idxmin()
    print(f"Best model for {metric}: {best}")