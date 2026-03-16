#!/usr/bin/env python3
"""
Permutation P-Value Leaderboard

Reads per-fold results (JSON files), aggregates folds per dataset to ensure
statistical independence (avoiding pseudoreplication), and computes empirical 
p-values by shuffling scores within each dataset row.

Outputs CSV tables with columns: model, rank, p_value, observed_mean, average_rank
"""
import argparse
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import importlib.util
import sys


def _load_write_latex_writer():
    """Dynamically load scoringbench/latex_tables.py without importing the
    package (avoids running scoringbench.__init__ and its side-effects).
    Returns the `write_latex_tables` callable or None on failure.
    """
    try:
        base = os.path.dirname(__file__)
        path = os.path.join(base, 'scoringbench', 'latex_tables.py')
        spec = importlib.util.spec_from_file_location('scoringbench_latex_tables', path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, 'write_latex_tables', None)
    except Exception:
        return None


def _collect_all_rows(root):
    rows = []
    if not os.path.exists(root):
        return rows
    for entry in os.listdir(root):
        entry_path = os.path.join(root, entry)
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
                if 'fold' in row and isinstance(row['fold'], (int, float)):
                    row['fold'] = f"fold_{int(row['fold'])}"
                rows.append(row)
            # ignore legacy directory layout; leaderboard uses parquet files only
            continue
        # skip non-parquet entries
        continue
    return rows



def _write_latex_tables(root, rows):
    writer = _load_write_latex_writer()
    if writer is None:
        return
    try:
        writer(root, rows)
    except Exception:
        pass


def _write_leaderboard_tex(figures_dir, metric, df, higher_is_better):
    # Directly append a LaTeX leaderboard table into absolute_metrics.tex
    try:
        os.makedirs(figures_dir, exist_ok=True)
        abs_tex = os.path.join(figures_dir, 'absolute_metrics.tex')
        lines = []
        lines.append('% --- Leaderboard for metric: ' + metric + ' ---')
        lines.append('\\begin{table}[htbp]')
        lines.append('\\centering')
        lines.append('\\caption{Leaderboard for ' + metric + '}')
        lines.append('\\label{tab:leaderboard_' + metric + '}')
        lines.append('\\small')
        lines.append('\\begin{tabular}{r l r r r}')
        lines.append('\\toprule')
        lines.append('Rank & Model & p-value & Observed & AverageRank \\\\')
        lines.append('\\midrule')

        for i, row in df.iterrows():
            rank = int(row.get('leader_rank', i + 1))
            model = str(row.get('model', ''))
            p = row.get('p_value', np.nan)
            obs = row.get('observed_mean', np.nan)
            norm = row.get('average_rank', np.nan)
            p_s = f"{p:.3f}" if np.isfinite(p) else ''
            obs_s = f"{obs:.3f}" if np.isfinite(obs) else ''
            norm_s = f"{norm:.3f}" if np.isfinite(norm) else ''
            if i == 0:
                model_s = '\\textbf{' + model.replace('_', '\\_') + '}'
                norm_s = '$\\mathbf{' + norm_s + '}$' if norm_s else norm_s
            else:
                model_s = model.replace('_', '\\_')
                norm_s = '$' + norm_s + '$' if norm_s else norm_s
            p_s = '$' + p_s + '$' if p_s else p_s
            obs_s = '$' + obs_s + '$' if obs_s else obs_s
            lines.append(f"{rank} & {model_s} & {p_s} & {obs_s} & {norm_s} \\\\")

        lines.append('\\bottomrule')
        lines.append('\\end{tabular}')
        lines.append('\\end{table}')
        lines.append('')

        # Append to absolute_metrics.tex
        with open(abs_tex, 'a') as af:
            af.write('\n'.join(lines))
    except Exception:
        pass


def discover_output_roots(output_arg):
    if output_arg:
        return [output_arg]
    return ["output"]


def load_metric_matrix(root, metric):
    # Aggregates folds per dataset before pivoting
    all_rows = []
    for entry in os.listdir(root):
        entry_path = os.path.join(root, entry)
        if not os.path.isfile(entry_path) or not entry.endswith('.parquet'):
            continue
        model = entry[:-8]
        try:
            df = pd.read_parquet(entry_path)
        except Exception:
            continue
        if df.empty or metric not in df.columns:
            continue
        grp = df.groupby('dataset')[metric].mean().dropna()
        for dataset, dataset_avg in grp.items():
            all_rows.append({"dataset": dataset, "model": model, "val": float(dataset_avg)})

    df = pd.DataFrame(all_rows)
    if df.empty:
        return None
    
    # Pivot: rows=datasets, cols=models
    pivot = df.pivot_table(index="dataset", columns="model", values="val")
    return pivot.dropna(how='any')


def compute_permutation_pvalues(mat, nsim=20000, seed=None, higher_is_better=False):
    rng = np.random.RandomState(seed)
    arr_raw = mat.values  # shape (n_datasets, n_models)
    n_datasets, n_models = arr_raw.shape

    # Always work in rank space: transform to ranks (0 is smallest, n_models-1 is largest)
    arr_working = arr_raw.argsort(axis=1).argsort(axis=1)

    # In rank-space, range is only 0 if there's only 1 model
    good_mask = np.ones(n_datasets, dtype=bool) if n_models > 1 else np.zeros(n_datasets, dtype=bool)

    if not np.all(good_mask):
        dropped = np.count_nonzero(~good_mask)
        print(f"Warning: dropping {dropped} datasets with zero range (identical scores)")

    # Use raw ranks (no min-max scaling) so the returned statistic is average rank
    arr_final = arr_working[good_mask]
    arr_norm = arr_final.astype(float)

    n_rows, _ = arr_norm.shape

    # Test statistics
    observed_norm = arr_norm.mean(axis=0)  # average rank
    observed_raw = arr_raw[good_mask].mean(axis=0)

    # Permutation loop
    sims = np.empty((nsim, n_models), dtype=float)
    for s in tqdm(range(nsim), desc="Permutations"):
        perm = np.argsort(rng.rand(n_rows, n_models), axis=1)
        permuted = arr_norm[np.arange(n_rows)[:, None], perm]
        sims[s] = permuted.mean(axis=0)

    # Empirical P-values
    if higher_is_better:
        counts = np.sum(sims >= observed_norm[None, :], axis=0)
    else:
        counts = np.sum(sims <= observed_norm[None, :], axis=0)

    pvals = (counts + 1) / (nsim + 1)
    return observed_raw, observed_norm, pvals


def discover_metrics(root, limit=50):
    """Return sorted list of numeric metric names discovered in per-model parquet files."""
    discovered_metrics = set()
    for model_entry in os.listdir(root):
        model_path = os.path.join(root, model_entry)
        if os.path.isfile(model_path) and model_entry.endswith('.parquet'):
            try:
                df = pd.read_parquet(model_path)
                if isinstance(df, pd.DataFrame):
                    for k in df.select_dtypes(include=['number']).columns:
                        discovered_metrics.add(k)
            except Exception:
                continue
        if len(discovered_metrics) > limit:
            break
    return sorted(discovered_metrics)


def _frames_equal(a, b, rtol=1e-8, atol=1e-12):
    """Compare two DataFrame subsets: exact model order and numerics with tolerance."""
    if a.shape != b.shape:
        return False
    if a['model'].tolist() != b['model'].tolist():
        return False
    for col in ['p_value', 'observed_mean', 'average_rank']:
        try:
            aval = a[col].astype(float).to_numpy()
            bval = b[col].astype(float).to_numpy()
        except Exception:
            return False
        if not np.allclose(aval, bval, equal_nan=True, atol=atol, rtol=rtol):
            return False
    return True


def write_pvalues_csv(results, out_csv, cols):
    """Write pvalues CSV unless the filename is blacklisted. Returns True if written."""
    if os.path.basename(out_csv) == 'pvalues_fold.csv':
        print(f"Skipping blacklisted output file: {out_csv}")
        return False
    results[cols].to_csv(out_csv, index=False)
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    parser.add_argument("--nsim", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    # `--metric` and `--higher_is_better` removed; metrics are auto-discovered
    parser.add_argument("--output_csv", default=None)

    args = parser.parse_args()
    roots = discover_output_roots(args.output)
    for root in roots:
        # Collect all rows (from parquet or legacy JSON) and write LaTeX tables
        try:
            rows_all = _collect_all_rows(root)
            _write_latex_tables(root, rows_all)
        except Exception:
            pass
        # Auto-discover all numeric metrics from per-model parquet files
        discovered_metrics = set()
        for model_entry in os.listdir(root):
            model_path = os.path.join(root, model_entry)
            # Only inspect per-model parquet files for leaderboard metrics
            if os.path.isfile(model_path) and model_entry.endswith('.parquet'):
                try:
                    df = pd.read_parquet(model_path)
                    if isinstance(df, pd.DataFrame):
                        for k in df.select_dtypes(include=['number']).columns:
                            discovered_metrics.add(k)
                except Exception:
                    continue
            if len(discovered_metrics) > 50:
                break
        metrics_to_run = sorted(discovered_metrics)

        for metric in metrics_to_run:
            try:
                pivot = load_metric_matrix(root, metric)
                if pivot is None or pivot.empty: continue

                hib = metric in ("r2", "dispersion")

                # Special handling for coverage: evaluate distance to nominal target
                is_coverage = metric.startswith("coverage_")
                if is_coverage:
                    try:
                        lvl = int(metric.split("_")[1])
                        target = lvl / 100.0
                    except Exception:
                        target = 0.0
                    # matrix used for permutation tests: absolute distance to nominal
                    mat_for_test = (pivot - target).abs()
                    # preserve original (coverage) raw means for display
                    raw_means = pivot.values.mean(axis=0)
                    # tie-break uses the mean distance
                    tie_means = mat_for_test.values.mean(axis=0)
                else:
                    mat_for_test = pivot
                    raw_means = pivot.values.mean(axis=0)
                    tie_means = raw_means

                # Run permutation test on the (possibly transformed) matrix
                _raw_test, norm_m, pvals = compute_permutation_pvalues(
                    mat_for_test, nsim=args.nsim, seed=args.seed, higher_is_better=hib
                )

                # observed_mean should show the original metric (coverage or raw performance)
                # For coverage metrics, report the mean absolute distance to nominal
                observed_mean = tie_means if is_coverage else raw_means

                # average_rank: transform so smaller is better for display
                avg_rank = norm_m
                try:
                    n_models = pivot.shape[1]
                except Exception:
                    n_models = int(len(avg_rank))
                if hib:
                    # For metrics where higher is better, invert ranks so lower=better
                    avg_rank = (n_models - 1) - avg_rank

                results = pd.DataFrame({
                    "model": pivot.columns,
                    "p_value": pvals,
                    "observed_mean": observed_mean,
                    "average_rank": avg_rank
                })

                # Sort by p_value, then tie-break appropriately.
                if is_coverage:
                    # For coverage we want models closest to the nominal target (smaller distance)
                    results["_tie"] = tie_means
                    results = results.sort_values(["p_value", "_tie"], ascending=[True, True]).reset_index(drop=True)
                    results = results.drop(columns=["_tie"])
                else:
                    results = results.sort_values(
                        ["p_value", "observed_mean"], 
                        ascending=[True, not hib]
                    ).reset_index(drop=True)
                results["rank"] = results.index + 1
                cols = ["model", "rank", "p_value", "observed_mean", "average_rank"]

                figures_dir = os.path.join(root, "figures", "permutation")
                os.makedirs(figures_dir, exist_ok=True)
                out_csv = args.output_csv or os.path.join(figures_dir, f"pvalues_{metric}.csv")
                write_pvalues_csv(results, out_csv, cols)

                # Pass a leaderboard DataFrame derived directly from `results`
                _write_leaderboard_tex(
                    os.path.join(root, 'figures'),
                    metric,
                              # average_rank is already oriented so smaller==better; always sort ascending
                              results.sort_values(["average_rank"], ascending=[True])
                           .reset_index(drop=True)
                           .assign(leader_rank=lambda df: df.index + 1),
                    hib,
                )

                print(f"\n--- {metric} ---")
                print(results[cols].to_string(index=False))
            except Exception as e:
                print(f"Skipping {metric}: {e}")

if __name__ == "__main__":
    main()