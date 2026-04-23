import os
from typing import List

import numpy as np
import pandas as pd


def write_latex_tables(root: str, rows: List[dict]) -> None:
    """Write three LaTeX tables (point, probabilistic, calibration) to
    <root>/figures/absolute_metrics.tex using numeric metrics found in
    the provided rows list.
    """
    if not rows:
        return
    df = pd.DataFrame(rows)
    if df.empty:
        return

    avg = df.groupby('model').mean(numeric_only=True)
    numeric_cols = [c for c in avg.columns if np.issubdtype(avg[c].dtype, np.number)]

    # Desired metrics with candidate names
    table1_wanted = [
        ('mae', ['mae']),
        ('rmse', ['rmse']),
        ('r2', ['r2', 'r_squared']),
        ('train_time', ['train_time', 'time', 'train_time_seconds']),
    ]

    table2_wanted = [
        ('crps', ['crps']),
        ('log_score', ['log_score', 'logscore', 'log_s']),
        ('crls', ['crls']),
        ('cde_loss', ['cde_loss']),
        # energy scores for multiple beta values
        ('energy_score_beta_0.2', ['energy_score_beta_0.2', 'energy_score_0.2']),
        ('energy_score_beta_0.5', ['energy_score_beta_0.5', 'energy_score_0.5']),
        ('energy_score_beta_1.0', ['energy_score_beta_1.0', 'energy_score_1.0', 'es_1.0', 'es_1']),
        ('energy_score_beta_1.5', ['energy_score_beta_1.5', 'energy_score_1.5']),
        ('energy_score_beta_2.0', ['energy_score_beta_2.0', 'energy_score_2.0']),
    ]

    table3_wanted = [
        ('sharpness', ['sharpness']),
        ('dispersion', ['dispersion']),
        ('coverage_90', ['cov_90', 'coverage_90', 'coverage_0.90', 'coverage_90.0']),
        ('interval_score_90', ['interval_score_90', 'is_90', 'interval_score_0.90']),
        ('coverage_95', ['cov_95', 'coverage_95', 'coverage_0.95']),
        ('interval_score_95', ['interval_score_95', 'is_95', 'interval_score_0.95']),
    ]

    def _find_col(cands):
        for cand in cands:
            if cand in numeric_cols:
                return cand
        return None

    table1 = []
    for display, cands in table1_wanted:
        found = _find_col(cands)
        if found:
            table1.append((found, '{:.3f}' if found != 'r2' else '{:.4f}', display.upper() if display != 'r2' else '$R^2$'))

    table2 = []
    for display, cands in table2_wanted:
        found = _find_col(cands)
        if found:
            fmt = '{:.3f}' if 'log' not in found else '{:.4f}'
            table2.append((found, fmt, display.upper()))
    for w in ['wcrps_left', 'wcrps_right', 'wcrps_center']:
        if w in numeric_cols:
            table2.append((w, '{:.3f}', w.upper()))

    table3 = []
    for display, cands in table3_wanted:
        found = _find_col(cands)
        if found:
            table3.append((found, '{:.3f}', display.upper()))

    def _esc(text):
        return str(text).replace('\\', '\\textbackslash{}').replace('_', '\\_').replace('%', '\\%')

    def _format_val(metric, val, is_best, fmt):
        try:
            s = fmt.format(val)
        except Exception:
            s = str(val)
        if is_best:
            return '$\\mathbf{' + s + '}$'
        return '$' + s + '$'

    def _best_mask(series, metric):
        if series.dropna().empty:
            return pd.Series([False] * len(series), index=series.index)
        if metric in ('r2', 'dispersion'):
            return series == series.max()
        if metric in ('cov_90', 'coverage_90'):
            return (series - 0.90).abs() == (series - 0.90).abs().min()
        if metric in ('cov_95', 'coverage_95'):
            return (series - 0.95).abs() == (series - 0.95).abs().min()
        return series == series.min()

    models = list(avg.index)

    lines = []

    # Table 1
    lines.append('\\begin{landscape}')
    lines.append('\\begin{table}[htbp]')
    lines.append('\\centering')
    lines.append('\\caption{Point estimation and predictive accuracy (means across folds and datasets).}')
    lines.append('\\label{tab:point_metrics}')
    lines.append('\\small')
    lines.append('\\begin{tabular}{l' + 'r'*len(table1) + '}')
    lines.append('\\toprule')
    hdrs = ['\\textbf{Model}'] + ['\\textbf{' + _esc(hdr) + '}' for (_, _, hdr) in table1]
    lines.append(' & '.join(hdrs) + ' \\\\')
    lines.append('\\midrule')

    bests_t1 = {}
    for m, _, _ in table1:
        bests_t1[m] = _best_mask(avg[m], m) if m in avg.columns else pd.Series([False] * len(avg), index=avg.index)

    for model in models:
        row = [_esc(model)]
        for m, fmt, _ in table1:
            if m in avg.columns and pd.notna(avg.loc[model, m]):
                val = avg.loc[model, m]
                is_best = bool(bests_t1[m].get(model, False))
                row.append(_format_val(m, val, is_best, fmt))
            else:
                row.append('')
        lines.append(' & '.join(row) + ' \\\\')

    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')
    lines.append('\\end{landscape}')
    lines.append('')

    # Table 2
    lines.append('\\begin{landscape}')
    lines.append('\\begin{table}[htbp]')
    lines.append('\\centering')
    lines.append('\\caption{Probabilistic performance (proper scoring rules).}')
    lines.append('\\label{tab:probabilistic_metrics}')
    lines.append('\\small')
    lines.append('\\resizebox{\\linewidth}{!}{%')
    lines.append('\\begin{tabular}{l' + 'r'*len(table2) + '}')
    lines.append('\\toprule')
    hdrs2 = ['\\textbf{Model}'] + ['\\textbf{' + _esc(hdr) + '}' for (_, _, hdr) in table2]
    lines.append(' & '.join(hdrs2) + ' \\\\')
    lines.append('\\midrule')

    bests_t2 = {}
    for m, _, _ in table2:
        bests_t2[m] = _best_mask(avg[m], m) if m in avg.columns else pd.Series([False] * len(avg), index=avg.index)

    for model in models:
        row = [_esc(model)]
        for m, fmt, _ in table2:
            if m in avg.columns and pd.notna(avg.loc[model, m]):
                val = avg.loc[model, m]
                is_best = bool(bests_t2[m].get(model, False))
                row.append(_format_val(m, val, is_best, fmt))
            else:
                row.append('')
        lines.append(' & '.join(row) + ' \\\\')

    lines.append('\\bottomrule')
    lines.append('\\end{tabular}%')
    lines.append('}')
    lines.append('\\end{table}')
    lines.append('\\end{landscape}')
    lines.append('')

    # Table 3 (no bolding)
    lines.append('\\begin{landscape}')
    lines.append('\\begin{table}[htbp]')
    lines.append('\\centering')
    lines.append('\\caption{Calibration and uncertainty diagnostics.}')
    lines.append('\\label{tab:calibration_metrics}')
    lines.append('\\small')
    lines.append('\\begin{tabular}{l' + 'r'*len(table3) + '}')
    lines.append('\\toprule')
    hdrs3 = ['\\textbf{Model}'] + ['\\textbf{' + _esc(hdr) + '}' for (_, _, hdr) in table3]
    lines.append(' & '.join(hdrs3) + ' \\\\')
    lines.append('\\midrule')

    for model in models:
        row = [_esc(model)]
        for m, fmt, _ in table3:
            if m in avg.columns and pd.notna(avg.loc[model, m]):
                val = avg.loc[model, m]
                row.append('$' + fmt.format(val) + '$')
            else:
                row.append('')
        lines.append(' & '.join(row) + ' \\\\')

    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')
    lines.append('\\end{landscape}')

    figures_dir = os.path.join(root, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    out_tex = os.path.join(figures_dir, 'absolute_metrics.tex')
    try:
        with open(out_tex, 'w') as fh:
            fh.write('\n'.join(lines))
    except Exception:
        pass


def write_leaderboard_table(figures_dir: str, metric: str, df: pd.DataFrame, higher_is_better: bool) -> None:
    """Write a LaTeX leaderboard table for a single metric to
    <figures_dir>/leaderboard_<metric>.tex. Expects `df` to contain
    columns: model, p_value, observed_mean, normalized_score, leader_rank.
    """
    try:
        os.makedirs(figures_dir, exist_ok=True)
        out = os.path.join(figures_dir, f"leaderboard_{metric}.tex")
        lines = []
        metric_esc = metric.replace('_', '\\_')
        lines.append('\\begin{table}[htbp]')
        lines.append('\\centering')
        lines.append('\\caption{Leaderboard for ' + metric_esc + '}')
        lines.append('\\label{tab:leaderboard_' + metric + '}')
        lines.append('\\small')
        lines.append('\\begin{tabular}{r l r r r}')
        lines.append('\\toprule')
        lines.append('Rank & Model & p-value & Observed & Normalized \\\\')
        lines.append('\\midrule')

        for i, row in df.iterrows():
            rank = int(row.get('leader_rank', i + 1))
            model = str(row.get('model', ''))
            p = row.get('p_value', np.nan)
            obs = row.get('observed_mean', np.nan)
            norm = row.get('normalized_score', np.nan)
            p_s = f"{p:.3f}" if np.isfinite(p) else ''
            obs_s = f"{obs:.3f}" if np.isfinite(obs) else ''
            norm_s = f"{norm:.3f}" if np.isfinite(norm) else ''
            if i == 0:
                model_s = '\\textbf{' + model.replace('\\', '\\textbackslash{}').replace('_', '\\_') + '}'
                norm_s = '$\\mathbf{' + norm_s + '}$' if norm_s else norm_s
            else:
                model_s = model.replace('\\', '\\textbackslash{}').replace('_', '\\_')
                norm_s = '$' + norm_s + '$' if norm_s else norm_s
            p_s = '$' + p_s + '$' if p_s else p_s
            obs_s = '$' + obs_s + '$' if obs_s else obs_s
            lines.append(f"{rank} & {model_s} & {p_s} & {obs_s} & {norm_s} \\\\")

        lines.append('\\bottomrule')
        lines.append('\\end{tabular}')
        lines.append('\\end{table}')

        with open(out, 'w') as fh:
            fh.write('\n'.join(lines))
    except Exception:
        pass
