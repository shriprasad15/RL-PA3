#!/usr/bin/env python3
"""Generate all required PA3 Pendulum plots and summary CSV tables from saved logs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pendulum_sac_lib import ensure_dir, load_all_eval_logs, safe_angle_label

AUTO_TARGETS = [0, -10, 30, -60, 90, -90, 120, -150]
SELECTED_TARGETS = [-60, 90, 120, -150]
METRICS = [
    'eval_return_mean',
    'mean_abs_angle_error_deg',
    'target_occupancy',
    'mean_abs_angular_velocity',
    'mean_squared_torque',
    'alpha',
]


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(',') if x.strip()]


def load_selected(path: Path) -> Dict[float, float]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    if 'selected_alphas' in raw:
        raw = raw['selected_alphas']
    return {float(k): float(v) for k, v in raw.items()}


def mean_ci_over_seeds(df: pd.DataFrame, group_cols: List[str], metric: str) -> pd.DataFrame:
    per_seed = df.groupby(group_cols + ['seed'], as_index=False)[metric].mean()
    g = per_seed.groupby(group_cols)[metric]
    out = g.agg(['mean', 'std', 'count']).reset_index()
    out['ci95'] = 1.96 * out['std'].fillna(0.0) / np.sqrt(out['count'].clip(lower=1))
    return out


def maybe_smooth(agg: pd.DataFrame, x: str, cols=('mean', 'ci95'), window: int = 1) -> pd.DataFrame:
    if window <= 1:
        return agg
    out = agg.sort_values(x).copy()
    for c in cols:
        out[c] = out[c].rolling(window=window, min_periods=1, center=False).mean()
    return out


def plot_ci(ax, agg: pd.DataFrame, x: str, label: str, smooth_window: int = 1):
    one = maybe_smooth(agg.sort_values(x), x, window=smooth_window)
    ax.plot(one[x], one['mean'], marker='o', markersize=3, linewidth=1.6, label=label)
    ax.fill_between(one[x].to_numpy(), (one['mean'] - one['ci95']).to_numpy(), (one['mean'] + one['ci95']).to_numpy(), alpha=0.18)


def plot_auto_all_targets(eval_df: pd.DataFrame, plots_dir: Path, metric: str, smooth_window: int = 1) -> Path | None:
    sub = eval_df[(eval_df['mode'] == 'auto') & np.isclose(eval_df['reward_scale'].astype(float), 1.0)].copy()
    if sub.empty or metric not in sub.columns:
        return None
    agg = mean_ci_over_seeds(sub, ['target_angle_deg', 'timestep'], metric)
    fig, ax = plt.subplots(figsize=(12, 7))
    for theta in AUTO_TARGETS:
        one = agg[np.isclose(agg['target_angle_deg'].astype(float), float(theta))]
        if not one.empty:
            plot_ci(ax, one, 'timestep', f'{theta:g}°', smooth_window)
    suffix = f'_smooth{smooth_window}' if smooth_window > 1 else ''
    ax.set_title(f'Auto-α SAC on modified Pendulum-v1: {metric}')
    ax.set_xlabel('Environment timesteps')
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)
    ax.legend(title='target', ncol=2)
    out = plots_dir / f'auto_all_targets_{metric}{suffix}.png'
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)
    return out


def plot_manual_vs_auto_panels(eval_df: pd.DataFrame, plots_dir: Path, selected: Dict[float, float], metric: str, smooth_window: int = 1) -> Path | None:
    if metric not in eval_df.columns:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axes = axes.flatten()
    used = False
    for ax, theta in zip(axes, SELECTED_TARGETS):
        auto = eval_df[(eval_df['mode'] == 'auto') & np.isclose(eval_df['target_angle_deg'].astype(float), theta) & np.isclose(eval_df['reward_scale'].astype(float), 1.0)]
        if not auto.empty:
            plot_ci(ax, mean_ci_over_seeds(auto, ['timestep'], metric), 'timestep', 'auto α', smooth_window)
            used = True
        alpha = selected.get(float(theta))
        if alpha is not None:
            man = eval_df[(eval_df['mode'] == 'manual') & np.isclose(eval_df['target_angle_deg'].astype(float), theta) & np.isclose(eval_df['manual_alpha'].astype(float), alpha) & np.isclose(eval_df['reward_scale'].astype(float), 1.0)]
            if not man.empty:
                plot_ci(ax, mean_ci_over_seeds(man, ['timestep'], metric), 'timestep', f'manual α={alpha:g}', smooth_window)
                used = True
        ax.set_title(f'target {theta:g}°')
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.legend()
    for ax in axes[-2:]:
        ax.set_xlabel('Environment timesteps')
    suffix = f'_smooth{smooth_window}' if smooth_window > 1 else ''
    fig.suptitle(f'Manual vs auto α on modified Pendulum-v1: {metric}', y=1.02)
    out = plots_dir / f'manual_vs_auto_panels_{metric}{suffix}.png'
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig)
    return out if used else None


def plot_reward_scaling_panels(eval_df: pd.DataFrame, plots_dir: Path, selected: Dict[float, float], metric: str, smooth_window: int = 1) -> Path | None:
    if metric not in eval_df.columns:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    used = False
    for ax, scale in zip(axes, [10.0, 0.1]):
        sub = eval_df[np.isclose(eval_df['target_angle_deg'].astype(float), 90.0) & np.isclose(eval_df['reward_scale'].astype(float), scale)]
        auto = sub[sub['mode'] == 'auto_scaled']
        man = sub[sub['mode'] == 'manual_scaled']
        if not auto.empty:
            plot_ci(ax, mean_ci_over_seeds(auto, ['timestep'], metric), 'timestep', 'auto α', smooth_window)
            used = True
        if not man.empty:
            alpha = selected.get(90.0, np.nan)
            plot_ci(ax, mean_ci_over_seeds(man, ['timestep'], metric), 'timestep', f'manual α={alpha:g}', smooth_window)
            used = True
        ax.set_title(f'reward scale {scale:g}×')
        ax.set_xlabel('Environment timesteps')
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[0].set_ylabel(metric)
    suffix = f'_smooth{smooth_window}' if smooth_window > 1 else ''
    fig.suptitle(f'Reward scaling at θ_target=90°: {metric}', y=1.02)
    out = plots_dir / f'reward_scaling_panels_{metric}{suffix}.png'
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig)
    return out if used else None


def final_rows(eval_df: pd.DataFrame, selected: Dict[float, float], out_dir: Path) -> None:
    # Auto final summary
    rows = []
    for theta in AUTO_TARGETS:
        sub = eval_df[(eval_df['mode'] == 'auto') & np.isclose(eval_df['target_angle_deg'].astype(float), theta) & np.isclose(eval_df['reward_scale'].astype(float), 1.0)]
        if sub.empty:
            continue
        step = sub['timestep'].max()
        final = sub[sub['timestep'] == step]
        for metric in ['eval_return_mean','mean_abs_angle_error_deg','target_occupancy','mean_squared_torque','alpha']:
            if metric in final.columns:
                rows.append({'table': 'auto_final', 'target_angle_deg': theta, 'metric': metric,
                             'mean': final[metric].mean(), 'std': final[metric].std(ddof=1), 'n_seeds': final['seed'].nunique(),
                             'ci95': 1.96 * final[metric].std(ddof=1) / np.sqrt(max(final['seed'].nunique(), 1))})
    pd.DataFrame(rows).to_csv(out_dir / 'summary_auto_final.csv', index=False)

    # Manual vs auto summary at selected targets
    rows = []
    for theta in SELECTED_TARGETS:
        alpha = selected.get(float(theta), np.nan)
        for label, mode, alpha_filter in [('auto', 'auto', None), ('manual', 'manual', alpha)]:
            sub = eval_df[(eval_df['mode'] == mode) & np.isclose(eval_df['target_angle_deg'].astype(float), theta) & np.isclose(eval_df['reward_scale'].astype(float), 1.0)]
            if alpha_filter is not None and not np.isnan(alpha_filter):
                sub = sub[np.isclose(sub['manual_alpha'].astype(float), alpha_filter)]
            if sub.empty:
                continue
            step = sub['timestep'].max(); final = sub[sub['timestep'] == step]
            rows.append({'target_angle_deg': theta, 'agent': label, 'manual_alpha': alpha if label == 'manual' else np.nan,
                         'final_step': step, 'final_return_mean_over_seeds': final['eval_return_mean'].mean(),
                         'final_return_std_over_seeds': final['eval_return_mean'].std(ddof=1), 'n_seeds': final['seed'].nunique(),
                         'ci95': 1.96 * final['eval_return_mean'].std(ddof=1) / np.sqrt(max(final['seed'].nunique(), 1))})
    pd.DataFrame(rows).to_csv(out_dir / 'summary_manual_vs_auto_final.csv', index=False)

    # Reward scaling summary
    rows = []
    for scale in [10.0, 0.1]:
        for mode in ['auto_scaled','manual_scaled']:
            sub = eval_df[(eval_df['mode'] == mode) & np.isclose(eval_df['target_angle_deg'].astype(float), 90.0) & np.isclose(eval_df['reward_scale'].astype(float), scale)]
            if sub.empty:
                continue
            step = sub['timestep'].max(); final = sub[sub['timestep'] == step]
            rows.append({'reward_scale': scale, 'agent': mode, 'final_step': step,
                         'final_return_mean_over_seeds': final['eval_return_mean'].mean(),
                         'final_return_std_over_seeds': final['eval_return_mean'].std(ddof=1), 'n_seeds': final['seed'].nunique(),
                         'ci95': 1.96 * final['eval_return_mean'].std(ddof=1) / np.sqrt(max(final['seed'].nunique(), 1))})
    pd.DataFrame(rows).to_csv(out_dir / 'summary_reward_scaling_final.csv', index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-root', default='outputs/pendulum_final_300k_clean')
    ap.add_argument('--plots-dir', default=None)
    ap.add_argument('--selected-alpha-json', default=None)
    ap.add_argument('--smooth-window', type=int, default=1, help='Use 1 for raw. Use 3 for a supplementary smoother plot.')
    args = ap.parse_args()

    output_root = Path(args.output_root)
    plots_dir = ensure_dir(args.plots_dir or (output_root / 'plots'))
    selected_path = Path(args.selected_alpha_json or (output_root / 'selected_manual_alphas.json'))
    selected = load_selected(selected_path)
    eval_df = load_all_eval_logs(output_root)
    if eval_df.empty:
        raise SystemExit(f'No eval logs found under {output_root}')
    # Save a combined log for transparent inspection.
    eval_df.to_csv(output_root / 'combined_eval_logs.csv', index=False)

    outputs = []
    for metric in METRICS:
        p = plot_auto_all_targets(eval_df, plots_dir, metric, args.smooth_window)
        if p: outputs.append(p)
        p = plot_manual_vs_auto_panels(eval_df, plots_dir, selected, metric, args.smooth_window)
        if p: outputs.append(p)
        p = plot_reward_scaling_panels(eval_df, plots_dir, selected, metric, args.smooth_window)
        if p: outputs.append(p)
    final_rows(eval_df, selected, output_root)
    print(f'Wrote {len(outputs)} plots to {plots_dir}')
    for p in outputs:
        print(' ', p)


if __name__ == '__main__':
    main()
