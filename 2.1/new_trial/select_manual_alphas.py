#!/usr/bin/env python3
"""Select fixed manual alpha per target using tuning seeds and final-return + AUC score."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from pendulum_sac_lib import load_all_eval_logs, ensure_dir


def parse_int_list(text: str) -> List[int]:
    if '-' in str(text) and ',' not in str(text):
        a, b = str(text).split('-', 1)
        return list(range(int(a), int(b) + 1))
    return [int(x.strip()) for x in str(text).split(',') if x.strip()]


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(',') if x.strip()]


def normalize(s: pd.Series) -> pd.Series:
    lo, hi = float(s.min()), float(s.max())
    if np.isclose(hi, lo):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def choose_best_manual_alphas_auc(eval_df: pd.DataFrame, selected_targets: List[float], tuning_seeds: List[int],
                                  w_final: float = 0.5, w_auc: float = 0.5) -> tuple[Dict[float, float], pd.DataFrame]:
    manual = eval_df[
        (eval_df['mode'] == 'manual')
        & np.isclose(eval_df['reward_scale'].astype(float), 1.0)
        & (eval_df['seed'].astype(int).isin(tuning_seeds))
    ].copy()
    if manual.empty:
        raise SystemExit('No manual search eval rows found. Run manual_search_jobs first.')

    all_rows = []
    selected: Dict[float, float] = {}
    for theta in selected_targets:
        sub_t = manual[np.isclose(manual['target_angle_deg'].astype(float), float(theta))].copy()
        if sub_t.empty:
            raise SystemExit(f'No manual search rows found for target {theta}')
        rows = []
        for alpha, sub_a in sub_t.groupby('manual_alpha'):
            seeds_present = sorted(int(x) for x in sub_a['seed'].unique())
            curve = sub_a.groupby('timestep', as_index=False)['eval_return_mean'].mean().sort_values('timestep')
            final_return = float(curve['eval_return_mean'].iloc[-1])
            auc = float(np.trapezoid(curve['eval_return_mean'].to_numpy(), curve['timestep'].to_numpy())
                        if hasattr(np, 'trapezoid') else
                        np.trapz(curve['eval_return_mean'].to_numpy(), curve['timestep'].to_numpy()))
            rows.append({
                'target_angle_deg': float(theta),
                'manual_alpha': float(alpha),
                'n_seeds': len(seeds_present),
                'seeds_present': ','.join(map(str, seeds_present)),
                'final_return': final_return,
                'auc': auc,
            })
        score_df = pd.DataFrame(rows)
        score_df['final_return_norm'] = normalize(score_df['final_return'])
        score_df['auc_norm'] = normalize(score_df['auc'])
        score_df['selection_score'] = w_final * score_df['final_return_norm'] + w_auc * score_df['auc_norm']
        score_df = score_df.sort_values(['selection_score', 'final_return', 'auc'], ascending=False)
        selected[float(theta)] = float(score_df.iloc[0]['manual_alpha'])
        all_rows.append(score_df)

    summary = pd.concat(all_rows, ignore_index=True).sort_values(['target_angle_deg', 'selection_score'], ascending=[True, False])
    return selected, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-root', default='outputs/pendulum_final_300k_clean')
    ap.add_argument('--selected-targets', default='-60,90,120,-150')
    ap.add_argument('--tuning-seeds', default='0,1,2')
    ap.add_argument('--w-final', type=float, default=0.5)
    ap.add_argument('--w-auc', type=float, default=0.5)
    args = ap.parse_args()

    output_root = Path(args.output_root)
    eval_df = load_all_eval_logs(output_root)
    selected_targets = parse_float_list(args.selected_targets)
    tuning_seeds = parse_int_list(args.tuning_seeds)

    selected, summary = choose_best_manual_alphas_auc(eval_df, selected_targets, tuning_seeds, args.w_final, args.w_auc)
    out_json = output_root / 'selected_manual_alphas.json'
    out_csv = output_root / 'manual_alpha_selection_table.csv'
    ensure_dir(output_root)
    with open(out_json, 'w') as f:
        json.dump({
            'selection_rule': f'{args.w_final}*normalized_final_return + {args.w_auc}*normalized_AUC',
            'tuning_seeds': tuning_seeds,
            'selected_alphas': {str(k): v for k, v in selected.items()},
        }, f, indent=2)
    summary.to_csv(out_csv, index=False)
    print('Selected manual alphas:')
    for k, v in selected.items():
        print(f'  target {k:g}° -> alpha={v:g}')
    print(f'Wrote {out_json}')
    print(f'Wrote {out_csv}')


if __name__ == '__main__':
    main()
