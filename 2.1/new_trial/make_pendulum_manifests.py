#!/usr/bin/env python3
"""Create JSONL job manifests for PA3 Section 2.1 Pendulum SAC.

Stages:
  auto           : 8 targets × 15 seeds, automatic alpha
  manual_search  : selected targets × alpha grid × 3 tuning seeds, fixed alpha
  manual_final   : selected targets × selected alpha per target × 15 seeds
  reward_scaling : target 90° × scales {10,0.1} × {auto,manual} × 15 seeds

The runner is run_pendulum_jobs.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

AUTO_TARGETS = [0, -10, 30, -60, 90, -90, 120, -150]
SELECTED_TARGETS = [-60, 90, 120, -150]
DEFAULT_ALPHA_GRID = [0.01, 0.03, 0.05, 0.1, 0.2, 0.5]
REWARD_SCALES = [10.0, 0.1]


def parse_int_list(text: str) -> List[int]:
    text = str(text).strip()
    if '-' in text and ',' not in text:
        a, b = text.split('-', 1)
        return list(range(int(a), int(b) + 1))
    return [int(x.strip()) for x in text.split(',') if x.strip()]


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(',') if x.strip()]


def load_cfg(path: str | Path) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_jsonl(path: Path, jobs: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        for j in jobs:
            f.write(json.dumps(j) + '\n')
    print(f"Wrote {len(jobs)} jobs -> {path}")


def with_common(base_cfg: Dict[str, Any], output_root: Path, theta: float, seed: int,
                mode: str, total_timesteps: int, device: str,
                reward_scale: float = 1.0, manual_alpha: float | None = None) -> Dict[str, Any]:
    return {
        'base_cfg': base_cfg,
        'output_root': str(output_root),
        'target_angle_deg': float(theta),
        'seed': int(seed),
        'mode': mode,
        'reward_scale': float(reward_scale),
        'manual_alpha': None if manual_alpha is None else float(manual_alpha),
        'total_timesteps': int(total_timesteps),
        'device': device,
    }


def build_auto(base_cfg, output_root, targets, seeds, total_timesteps, device):
    return [with_common(base_cfg, output_root, t, s, 'auto', total_timesteps, device)
            for t in targets for s in seeds]


def build_manual_search(base_cfg, output_root, targets, tuning_seeds, alpha_grid, total_timesteps, device):
    return [with_common(base_cfg, output_root, t, s, 'manual', total_timesteps, device, manual_alpha=a)
            for t in targets for a in alpha_grid for s in tuning_seeds]


def load_selected_alphas(path: str | Path) -> Dict[float, float]:
    with open(path, 'r') as f:
        raw = json.load(f)
    # Accept either {"-60": 0.01} or {"selected_alphas": {...}}
    if 'selected_alphas' in raw:
        raw = raw['selected_alphas']
    return {float(k): float(v) for k, v in raw.items()}


def build_manual_final(base_cfg, output_root, targets, seeds, selected_alpha_json, total_timesteps, device):
    selected = load_selected_alphas(selected_alpha_json)
    missing = [t for t in targets if float(t) not in selected]
    if missing:
        raise SystemExit(f"Missing selected alpha for targets: {missing}. Run select_manual_alphas.py first.")
    return [with_common(base_cfg, output_root, t, s, 'manual', total_timesteps, device, manual_alpha=selected[float(t)])
            for t in targets for s in seeds]


def build_reward_scaling(base_cfg, output_root, seeds, selected_alpha_json, total_timesteps, device, target=90.0):
    selected = load_selected_alphas(selected_alpha_json)
    if float(target) not in selected:
        raise SystemExit(f"Missing selected alpha for target {target}. Run select_manual_alphas.py first.")
    jobs = []
    for scale in REWARD_SCALES:
        for s in seeds:
            jobs.append(with_common(base_cfg, output_root, target, s, 'auto_scaled', total_timesteps, device, reward_scale=scale))
            jobs.append(with_common(base_cfg, output_root, target, s, 'manual_scaled', total_timesteps, device,
                                    reward_scale=scale, manual_alpha=selected[float(target)]))
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/pendulum_final_300k.yaml')
    ap.add_argument('--output-root', default='outputs/pendulum_final_300k_clean')
    ap.add_argument('--stage', required=True,
                    choices=['auto', 'manual_search', 'manual_final', 'reward_scaling', 'all_possible'])
    ap.add_argument('--total-timesteps', type=int, default=None,
                    help='Default comes from config. Use 100000 only for quick pilot, 300000 for final.')
    ap.add_argument('--seeds', default='0-14')
    ap.add_argument('--tuning-seeds', default='0,1,2')
    ap.add_argument('--targets', default=','.join(map(str, AUTO_TARGETS)))
    ap.add_argument('--selected-targets', default=','.join(map(str, SELECTED_TARGETS)))
    ap.add_argument('--alpha-grid', default=','.join(map(str, DEFAULT_ALPHA_GRID)))
    ap.add_argument('--selected-alpha-json', default=None)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    base_cfg = load_cfg(args.config)
    total_timesteps = int(args.total_timesteps or base_cfg['experiment']['total_timesteps'])
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Save exact base config used for the run folder.
    with open(output_root / 'base_config_used.yaml', 'w') as f:
        yaml.safe_dump(base_cfg, f, sort_keys=False)

    seeds = parse_int_list(args.seeds)
    tuning_seeds = parse_int_list(args.tuning_seeds)
    targets = parse_float_list(args.targets)
    selected_targets = parse_float_list(args.selected_targets)
    alpha_grid = parse_float_list(args.alpha_grid)

    if args.stage == 'auto':
        jobs = build_auto(base_cfg, output_root, targets, seeds, total_timesteps, args.device)
        save_jsonl(output_root / 'auto_jobs.jsonl', jobs)
    elif args.stage == 'manual_search':
        jobs = build_manual_search(base_cfg, output_root, selected_targets, tuning_seeds, alpha_grid, total_timesteps, args.device)
        save_jsonl(output_root / 'manual_search_jobs.jsonl', jobs)
    elif args.stage == 'manual_final':
        if args.selected_alpha_json is None:
            args.selected_alpha_json = str(output_root / 'selected_manual_alphas.json')
        jobs = build_manual_final(base_cfg, output_root, selected_targets, seeds, args.selected_alpha_json, total_timesteps, args.device)
        save_jsonl(output_root / 'manual_final_jobs.jsonl', jobs)
    elif args.stage == 'reward_scaling':
        if args.selected_alpha_json is None:
            args.selected_alpha_json = str(output_root / 'selected_manual_alphas.json')
        jobs = build_reward_scaling(base_cfg, output_root, seeds, args.selected_alpha_json, total_timesteps, args.device, target=90.0)
        save_jsonl(output_root / 'reward_scaling_jobs.jsonl', jobs)
    elif args.stage == 'all_possible':
        save_jsonl(output_root / 'auto_jobs.jsonl', build_auto(base_cfg, output_root, targets, seeds, total_timesteps, args.device))
        save_jsonl(output_root / 'manual_search_jobs.jsonl', build_manual_search(base_cfg, output_root, selected_targets, tuning_seeds, alpha_grid, total_timesteps, args.device))
        print('manual_final and reward_scaling require selected_manual_alphas.json, so create them after manual search.')


if __name__ == '__main__':
    main()
