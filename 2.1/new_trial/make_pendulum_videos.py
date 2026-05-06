#!/usr/bin/env python3
"""Generate rollout videos/GIFs from saved Pendulum policies.

By default this chooses the best seed for each requested condition using final eval return.
Use GIF output if mp4 support is missing on your machine.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from pendulum_sac_lib import generate_rollout_video, load_all_eval_logs, safe_angle_label, condition_name, ensure_dir


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(',') if x.strip()]


def load_selected(path: Path) -> Dict[float, float]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    if 'selected_alphas' in raw:
        raw = raw['selected_alphas']
    return {float(k): float(v) for k, v in raw.items()}


def best_seed_dir(eval_df: pd.DataFrame, output_root: Path, mode: str, theta: float, reward_scale: float = 1.0, manual_alpha: float | None = None) -> Path | None:
    sub = eval_df[(eval_df['mode'] == mode) & np.isclose(eval_df['target_angle_deg'].astype(float), float(theta)) & np.isclose(eval_df['reward_scale'].astype(float), reward_scale)]
    if manual_alpha is not None:
        sub = sub[np.isclose(sub['manual_alpha'].astype(float), float(manual_alpha))]
    if sub.empty:
        return None
    step = sub['timestep'].max()
    final = sub[sub['timestep'] == step].sort_values('eval_return_mean', ascending=False)
    seed = int(final.iloc[0]['seed'])
    cond = condition_name(mode, theta, reward_scale, manual_alpha)
    return output_root / cond / f'seed_{seed:02d}'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-root', default='outputs/pendulum_final_300k_clean')
    ap.add_argument('--videos-dir', default=None)
    ap.add_argument('--selected-alpha-json', default=None)
    ap.add_argument('--targets', default='0,-10,30,-60,90,-90,120,-150')
    ap.add_argument('--include-manual', action='store_true')
    ap.add_argument('--include-scaling', action='store_true')
    ap.add_argument('--checkpoint', default='best_model.pt', choices=['best_model.pt','final_model.pt'])
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--ext', default='gif', choices=['gif','mp4'])
    ap.add_argument('--fps', type=int, default=30)
    args = ap.parse_args()

    output_root = Path(args.output_root)
    videos_dir = ensure_dir(args.videos_dir or (output_root / 'videos'))
    selected = load_selected(Path(args.selected_alpha_json or (output_root / 'selected_manual_alphas.json')))
    eval_df = load_all_eval_logs(output_root)
    if eval_df.empty:
        raise SystemExit(f'No eval logs found under {output_root}')

    targets = parse_float_list(args.targets)
    made = []
    # Auto all requested targets
    for theta in targets:
        seed_dir = best_seed_dir(eval_df, output_root, 'auto', theta, 1.0)
        if seed_dir and seed_dir.exists():
            out = videos_dir / f'auto_target_{safe_angle_label(theta)}.{args.ext}'
            made.append(generate_rollout_video(seed_dir, out, checkpoint_name=args.checkpoint, device=args.device, fps=args.fps))

    if args.include_manual:
        for theta, alpha in selected.items():
            seed_dir = best_seed_dir(eval_df, output_root, 'manual', theta, 1.0, alpha)
            if seed_dir and seed_dir.exists():
                out = videos_dir / f'manual_target_{safe_angle_label(theta)}_alpha_{str(alpha).replace(".","p")}.{args.ext}'
                made.append(generate_rollout_video(seed_dir, out, checkpoint_name=args.checkpoint, device=args.device, fps=args.fps))

    if args.include_scaling:
        alpha = selected.get(90.0)
        for scale in [10.0, 0.1]:
            seed_dir = best_seed_dir(eval_df, output_root, 'auto_scaled', 90.0, scale)
            if seed_dir and seed_dir.exists():
                out = videos_dir / f'auto_target_p090_scale_{scale:g}.{args.ext}'
                made.append(generate_rollout_video(seed_dir, out, checkpoint_name=args.checkpoint, device=args.device, fps=args.fps))
            if alpha is not None:
                seed_dir = best_seed_dir(eval_df, output_root, 'manual_scaled', 90.0, scale, alpha)
                if seed_dir and seed_dir.exists():
                    out = videos_dir / f'manual_target_p090_scale_{scale:g}_alpha_{str(alpha).replace(".","p")}.{args.ext}'
                    made.append(generate_rollout_video(seed_dir, out, checkpoint_name=args.checkpoint, device=args.device, fps=args.fps))

    print(f'Wrote {len(made)} videos to {videos_dir}')
    for p in made:
        print(' ', p)


if __name__ == '__main__':
    main()
