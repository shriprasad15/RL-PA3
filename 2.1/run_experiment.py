"""CLI for launching one Pendulum experiment.

Usage (from inside the 2.1/ folder):
    python run_experiment.py --kind auto    --theta 90 --seed 3 --total-steps 150000 --eval-every 10000
    python run_experiment.py --kind manual  --theta 90 --seed 3 --alpha 0.2 --total-steps 150000 --eval-every 10000
    python run_experiment.py --kind scale   --theta 90 --seed 3 --scale 10.0 --mode auto   --total-steps 150000 --eval-every 10000

Exits 0 on success. Skips and exits 0 if the output log already exists.
"""
from __future__ import annotations

import argparse
import os
import sys

from runners import run_one, LOG_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["auto", "manual", "scale"])
    ap.add_argument("--theta", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--mode", choices=["auto", "manual"], default="auto",
                    help="only for --kind scale")
    ap.add_argument("--total-steps", type=int, required=True)
    ap.add_argument("--eval-every", type=int, required=True)
    args = ap.parse_args()

    if args.kind == "auto":
        tag = f"auto_theta{args.theta}"
        autotune, init_alpha, reward_scale = True, 0.2, 1.0
    elif args.kind == "manual":
        tag = f"manual_a{args.alpha}_theta{args.theta}"
        autotune, init_alpha, reward_scale = False, args.alpha, 1.0
    elif args.kind == "scale":
        tag = f"scale{args.scale}_{args.mode}_theta{args.theta}"
        autotune = args.mode == "auto"
        init_alpha = args.alpha if args.mode == "manual" else 0.2
        reward_scale = args.scale
    else:
        print(f"unknown kind: {args.kind}", file=sys.stderr); sys.exit(2)

    out = os.path.join(LOG_DIR, f"{tag}_seed{args.seed}.json")
    if os.path.exists(out):
        print(f"SKIP: {out} exists"); sys.exit(0)

    print(f"RUN  {tag} seed={args.seed}")
    run_one(args.theta, args.seed, tag,
            total_steps=args.total_steps, eval_every=args.eval_every,
            reward_scale=reward_scale, autotune=autotune, init_alpha=init_alpha)
    print(f"DONE {tag} seed={args.seed}")


if __name__ == "__main__":
    main()
