"""CLI for launching one bonus-section experiment.

Usage from inside 3/:
    python run_experiment.py --kind sac_gt  --theta 90 --seed 3 --total-steps 200000 --eval-every 10000
    python run_experiment.py --kind pebble_pend --theta 90 --seed 3 --budget 500 --total-steps 200000 --eval-every 10000
    python run_experiment.py --kind pebble_reacher --reward Ra --seed 3 --budget 1000 --total-steps 500000 --eval-every 10000
"""
from __future__ import annotations

import argparse
import os
import sys

from runners import (run_sac_gt_pendulum, run_pebble_pendulum, run_pebble_reacher, LOG_DIR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["sac_gt", "pebble_pend", "pebble_reacher"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--theta", type=int, default=None)
    ap.add_argument("--reward", choices=["Ra", "Rb", "Rc"], default=None)
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--total-steps", type=int, required=True)
    ap.add_argument("--eval-every", type=int, required=True)
    ap.add_argument("--tag", type=str, default=None,
                    help="explicit log tag; overrides auto-generated tag")
    args = ap.parse_args()

    if args.kind == "sac_gt":
        assert args.theta is not None
        tag = f"sacgt_theta{args.theta}"
        out = os.path.join(LOG_DIR, f"{tag}_seed{args.seed}.json")
        if os.path.exists(out): print(f"SKIP: {out}"); sys.exit(0)
        print(f"RUN  {tag} seed={args.seed}")
        run_sac_gt_pendulum(args.theta, args.seed, tag,
                             total_steps=args.total_steps, eval_every=args.eval_every)
    elif args.kind == "pebble_pend":
        assert args.theta is not None and args.budget is not None
        tag = args.tag or f"pebble_pend_theta{args.theta}"
        out = os.path.join(LOG_DIR, f"{tag}_seed{args.seed}.json")
        if os.path.exists(out): print(f"SKIP: {out}"); sys.exit(0)
        print(f"RUN  {tag} seed={args.seed}")
        run_pebble_pendulum(args.theta, args.seed, tag,
                             total_steps=args.total_steps, eval_every=args.eval_every,
                             budget=args.budget)
    elif args.kind == "pebble_reacher":
        assert args.reward is not None and args.budget is not None
        tag = f"pebble_reacher_{args.reward}"
        out = os.path.join(LOG_DIR, f"{tag}_seed{args.seed}.json")
        if os.path.exists(out): print(f"SKIP: {out}"); sys.exit(0)
        print(f"RUN  {tag} seed={args.seed}")
        run_pebble_reacher(args.reward, args.seed, tag,
                            total_steps=args.total_steps, eval_every=args.eval_every,
                            budget=args.budget)
    print(f"DONE seed={args.seed}")


if __name__ == "__main__":
    main()
