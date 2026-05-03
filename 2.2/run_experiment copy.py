"""CLI for launching one LunarLander experiment.

Usage from inside 2.2/:
    python run_experiment.py --kind continuous --seed 3 --total-steps 400000 --eval-every 10000
    python run_experiment.py --kind hover --seed 3 --mode auto  --steps-pre 300000 --steps-post 300000 --eval-every 10000
    python run_experiment.py --kind hover --seed 3 --mode fixed --steps-pre 300000 --steps-post 300000 --eval-every 10000
    python run_experiment.py --kind disc --seed 3 --total-steps 300000 --eval-every 10000
    python run_experiment.py --kind dqn  --seed 3 --total-steps 300000 --eval-every 10000
"""
from __future__ import annotations

import argparse
import os
import sys

from runners import run_continuous, run_hover_switch, run_disc_sac, run_dqn, LOG_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["continuous", "hover", "disc", "dqn"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--eval-every", type=int, required=True)
    ap.add_argument("--mode", choices=["auto", "fixed"], default=None,
                    help="only for --kind hover")
    ap.add_argument("--steps-pre", type=int, default=None)
    ap.add_argument("--steps-post", type=int, default=None)
    args = ap.parse_args()

    if args.kind == "continuous":
        tag = "cont_auto"
    elif args.kind == "hover":
        tag = f"hover_{args.mode}"
    elif args.kind == "disc":
        tag = "disc_sac"
    elif args.kind == "dqn":
        tag = "dqn"
    else:
        print(f"unknown: {args.kind}", file=sys.stderr); sys.exit(2)

    out = os.path.join(LOG_DIR, f"{tag}_seed{args.seed}.json")
    if os.path.exists(out):
        print(f"SKIP: {out} exists"); sys.exit(0)

    print(f"RUN  {tag} seed={args.seed}")
    if args.kind == "continuous":
        run_continuous(args.seed, tag,
                       total_steps=args.total_steps, eval_every=args.eval_every)
    elif args.kind == "hover":
        autotune = args.mode == "auto"
        init_alpha = 0.2 if autotune else 0.01
        run_hover_switch(args.seed, tag, autotune=autotune, init_alpha=init_alpha,
                         steps_pre=args.steps_pre, steps_post=args.steps_post,
                         eval_every=args.eval_every)
    elif args.kind == "disc":
        run_disc_sac(args.seed, tag,
                     total_steps=args.total_steps, eval_every=args.eval_every)
    elif args.kind == "dqn":
        run_dqn(args.seed, tag,
                total_steps=args.total_steps, eval_every=args.eval_every)
    print(f"DONE {tag} seed={args.seed}")


if __name__ == "__main__":
    main()
