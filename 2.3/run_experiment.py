"""CLI for launching one Reacher experiment.

Usage from inside 2.3/:
    python run_experiment.py --reward Ra --seed 3 --total-steps 500000 --eval-every 10000
"""
from __future__ import annotations

import argparse
import os
import sys

from runners import run_reacher, LOG_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward", required=True, choices=["Ra", "Rb", "Rc"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--total-steps", type=int, required=True)
    ap.add_argument("--eval-every", type=int, required=True)
    args = ap.parse_args()

    tag = f"reacher_{args.reward}"
    out = os.path.join(LOG_DIR, f"{tag}_seed{args.seed}.json")
    if os.path.exists(out):
        print(f"SKIP: {out} exists"); sys.exit(0)
    print(f"RUN  {tag} seed={args.seed}")
    run_reacher(args.seed, args.reward, tag,
                total_steps=args.total_steps, eval_every=args.eval_every)
    print(f"DONE {tag} seed={args.seed}")


if __name__ == "__main__":
    main()
