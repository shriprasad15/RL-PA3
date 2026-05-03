"""CLI for launching one PA3 Reacher experiment.

Examples from inside the Section 2.3 folder:
    python run_experiment.py --reward Ra --seed 0 --total-steps 500000 --eval-every 10000
    python run_experiment.py --reward Rb --seed 0 --total-steps 500000 --eval-every 10000
    python run_experiment.py --reward Rc --seed 0 --total-steps 500000 --eval-every 10000

For all seeds sequentially:
    for reward in Ra Rb Rc; do
      for seed in {0..14}; do
        python run_experiment.py --reward $reward --seed $seed --total-steps 500000 --eval-every 10000
      done
    done
"""
from __future__ import annotations

import argparse
import os
import sys

from runners import LOG_DIR, run_reacher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward", required=True, choices=["Ra", "Rb", "Rc"])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--total-steps", required=True, type=int)
    parser.add_argument("--eval-every", required=True, type=int)
    parser.add_argument("--force", action="store_true", help="rerun even if log exists")
    args = parser.parse_args()

    tag = f"reacher_{args.reward}"
    out_path = os.path.join(LOG_DIR, f"{tag}_seed{args.seed}.json")

    if os.path.exists(out_path) and not args.force:
        print(f"SKIP: {out_path} already exists. Use --force to rerun.")
        sys.exit(0)

    print(
        f"RUN reward={args.reward} seed={args.seed} "
        f"total_steps={args.total_steps} eval_every={args.eval_every}"
    )
    run_reacher(
        seed=args.seed,
        reward_name=args.reward,
        tag=tag,
        total_steps=args.total_steps,
        eval_every=args.eval_every,
    )
    print(f"DONE reward={args.reward} seed={args.seed}")


if __name__ == "__main__":
    main()
