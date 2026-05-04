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
from pathlib import Path

from runners import DEFAULT_OUTPUT_ROOT, LOG_DIR, is_done, run_reacher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward", required=True, choices=["Ra", "Rb", "Rc"])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--total-steps", required=True, type=int)
    parser.add_argument("--eval-every", required=True, type=int)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--no-rollout", action="store_true", help="skip final_rollout.pkl")
    parser.add_argument("--force", action="store_true", help="rerun even if artifacts/log exist")
    args = parser.parse_args()

    tag = f"reacher_{args.reward}"
    legacy_out_path = Path(LOG_DIR) / f"{tag}_seed{args.seed}.json"

    if not args.force:
        if is_done(args.output_root, tag, args.seed):
            print(f"SKIP: artifacts already complete for {tag} seed={args.seed}")
            sys.exit(0)
        if legacy_out_path.exists():
            print(f"SKIP: {legacy_out_path} already exists. Use --force to rerun.")
            sys.exit(0)

    print(
        f"RUN reward={args.reward} seed={args.seed} total_steps={args.total_steps} "
        f"eval_every={args.eval_every} output_root={args.output_root}"
    )
    run_reacher(
        seed=args.seed,
        reward_name=args.reward,
        tag=tag,
        total_steps=args.total_steps,
        eval_every=args.eval_every,
        output_root=args.output_root,
        save_rollout=not args.no_rollout,
    )
    print(f"DONE reward={args.reward} seed={args.seed}")


if __name__ == "__main__":
    main()
