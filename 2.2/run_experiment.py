"""CLI for launching one PA3 2.2 LunarLander experiment.

Examples:
    python run_experiment.py --kind continuous --seed 0 --total-steps 500000 --eval-every 10000

    python run_experiment.py --kind hover --mode fixed --seed 0 \
      --steps-pre 250000 --steps-post 250000 --eval-every 10000

    python run_experiment.py --kind hover --mode auto --seed 0 \
      --steps-pre 250000 --steps-post 250000 --eval-every 10000

    python run_experiment.py --kind disc --seed 0 --total-steps 500000 --eval-every 10000
    python run_experiment.py --kind dqn  --seed 0 --total-steps 500000 --eval-every 10000

The new artifact layout is:
    runs/<tag>/seed_XX/{config.json,eval_log.csv,train_log.csv,best_model.pt,final_model.pt,DONE.txt}

For backwards compatibility, this also writes:
    logs/<tag>_seed<seed>.json
"""
from __future__ import annotations

import argparse
import os
import sys

from runners import run_continuous, run_hover_switch, run_disc_sac, run_dqn, RUNS_DIR


def _done_path(output_root: str, tag: str, seed: int) -> str:
    return os.path.join(output_root, tag, f"seed_{seed:02d}", "DONE.txt")


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
    ap.add_argument("--output-root", type=str, default=RUNS_DIR,
                    help="artifact root directory; default: runs")
    ap.add_argument("--force", action="store_true",
                    help="rerun even if DONE.txt exists")
    args = ap.parse_args()

    if args.kind == "continuous":
        tag = "cont_auto"
    elif args.kind == "hover":
        if args.mode is None:
            raise SystemExit("--mode auto|fixed is required for --kind hover")
        tag = f"hover_{args.mode}"
    elif args.kind == "disc":
        tag = "disc_sac"
    elif args.kind == "dqn":
        tag = "dqn"
    else:
        raise SystemExit(f"unknown kind: {args.kind}")

    done = _done_path(args.output_root, tag, args.seed)
    if os.path.exists(done) and not args.force:
        print(f"SKIP: {done} exists")
        sys.exit(0)

    print(f"RUN  {tag} seed={args.seed}")

    if args.kind == "continuous":
        if args.total_steps is None:
            raise SystemExit("--total-steps is required for continuous")
        run_continuous(
            args.seed,
            tag,
            total_steps=args.total_steps,
            eval_every=args.eval_every,
            output_root=args.output_root,
        )

    elif args.kind == "hover":
        if args.steps_pre is None or args.steps_post is None:
            raise SystemExit("--steps-pre and --steps-post are required for hover")
        autotune = args.mode == "auto"
        init_alpha = 0.2 if autotune else 0.01
        run_hover_switch(
            args.seed,
            tag,
            autotune=autotune,
            init_alpha=init_alpha,
            steps_pre=args.steps_pre,
            steps_post=args.steps_post,
            eval_every=args.eval_every,
            output_root=args.output_root,
        )

    elif args.kind == "disc":
        if args.total_steps is None:
            raise SystemExit("--total-steps is required for disc")
        run_disc_sac(
            args.seed,
            tag,
            total_steps=args.total_steps,
            eval_every=args.eval_every,
            output_root=args.output_root,
        )

    elif args.kind == "dqn":
        if args.total_steps is None:
            raise SystemExit("--total-steps is required for dqn")
        run_dqn(
            args.seed,
            tag,
            total_steps=args.total_steps,
            eval_every=args.eval_every,
            output_root=args.output_root,
        )

    print(f"DONE {tag} seed={args.seed}")


if __name__ == "__main__":
    main()
