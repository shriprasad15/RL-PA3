"""Parallel dispatcher for PA3 Section 3 (PEBBLE bonus).

Usage (from 3/):
    python run_all.py --full                  # 6 seeds, full steps
    python run_all.py --smoke                 # 2 seeds, quick
    python run_all.py --workers 4             # limit parallelism
    python run_all.py --sections q1 q2 q3     # run specific sections

Sections:
    q1  -- SAC-GT Pendulum + PEBBLE Pendulum (5 angles × 6 seeds each)
    q2  -- PEBBLE feedback budget ablation (4 budgets × 6 seeds)
    q3  -- PEBBLE Reacher Ra/Rb/Rc (3 rewards × 6 seeds)

Outputs: logs/<tag>_seed<k>.json  (same convention as 2.2/2.3)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from runners import (run_sac_gt_pendulum, run_pebble_pendulum,
                     run_pebble_reacher, LOG_DIR)

PEND_ANGLES    = [0, -60, 90, 120, -150]
REACH_REWARDS  = ["Ra", "Rb", "Rc"]
ABLATION_ANGLE = 90
BUDGETS        = [50, 200, 500, 1000]


def _job(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
        return "ok"
    except Exception as e:
        return f"ERROR: {e}"


def build_jobs(sections, n_seeds, total_steps_pend, total_steps_reach,
               eval_every, pend_budget, reach_budget):
    jobs = []

    if "q1" in sections:
        for theta in PEND_ANGLES:
            tag = f"sac_gt_theta{theta}"
            for seed in range(n_seeds):
                p = Path(LOG_DIR) / f"{tag}_seed{seed}.json"
                if p.exists():
                    continue
                jobs.append((run_sac_gt_pendulum, theta, seed, tag,
                              dict(total_steps=total_steps_pend,
                                   eval_every=eval_every)))
        for theta in PEND_ANGLES:
            tag = f"pebble_pend_theta{theta}_budget{pend_budget}"
            for seed in range(n_seeds):
                p = Path(LOG_DIR) / f"{tag}_seed{seed}.json"
                if p.exists():
                    continue
                jobs.append((run_pebble_pendulum, theta, seed, tag,
                              dict(total_steps=total_steps_pend,
                                   eval_every=eval_every,
                                   budget=pend_budget)))

    if "q2" in sections:
        for budget in BUDGETS:
            tag = f"pebble_budget{budget}_theta{ABLATION_ANGLE}"
            for seed in range(n_seeds):
                p = Path(LOG_DIR) / f"{tag}_seed{seed}.json"
                if p.exists():
                    continue
                jobs.append((run_pebble_pendulum, ABLATION_ANGLE, seed, tag,
                              dict(total_steps=total_steps_pend,
                                   eval_every=eval_every,
                                   budget=budget)))

    if "q3" in sections:
        for reward in REACH_REWARDS:
            tag = f"pebble_reacher_{reward}_budget{reach_budget}"
            for seed in range(n_seeds):
                p = Path(LOG_DIR) / f"{tag}_seed{seed}.json"
                if p.exists():
                    continue
                jobs.append((run_pebble_reacher, reward, seed, tag,
                              dict(total_steps=total_steps_reach,
                                   eval_every=eval_every,
                                   budget=reach_budget)))

    return jobs


def run_jobs(jobs, workers):
    os.makedirs(LOG_DIR, exist_ok=True)
    total = len(jobs)
    if total == 0:
        print("All jobs already done — nothing to run.")
        return
    print(f"Launching {total} jobs with {workers} workers...")
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_job, fn, *args, **kwargs): (fn.__name__, args[:3])
                for fn, *args, kwargs in jobs}
        for fut in as_completed(futs):
            name, ids = futs[fut]
            result = fut.result()
            done += 1
            status = "✓" if result == "ok" else result
            print(f"  [{done}/{total}] {name}{ids} -> {status}")
    print("Done.")


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--full",  action="store_true", help="Full run (6 seeds, full steps)")
    mode.add_argument("--smoke", action="store_true", help="Smoke test (2 seeds, 40K steps)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sections", nargs="+", default=["q1", "q2", "q3"],
                    choices=["q1", "q2", "q3"])
    ap.add_argument("--seeds", type=int, default=None, help="Override seed count")
    args = ap.parse_args()

    if args.smoke:
        n_seeds = 2; steps_pend = 40_000; steps_reach = 40_000
        eval_every = 5_000; pend_budget = 200; reach_budget = 200
    else:
        n_seeds = 6; steps_pend = 200_000; steps_reach = 500_000
        eval_every = 10_000; pend_budget = 500; reach_budget = 1000

    if args.seeds is not None:
        n_seeds = args.seeds

    print(f"Mode: {'smoke' if args.smoke else 'full'} | seeds={n_seeds} | "
          f"sections={args.sections} | workers={args.workers}")

    jobs = build_jobs(args.sections, n_seeds, steps_pend, steps_reach,
                      eval_every, pend_budget, reach_budget)
    run_jobs(jobs, args.workers)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Total wall time: {time.time()-t0:.1f}s")
