"""Parallel dispatcher for 2.3 Reacher 15-seed runs.
Skips already-completed seeds. Uses ProcessPoolExecutor for parallelism.

Usage (from 2.3/):
    python parallel_run_15seeds.py --workers 8
    python parallel_run_15seeds.py --workers 8 2>&1 | tee logs_15seeds/parallel_run.log
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

LOG_DIR = "logs_15seeds"
SRC_DIR = "logs"
N_SEEDS = 15
os.makedirs(LOG_DIR, exist_ok=True)

for fname in os.listdir(SRC_DIR):
    if not fname.endswith(".json"):
        continue
    dst = os.path.join(LOG_DIR, fname)
    src = os.path.abspath(os.path.join(SRC_DIR, fname))
    if not os.path.exists(dst):
        os.symlink(src, dst)


def _job(reward_name, seed):
    import os, sys
    sys.path.insert(0, os.path.dirname(__file__))
    import runners
    runners.LOG_DIR = LOG_DIR
    from runners import run_reacher
    from sac_core import get_device
    device = get_device()
    try:
        tag = f"reacher_{reward_name}"
        run_reacher(seed, reward_name, tag,
                    total_steps=500_000,
                    eval_every=10_000,
                    device=device)
        return "ok"
    except Exception as e:
        import traceback
        return f"ERROR: {e}\n{traceback.format_exc()}"


def build_jobs():
    jobs = []
    for reward_name in ["Ra", "Rb", "Rc"]:
        for seed in range(N_SEEDS):
            p = Path(LOG_DIR) / f"reacher_{reward_name}_seed{seed}.json"
            if p.exists() and p.stat().st_size > 100:
                continue
            jobs.append((reward_name, seed))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    jobs = build_jobs()
    total = len(jobs)
    if total == 0:
        print("All jobs done!")
        return

    print(f"Launching {total} jobs with {args.workers} workers...")
    print(f"Jobs: {jobs}\n")

    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_job, r, s): (r, s) for r, s in jobs}
        for fut in as_completed(futs):
            r, s = futs[fut]
            result = fut.result()
            done += 1
            status = "✓" if result == "ok" else result[:80]
            elapsed = time.time() - t0
            print(f"[{done}/{total}] reacher_{r} seed={s} -> {status}  "
                  f"(elapsed {elapsed/3600:.1f}h)")

    print(f"\nAll done. Total wall time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
