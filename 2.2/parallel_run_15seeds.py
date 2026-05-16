"""Parallel dispatcher for 2.2 LunarLander 15-seed runs.
Skips already-completed seeds. Uses ProcessPoolExecutor for parallelism.

Usage (from 2.2/):
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

# Symlink existing seed 0-5 logs on startup
for fname in os.listdir(SRC_DIR):
    if not fname.endswith(".json"):
        continue
    dst = os.path.join(LOG_DIR, fname)
    src = os.path.abspath(os.path.join(SRC_DIR, fname))
    if not os.path.exists(dst):
        os.symlink(src, dst)


def _job(exp, seed, kwargs):
    """Worker function — runs in a subprocess."""
    import os, sys
    sys.path.insert(0, os.path.dirname(__file__))
    import runners
    runners.LOG_DIR = LOG_DIR
    from runners import run_continuous, run_hover_switch, run_disc_sac, run_dqn
    from sac_core import get_device
    device = get_device()

    try:
        if exp == "cont_auto":
            run_continuous(seed, "cont_auto", device=device, **kwargs)
        elif exp == "hover_fixed":
            run_hover_switch(seed, "hover_fixed", autotune=False, init_alpha=0.01,
                             steps_pre=300_000, steps_post=300_000,
                             eval_every=10_000, device=device)
        elif exp == "hover_auto":
            run_hover_switch(seed, "hover_auto", autotune=True, init_alpha=0.2,
                             steps_pre=300_000, steps_post=300_000,
                             eval_every=10_000, device=device)
        elif exp == "disc_sac":
            run_disc_sac(seed, "disc_sac", device=device, **kwargs)
        elif exp == "dqn":
            run_dqn(seed, "dqn", device=device, **kwargs)
        return "ok"
    except Exception as e:
        import traceback
        return f"ERROR: {e}\n{traceback.format_exc()}"


def build_jobs():
    jobs = []
    experiments = [
        ("cont_auto",   {"total_steps": 400_000, "eval_every": 10_000}),
        ("hover_fixed", {}),
        ("hover_auto",  {}),
        ("disc_sac",    {"total_steps": 300_000, "eval_every": 10_000}),
        ("dqn",         {"total_steps": 300_000, "eval_every": 10_000}),
    ]
    for exp, kwargs in experiments:
        for seed in range(N_SEEDS):
            p = Path(LOG_DIR) / f"{exp}_seed{seed}.json"
            if p.exists() and p.stat().st_size > 100:  # skip completed (non-empty)
                continue
            jobs.append((exp, seed, kwargs))
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
    print(f"Jobs: {[(e, s) for e, s, _ in jobs]}\n")

    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_job, exp, seed, kw): (exp, seed)
                for exp, seed, kw in jobs}
        for fut in as_completed(futs):
            exp, seed = futs[fut]
            result = fut.result()
            done += 1
            status = "✓" if result == "ok" else result[:80]
            elapsed = time.time() - t0
            print(f"[{done}/{total}] {exp} seed={seed} -> {status}  "
                  f"(elapsed {elapsed/3600:.1f}h)")

    print(f"\nAll done. Total wall time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
