"""Run 2.2 LunarLander with 15 seeds, saving to logs_15seeds/.
Existing seed 0-5 logs are symlinked in so they are skipped automatically.
Only seeds 6-14 will actually train.

Usage (from 2.2/):
    conda activate pa3 && python run_15seeds.py 2>&1 | tee logs_15seeds/run.log
"""
import os, sys, time
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import runners  # import before patching

N_SEEDS          = 15
STEPS_CONTINUOUS = 400_000
STEPS_DISCRETE   = 300_000
STEPS_HOVER      = 300_000 + 300_000  # pre + post
EVAL_EVERY       = 10_000

LOG_DIR = "logs_15seeds"
SRC_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Redirect runners to save legacy JSON logs into logs_15seeds/ instead of logs/
runners.LOG_DIR = LOG_DIR

from runners import run_continuous, run_hover_switch, run_disc_sac, run_dqn
from sac_core import get_device

DEVICE = get_device()
print("Device:", DEVICE)

# Symlink existing seed 0-5 JSON logs so the exists() guard skips them
for fname in os.listdir(SRC_DIR):
    if not fname.endswith(".json"):
        continue
    dst = os.path.join(LOG_DIR, fname)
    src = os.path.abspath(os.path.join(SRC_DIR, fname))
    if not os.path.exists(dst):
        os.symlink(src, dst)

print(f"Symlinked existing logs from {SRC_DIR}/ into {LOG_DIR}/")

t0_total = time.time()


def skip_or_run(tag, seed, fn, *args, **kwargs):
    p = Path(LOG_DIR) / f"{tag}_seed{seed}.json"
    if p.exists():
        print(f"  {tag} seed={seed}: skip")
        return
    t0 = time.time()
    fn(*args, **kwargs)
    print(f"  {tag} seed={seed}: {time.time()-t0:.1f}s")


print("\n=== cont_auto ===")
for seed in range(N_SEEDS):
    skip_or_run("cont_auto", seed,
                run_continuous, seed, "cont_auto",
                total_steps=STEPS_CONTINUOUS, eval_every=EVAL_EVERY, device=DEVICE)

print("\n=== hover_fixed ===")
for seed in range(N_SEEDS):
    skip_or_run("hover_fixed", seed,
                run_hover_switch, seed, "hover_fixed",
                autotune=False, init_alpha=0.01,
                steps_pre=300_000, steps_post=300_000,
                eval_every=EVAL_EVERY, device=DEVICE)

print("\n=== hover_auto ===")
for seed in range(N_SEEDS):
    skip_or_run("hover_auto", seed,
                run_hover_switch, seed, "hover_auto",
                autotune=True, init_alpha=0.2,
                steps_pre=300_000, steps_post=300_000,
                eval_every=EVAL_EVERY, device=DEVICE)

print("\n=== disc_sac ===")
for seed in range(N_SEEDS):
    skip_or_run("disc_sac", seed,
                run_disc_sac, seed, "disc_sac",
                total_steps=STEPS_DISCRETE, eval_every=EVAL_EVERY, device=DEVICE)

print("\n=== dqn ===")
for seed in range(N_SEEDS):
    skip_or_run("dqn", seed,
                run_dqn, seed, "dqn",
                total_steps=STEPS_DISCRETE, eval_every=EVAL_EVERY, device=DEVICE)

print(f"\nAll done. Total wall time: {time.time()-t0_total:.1f}s")
print(f"Results saved to: {LOG_DIR}/")
