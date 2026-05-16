"""Run 2.3 Reacher with 15 seeds, saving to logs_15seeds/.
Existing seed 0-5 logs are symlinked in so they are skipped automatically.
Only seeds 6-14 will actually train.

Usage (from 2.3/):
    conda activate pa3 && python run_15seeds.py 2>&1 | tee logs_15seeds/run.log
"""
import os, sys, time
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import runners  # import before patching

N_SEEDS     = 15
TOTAL_STEPS = 500_000
EVAL_EVERY  = 10_000

LOG_DIR = "logs_15seeds"
SRC_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Redirect runners to save legacy JSON logs into logs_15seeds/ instead of logs/
runners.LOG_DIR = LOG_DIR

from runners import run_reacher, is_done
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
REWARDS = ["Ra", "Rb", "Rc"]

for reward_name in REWARDS:
    print(f"\n=== {reward_name} ===")
    tag = f"reacher_{reward_name}"
    for seed in range(N_SEEDS):
        p = Path(LOG_DIR) / f"{tag}_seed{seed}.json"
        if p.exists():
            print(f"  {reward_name} seed={seed}: skip")
            continue
        t0 = time.time()
        run_reacher(seed, reward_name, tag,
                    total_steps=TOTAL_STEPS,
                    eval_every=EVAL_EVERY,
                    device=DEVICE)
        print(f"  {reward_name} seed={seed}: {time.time()-t0:.1f}s")

print(f"\nAll done. Total wall time: {time.time()-t0_total:.1f}s")
print(f"Results saved to: {LOG_DIR}/")
