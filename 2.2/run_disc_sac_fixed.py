"""Quick re-run of Discrete SAC with target_entropy_ratio=0.98 (fix alpha explosion).
Saves to logs_disc_fixed/ so original results are untouched.

Usage: python run_disc_sac_fixed.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))

from sac_core import set_global_seed, get_device, save_log
from discrete_sac_agent import DiscreteSACAgent, DiscreteSACConfig, train_discrete_sac
from lunar_env import make_lunar_discrete

N_SEEDS = 4
TOTAL_STEPS = 300_000
EVAL_EVERY = 10_000
EVAL_EPISODES = 20

LOG_DIR = "logs_disc_fixed"
os.makedirs(LOG_DIR, exist_ok=True)

DEVICE = get_device()
print(f"Device: {DEVICE}")
print(f"Fix: target_entropy_ratio=0.98 (was 0.50)")

t0_total = time.time()

for seed in range(N_SEEDS):
    p = os.path.join(LOG_DIR, f"disc_sac_fixed_seed{seed}.json")
    if os.path.exists(p) and os.path.getsize(p) > 100:
        print(f"  disc_sac_fixed seed={seed}: skip")
        continue

    t0 = time.time()
    set_global_seed(seed)
    env_fn = lambda: make_lunar_discrete()
    e = env_fn()
    obs_dim = e.observation_space.shape[0]
    n_actions = e.action_space.n
    e.close()

    cfg = DiscreteSACConfig(
        start_steps=10_000,
        update_after=1_000,
        target_entropy_ratio=0.98,  # FIX: was 0.50
    )
    agent = DiscreteSACAgent(obs_dim, n_actions, cfg, device=DEVICE)
    log = train_discrete_sac(env_fn, agent, total_steps=TOTAL_STEPS, eval_env_fn=env_fn,
                             eval_every=EVAL_EVERY, eval_episodes=EVAL_EPISODES,
                             log_stdout=False, seed=seed)

    run_config = {
        'env': 'LunarLander-v3 (discrete)', 'agent': 'Discrete SAC (fixed)',
        'seed': seed, 'total_steps': TOTAL_STEPS,
        'initial_random_steps': cfg.start_steps,
        'target_entropy_ratio': cfg.target_entropy_ratio,
        'target_entropy': float(agent.target_entropy),
        'autotune_alpha': cfg.autotune_alpha,
        'init_alpha': cfg.init_alpha,
        'eval_every': EVAL_EVERY, 'eval_episodes': EVAL_EPISODES,
    }
    save_log(log, p, config=run_config)
    print(f"  disc_sac_fixed seed={seed}: {time.time()-t0:.1f}s")

print(f"\nDone. Total: {time.time()-t0_total:.1f}s")
print(f"Results in: {LOG_DIR}/")
