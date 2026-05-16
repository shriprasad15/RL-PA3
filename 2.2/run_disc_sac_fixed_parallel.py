"""Parallel run of Discrete SAC with target_entropy_ratio=0.98 (fix alpha explosion).
4 seeds, 4 workers, all parallel.
"""
import os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(__file__))

N_SEEDS = 4
TOTAL_STEPS = 300_000
EVAL_EVERY = 10_000
EVAL_EPISODES = 20
LOG_DIR = "logs_disc_fixed"
os.makedirs(LOG_DIR, exist_ok=True)

def run_one(seed):
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from sac_core import set_global_seed, get_device, save_log
    from discrete_sac_agent import DiscreteSACAgent, DiscreteSACConfig, train_discrete_sac
    from lunar_env import make_lunar_discrete

    device = get_device()
    p = os.path.join(LOG_DIR, f"disc_sac_fixed_seed{seed}.json")
    if os.path.exists(p) and os.path.getsize(p) > 100:
        return f"seed {seed}: skip"

    t0 = time.time()
    set_global_seed(seed)
    env_fn = lambda: make_lunar_discrete()
    e = env_fn()
    obs_dim = e.observation_space.shape[0]
    n_actions = e.action_space.n
    e.close()

    cfg = DiscreteSACConfig(start_steps=10_000, update_after=1_000, target_entropy_ratio=0.98)
    agent = DiscreteSACAgent(obs_dim, n_actions, cfg, device=device)
    log, train_rows = train_discrete_sac(env_fn, agent, total_steps=TOTAL_STEPS, eval_env_fn=env_fn,
                             eval_every=EVAL_EVERY, eval_episodes=EVAL_EPISODES,
                             log_stdout=False, seed=seed)
    run_config = {
        'env': 'LunarLander-v3 (discrete)', 'agent': 'Discrete SAC (fixed)',
        'seed': seed, 'total_steps': TOTAL_STEPS,
        'target_entropy_ratio': cfg.target_entropy_ratio,
        'target_entropy': float(agent.target_entropy),
    }
    save_log(log, p, config=run_config)
    return f"seed {seed}: {time.time()-t0:.1f}s"

if __name__ == "__main__":
    print(f"Running {N_SEEDS} seeds in parallel (target_entropy_ratio=0.98)")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_SEEDS) as ex:
        futs = {ex.submit(run_one, s): s for s in range(N_SEEDS)}
        for f in as_completed(futs):
            print(f"  {f.result()}")
    print(f"Done. Total: {time.time()-t0:.1f}s")
