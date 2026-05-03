"""Bonus-section training entry points (SAC-GT Pendulum and PEBBLE)."""
from __future__ import annotations

import os

from sac_core import set_global_seed, save_log, get_device
from sac_agent import SACAgent, SACConfig, train_sac
from pebble import PebbleAgent, PebbleConfig, train_pebble
from pendulum_env import make_pendulum
from reacher_env import make_reacher

LOG_DIR = "logs"
EVAL_EPISODES = 20


def run_sac_gt_pendulum(theta: int, seed: int, tag: str, *,
                         total_steps: int, eval_every: int, device=None):
    set_global_seed(seed)
    env_fn = lambda: make_pendulum(theta)
    s = env_fn(); obs_dim = s.observation_space.shape[0]; act_dim = s.action_space.shape[0]
    lim = float(s.action_space.high[0]); s.close()
    cfg = SACConfig(start_steps=500, update_after=500)  # TA rule for Pendulum
    agent = SACAgent(obs_dim, act_dim, lim, cfg, device=device or get_device())
    log = train_sac(env_fn, agent, total_steps=total_steps, eval_env_fn=env_fn,
                    eval_every=eval_every, eval_episodes=EVAL_EPISODES,
                    log_stdout=False, seed=seed)
    run_config = {
        "env": f"Pendulum theta={theta}", "theta_target_deg": theta,
        "algo": "SAC (ground-truth reward)", "seed": seed,
        "total_steps": total_steps, "initial_random_steps": cfg.start_steps,
        "replay_buffer_size": cfg.buffer_size,
        "replay_buffer_type": "infinite-equivalent (1M)",
        "eval_every": eval_every, "eval_episodes": EVAL_EPISODES,
    }
    os.makedirs(LOG_DIR, exist_ok=True)
    save_log(log, os.path.join(LOG_DIR, f"{tag}_seed{seed}.json"), config=run_config)


def run_pebble_pendulum(theta: int, seed: int, tag: str, *,
                         total_steps: int, eval_every: int,
                         budget: int, device=None):
    set_global_seed(seed)
    env_fn = lambda: make_pendulum(theta)
    s = env_fn(); obs_dim = s.observation_space.shape[0]; act_dim = s.action_space.shape[0]
    lim = float(s.action_space.high[0]); s.close()
    cfg = PebbleConfig(total_feedback=budget, unsup_steps=5_000,
                       start_steps=500, update_after=500)
    agent = PebbleAgent(obs_dim, act_dim, lim, cfg, device=device or get_device())
    log = train_pebble(env_fn, agent, total_steps=total_steps, eval_env_fn=env_fn,
                       eval_every=eval_every, eval_episodes=EVAL_EPISODES,
                       log_stdout=False, seed=seed)
    run_config = {
        "env": f"Pendulum theta={theta}", "theta_target_deg": theta,
        "algo": "PEBBLE", "seed": seed, "feedback_budget": budget,
        "total_steps": total_steps, "initial_random_steps": cfg.start_steps,
        "unsup_steps": cfg.unsup_steps, "segment_len": cfg.segment_len,
        "n_init_queries": cfg.n_init_queries,
        "n_queries_per_batch": cfg.n_queries_per_batch,
        "query_every": cfg.query_every,
        "reward_epochs_init": cfg.reward_epochs_init,
        "reward_epochs_per_batch": cfg.reward_epochs_per_batch,
        "reward_ensemble_size": cfg.n_ensemble, "reward_lr": cfg.reward_lr,
        "replay_buffer_size": cfg.buffer_size,
        "replay_buffer_type": "infinite-equivalent (500K)",
        "eval_every": eval_every, "eval_episodes": EVAL_EPISODES,
    }
    os.makedirs(LOG_DIR, exist_ok=True)
    save_log(log, os.path.join(LOG_DIR, f"{tag}_seed{seed}.json"), config=run_config)


def run_pebble_reacher(reward_name: str, seed: int, tag: str, *,
                        total_steps: int, eval_every: int,
                        budget: int, device=None):
    assert reward_name in ("Ra", "Rb", "Rc")
    set_global_seed(seed)
    train_env_fn = lambda: make_reacher(reward_name, mode="train")
    eval_env_fn = lambda: make_reacher(reward_name, mode="eval")
    s = train_env_fn(); obs_dim = s.observation_space.shape[0]; act_dim = s.action_space.shape[0]
    lim = float(s.action_space.high[0]); s.close()
    cfg = PebbleConfig(total_feedback=budget, unsup_steps=9_000,
                       start_steps=10_000, update_after=1_000,
                       n_init_queries=200, n_queries_per_batch=64,
                       query_every=10_000)
    agent = PebbleAgent(obs_dim, act_dim, lim, cfg, device=device or get_device())
    log = train_pebble(train_env_fn, agent, total_steps=total_steps,
                       eval_env_fn=eval_env_fn,
                       eval_every=eval_every, eval_episodes=EVAL_EPISODES,
                       log_stdout=False, seed=seed)
    run_config = {
        "env": f"DMC reacher-easy ({reward_name})",
        "reward_formulation": reward_name, "algo": "PEBBLE",
        "seed": seed, "feedback_budget": budget,
        "total_steps": total_steps, "initial_random_steps": cfg.start_steps,
        "unsup_steps": cfg.unsup_steps, "segment_len": cfg.segment_len,
        "n_init_queries": cfg.n_init_queries,
        "n_queries_per_batch": cfg.n_queries_per_batch,
        "query_every": cfg.query_every,
        "reward_ensemble_size": cfg.n_ensemble, "reward_lr": cfg.reward_lr,
        "eval_every": eval_every, "eval_episodes": EVAL_EPISODES,
        "rc_eval_timeout_return": -1020,
    }
    os.makedirs(LOG_DIR, exist_ok=True)
    save_log(log, os.path.join(LOG_DIR, f"{tag}_seed{seed}.json"), config=run_config)
