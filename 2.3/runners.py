"""2.3 training entry points (Reacher-easy with Ra/Rb/Rc)."""
from __future__ import annotations

import os

import torch

from sac_core import set_global_seed, save_log, get_device
from sac_agent import SACAgent, SACConfig, train_sac
from reacher_env import make_reacher, CROSS_EVAL_FNS

LOG_DIR = "logs"
EVAL_EPISODES = 20


def run_reacher(seed: int, reward_name: str, tag: str, *,
                 total_steps: int, eval_every: int, device=None):
    assert reward_name in ("Ra", "Rb", "Rc")
    set_global_seed(seed)
    train_env_fn = lambda: make_reacher(reward_name, mode="train")
    eval_env_fn = lambda: make_reacher(reward_name, mode="eval")
    s = train_env_fn(); obs_dim = s.observation_space.shape[0]; act_dim = s.action_space.shape[0]
    act_limit = float(s.action_space.high[0]); s.close()

    cfg = SACConfig(start_steps=10_000, update_after=1_000)
    agent = SACAgent(obs_dim, act_dim, act_limit, cfg, device=device or get_device())
    log = train_sac(train_env_fn, agent, total_steps=total_steps, eval_env_fn=eval_env_fn,
                    eval_every=eval_every, eval_episodes=EVAL_EPISODES,
                    eval_extra_reward_fns=CROSS_EVAL_FNS,
                    log_stdout=False, seed=seed)

    run_config = {
        "env": f"DMC reacher-easy ({reward_name})",
        "reward_formulation": reward_name, "seed": seed,
        "total_steps": total_steps, "initial_random_steps": cfg.start_steps,
        "replay_buffer_size": cfg.buffer_size,
        "replay_buffer_type": "infinite-equivalent (1M)",
        "batch_size": cfg.batch_size, "gamma": cfg.gamma, "tau": cfg.tau,
        "actor_lr": cfg.actor_lr, "critic_lr": cfg.critic_lr, "alpha_lr": cfg.alpha_lr,
        "autotune_alpha": cfg.autotune_alpha, "init_alpha": cfg.init_alpha,
        "eval_every": eval_every, "eval_episodes": EVAL_EPISODES,
        "rc_timeout_steps": 1000, "rc_timeout_penalty": -20.0,
        "rc_goal_velocity_threshold": 0.5, "rc_eval_timeout_return": -1020,
    }
    os.makedirs(LOG_DIR, exist_ok=True)
    save_log(log, os.path.join(LOG_DIR, f"{tag}_seed{seed}.json"), config=run_config)
    torch.save(
        {"actor": agent.actor.state_dict(),
         "obs_dim": obs_dim, "act_dim": act_dim, "act_limit": act_limit},
        os.path.join(LOG_DIR, f"{tag}_seed{seed}_actor.pt"),
    )
