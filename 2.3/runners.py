"""Training entry points for PA3 Section 2.3: Reacher-easy with Ra/Rb/Rc."""
from __future__ import annotations

import os

import torch

from sac_core import set_global_seed, save_log, get_device
from sac_agent import SACAgent, SACConfig, train_sac
from reacher_env import make_reacher

LOG_DIR = "logs"
EVAL_EPISODES = 20
REWARDS = ("Ra", "Rb", "Rc")


def run_reacher(seed: int, reward_name: str, tag: str, *,
                total_steps: int, eval_every: int, device=None):
    """Train SAC on one reward formulation and evaluate on all three rewards.

    This directly supports the assignment requirement:
    when training SAC-Ra/SAC-Rb/SAC-Rc, log returns under Ra, Rb, and Rc.
    """
    assert reward_name in REWARDS
    set_global_seed(seed)

    train_env_fn = lambda: make_reacher(reward_name, mode="train")

    # True cross-evaluation: separate evaluation env for each reward formulation.
    eval_env_fns = {
        "Ra_eval": lambda: make_reacher("Ra", mode="eval"),
        "Rb_eval": lambda: make_reacher("Rb", mode="eval"),
        "Rc_eval": lambda: make_reacher("Rc", mode="eval"),
    }
    primary_eval_key = f"{reward_name}_eval"

    probe_env = train_env_fn()
    obs_dim = probe_env.observation_space.shape[0]
    act_dim = probe_env.action_space.shape[0]
    act_limit = float(probe_env.action_space.high[0])
    probe_env.close()

    cfg = SACConfig(start_steps=10_000, update_after=10_000)
    agent = SACAgent(obs_dim, act_dim, act_limit, cfg, device=device or get_device())

    log = train_sac(
        train_env_fn,
        agent,
        total_steps=total_steps,
        eval_every=eval_every,
        eval_episodes=EVAL_EPISODES,
        eval_env_fns=eval_env_fns,
        primary_eval_key=primary_eval_key,
        log_stdout=False,
        seed=seed,
    )

    run_config = {
        "env": f"DMC reacher-easy ({reward_name})",
        "reward_formulation": reward_name,
        "seed": seed,
        "total_steps": total_steps,
        "initial_random_steps": cfg.start_steps,
        "update_after": cfg.update_after,
        "replay_buffer_size": cfg.buffer_size,
        "replay_buffer_type": "fixed-size circular (1M)",
        "batch_size": cfg.batch_size,
        "gamma": cfg.gamma,
        "tau": cfg.tau,
        "actor_lr": cfg.actor_lr,
        "critic_lr": cfg.critic_lr,
        "alpha_lr": cfg.alpha_lr,
        "autotune_alpha": cfg.autotune_alpha,
        "init_alpha": cfg.init_alpha,
        "target_entropy": "-action_dim",
        "eval_every": eval_every,
        "eval_episodes": EVAL_EPISODES,
        "cross_eval_keys": ["Ra_eval", "Rb_eval", "Rc_eval"],
        "rc_timeout_steps": 1000,
        "rc_timeout_penalty": -20.0,
        "rc_goal_velocity_threshold": 0.5,
        "final_diagnostic_episode_length": 5000,
    }

    os.makedirs(LOG_DIR, exist_ok=True)
    save_log(log, os.path.join(LOG_DIR, f"{tag}_seed{seed}.json"), config=run_config)
    torch.save(
        {
            "actor": agent.actor.state_dict(),
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "act_limit": act_limit,
            "hidden": cfg.hidden,
            "reward_formulation": reward_name,
            "seed": seed,
        },
        os.path.join(LOG_DIR, f"{tag}_seed{seed}_actor.pt"),
    )
