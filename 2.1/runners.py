"""2.1 training entry points — called by the notebook AND by run_experiment.py."""
from __future__ import annotations

import os

import torch

from sac_core import set_global_seed, save_log, get_device
from sac_agent import SACAgent, SACConfig, train_sac
from pendulum_env import make_pendulum

LOG_DIR = "logs"
EVAL_EPISODES = 20


def run_one(theta_target: float, seed: int, tag: str, *,
             total_steps: int, eval_every: int,
             reward_scale: float = 1.0,
             autotune: bool = True, init_alpha: float = 0.2,
             device=None):
    """Train one SAC agent on the target-angle Pendulum and save (log, config)."""
    set_global_seed(seed)
    env_fn = lambda: make_pendulum(theta_target, reward_scale=reward_scale)
    s = env_fn(); obs_dim = s.observation_space.shape[0]; act_dim = s.action_space.shape[0]
    act_limit = float(s.action_space.high[0]); s.close()

    cfg = SACConfig(
        autotune_alpha=autotune,
        init_alpha=init_alpha,
        reward_scale=1.0,            # scaling applied in env
        start_steps=500,             # TA rule for Pendulum
        update_after=500,
    )
    agent = SACAgent(obs_dim, act_dim, act_limit, cfg, device=device or get_device())
    log = train_sac(env_fn, agent, total_steps=total_steps, eval_env_fn=env_fn,
                    eval_every=eval_every, eval_episodes=EVAL_EPISODES,
                    log_stdout=False, seed=seed)

    run_config = {
        "env": "Pendulum-v1-target-angle", "theta_target_deg": theta_target,
        "reward_scale_env": reward_scale, "seed": seed,
        "total_steps": total_steps, "initial_random_steps": cfg.start_steps,
        "replay_buffer_size": cfg.buffer_size,
        "replay_buffer_type": "infinite-equivalent (1M)",
        "batch_size": cfg.batch_size, "gamma": cfg.gamma, "tau": cfg.tau,
        "actor_lr": cfg.actor_lr, "critic_lr": cfg.critic_lr, "alpha_lr": cfg.alpha_lr,
        "autotune_alpha": cfg.autotune_alpha, "init_alpha": cfg.init_alpha,
        "eval_every": eval_every, "eval_episodes": EVAL_EPISODES,
    }
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f"{tag}_seed{seed}.json")
    save_log(log, path, config=run_config)

    # Save final actor + critic checkpoint so the policy can be re-loaded for
    # inspection / render / additional eval episodes later.
    ckpt_path = os.path.join(LOG_DIR, f"{tag}_seed{seed}_checkpoint.pt")
    torch.save({
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
        "critic_target": agent.critic_target.state_dict(),
        "log_alpha": agent.log_alpha.detach().cpu(),
        "obs_dim": obs_dim, "act_dim": act_dim, "act_limit": act_limit,
    }, ckpt_path)
    return path
