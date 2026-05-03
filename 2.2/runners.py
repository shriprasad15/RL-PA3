"""2.2 training entry points."""
from __future__ import annotations

import os

import numpy as np
import torch

from sac_core import (set_global_seed, save_log, EvalLog, evaluate_policy, get_device)
from sac_agent import SACAgent, SACConfig, train_sac
from discrete_sac_agent import DiscreteSACAgent, DiscreteSACConfig, train_discrete_sac
from dqn_agent import DQNAgent, DQNConfig, train_dqn
from lunar_env import make_lunar_continuous, make_lunar_discrete

LOG_DIR = "logs"
EVAL_EPISODES = 20


def run_continuous(seed: int, tag: str, *, total_steps: int, eval_every: int,
                    hover_bonus=None, buffer_size=1_000_000,
                    buffer_type="infinite-equivalent (1M)",
                    autotune=True, init_alpha=0.2, device=None):
    set_global_seed(seed)
    env_fn = lambda: make_lunar_continuous(hover_bonus=hover_bonus)
    s = env_fn(); obs_dim = s.observation_space.shape[0]; act_dim = s.action_space.shape[0]
    lim = float(s.action_space.high[0]); s.close()
    cfg = SACConfig(start_steps=10_000, update_after=1_000, autotune_alpha=autotune,
                     init_alpha=init_alpha, buffer_size=buffer_size)
    agent = SACAgent(obs_dim, act_dim, lim, cfg, device=device or get_device())
    log = train_sac(env_fn, agent, total_steps=total_steps, eval_env_fn=env_fn,
                    eval_every=eval_every, eval_episodes=EVAL_EPISODES,
                    log_stdout=False, seed=seed)
    run_config = {
        "env": "LunarLander-v3 (continuous)" + (" + hover bonus" if hover_bonus is not None else ""),
        "hover_bonus": hover_bonus, "seed": seed, "total_steps": total_steps,
        "initial_random_steps": cfg.start_steps,
        "replay_buffer_size": cfg.buffer_size, "replay_buffer_type": buffer_type,
        "batch_size": cfg.batch_size, "gamma": cfg.gamma, "tau": cfg.tau,
        "actor_lr": cfg.actor_lr, "critic_lr": cfg.critic_lr, "alpha_lr": cfg.alpha_lr,
        "autotune_alpha": cfg.autotune_alpha, "init_alpha": cfg.init_alpha,
        "eval_every": eval_every, "eval_episodes": EVAL_EPISODES,
    }
    os.makedirs(LOG_DIR, exist_ok=True)
    save_log(log, os.path.join(LOG_DIR, f"{tag}_seed{seed}.json"), config=run_config)
    torch.save({
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
        "critic_target": agent.critic_target.state_dict(),
        "log_alpha": agent.log_alpha.detach().cpu(),
        "obs_dim": obs_dim, "act_dim": act_dim, "act_limit": lim,
    }, os.path.join(LOG_DIR, f"{tag}_seed{seed}_checkpoint.pt"))


def _hover_hits_fn(env, obs, action, r, info):
    return 1.0 if info.get("hover_bonus_given", False) else 0.0


def run_hover_switch(seed: int, tag: str, *, autotune: bool, init_alpha: float,
                      steps_pre: int, steps_post: int, eval_every: int,
                      buffer_size: int = 100_000, device=None):
    set_global_seed(seed)
    live_env = make_lunar_continuous(hover_bonus=+200.0)
    current_bonus = {"v": 200.0}
    eval_env_fn = lambda: make_lunar_continuous(hover_bonus=current_bonus["v"])

    obs_dim = live_env.observation_space.shape[0]
    act_dim = live_env.action_space.shape[0]
    act_limit = float(live_env.action_space.high[0])

    cfg = SACConfig(autotune_alpha=autotune, init_alpha=init_alpha,
                    start_steps=10_000, update_after=1_000, buffer_size=buffer_size)
    agent = SACAgent(obs_dim, act_dim, act_limit, cfg, device=device or get_device())

    log = EvalLog()
    rng = np.random.default_rng(seed)
    obs, _ = live_env.reset(seed=seed)

    def _eval_now(step):
        out = evaluate_policy(eval_env_fn,
                              act_fn=lambda o: agent.act(o, deterministic=True),
                              n_episodes=EVAL_EPISODES,
                              extra_reward_fns={"hover_hits": _hover_hits_fn})
        log.append(step, out["return"], {"hover_hits": out["hover_hits"]})

    _eval_now(0)
    total = steps_pre + steps_post
    flipped = False
    for t in range(1, total + 1):
        if (not flipped) and t == steps_pre + 1:
            live_env.set_hover_bonus(-100.0)
            current_bonus["v"] = -100.0
            flipped = True
        a = agent.random_action(rng) if t <= cfg.start_steps else agent.act(obs)
        next_obs, r, term, trunc, _ = live_env.step(a)
        agent.buffer.add(obs, a, r, next_obs, float(term))
        obs = next_obs
        agent.total_env_steps = t
        if term or trunc:
            obs, _ = live_env.reset()
        if t >= cfg.update_after:
            agent.update()
        if t % eval_every == 0:
            _eval_now(t)
    live_env.close()

    run_config = {
        "env": "LunarLander-v3 (continuous) + hover box, reward switch +200->-100",
        "flip_step": steps_pre, "seed": seed,
        "total_steps": total, "initial_random_steps": cfg.start_steps,
        "replay_buffer_size": cfg.buffer_size,
        "replay_buffer_type": "fixed-size (TA rule)",
        "batch_size": cfg.batch_size, "gamma": cfg.gamma, "tau": cfg.tau,
        "actor_lr": cfg.actor_lr, "critic_lr": cfg.critic_lr, "alpha_lr": cfg.alpha_lr,
        "autotune_alpha": cfg.autotune_alpha, "init_alpha": cfg.init_alpha,
        "eval_every": eval_every, "eval_episodes": EVAL_EPISODES,
    }
    os.makedirs(LOG_DIR, exist_ok=True)
    save_log(log, os.path.join(LOG_DIR, f"{tag}_seed{seed}.json"), config=run_config)
    torch.save({
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
        "critic_target": agent.critic_target.state_dict(),
        "log_alpha": agent.log_alpha.detach().cpu(),
        "obs_dim": obs_dim, "act_dim": act_dim, "act_limit": act_limit,
    }, os.path.join(LOG_DIR, f"{tag}_seed{seed}_checkpoint.pt"))


def run_disc_sac(seed: int, tag: str, *, total_steps: int, eval_every: int, device=None):
    set_global_seed(seed)
    env_fn = lambda: make_lunar_discrete()
    e = env_fn(); obs_dim = e.observation_space.shape[0]; n_actions = e.action_space.n; e.close()
    cfg = DiscreteSACConfig(start_steps=10_000, update_after=1_000)
    agent = DiscreteSACAgent(obs_dim, n_actions, cfg, device=device or get_device())
    log = train_discrete_sac(env_fn, agent, total_steps=total_steps, eval_env_fn=env_fn,
                             eval_every=eval_every, eval_episodes=EVAL_EPISODES,
                             log_stdout=False, seed=seed)
    run_config = {
        "env": "LunarLander-v3 (discrete, default reward)", "seed": seed,
        "total_steps": total_steps, "initial_random_steps": cfg.start_steps,
        "replay_buffer_size": cfg.buffer_size,
        "replay_buffer_type": "infinite-equivalent (1M)",
        "batch_size": cfg.batch_size, "gamma": cfg.gamma, "tau": cfg.tau,
        "actor_lr": cfg.actor_lr, "critic_lr": cfg.critic_lr, "alpha_lr": cfg.alpha_lr,
        "autotune_alpha": cfg.autotune_alpha, "init_alpha": cfg.init_alpha,
        "target_entropy_ratio": cfg.target_entropy_ratio,
        "eval_every": eval_every, "eval_episodes": EVAL_EPISODES,
    }
    os.makedirs(LOG_DIR, exist_ok=True)
    save_log(log, os.path.join(LOG_DIR, f"{tag}_seed{seed}.json"), config=run_config)
    torch.save({
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
        "critic_target": agent.critic_target.state_dict(),
        "log_alpha": agent.log_alpha.detach().cpu(),
        "obs_dim": obs_dim, "n_actions": n_actions,
    }, os.path.join(LOG_DIR, f"{tag}_seed{seed}_checkpoint.pt"))


def run_dqn(seed: int, tag: str, *, total_steps: int, eval_every: int, device=None):
    set_global_seed(seed)
    env_fn = lambda: make_lunar_discrete()
    e = env_fn(); obs_dim = e.observation_space.shape[0]; n_actions = e.action_space.n; e.close()
    cfg = DQNConfig(start_steps=10_000, update_after=1_000, grad_steps_per_update=1)
    agent = DQNAgent(obs_dim, n_actions, cfg, device=device or get_device())
    log = train_dqn(env_fn, agent, total_steps=total_steps, eval_env_fn=env_fn,
                    eval_every=eval_every, eval_episodes=EVAL_EPISODES,
                    log_stdout=False, seed=seed)
    run_config = {
        "env": "LunarLander-v3 (discrete, default reward)", "agent": "DQN",
        "seed": seed, "total_steps": total_steps,
        "initial_random_steps": cfg.start_steps,
        "replay_buffer_size": cfg.buffer_size,
        "replay_buffer_type": "infinite-equivalent (1M)",
        "batch_size": cfg.batch_size, "gamma": cfg.gamma, "lr": cfg.lr,
        "target_update_every": cfg.target_update_every,
        "replay_factor": cfg.grad_steps_per_update,
        "epsilon_start": cfg.epsilon_start, "epsilon_end": cfg.epsilon_end,
        "epsilon_decay_steps": cfg.epsilon_decay_steps,
        "eval_every": eval_every, "eval_episodes": EVAL_EPISODES,
    }
    os.makedirs(LOG_DIR, exist_ok=True)
    save_log(log, os.path.join(LOG_DIR, f"{tag}_seed{seed}.json"), config=run_config)
    torch.save({
        "q": agent.q.state_dict(), "q_target": agent.q_target.state_dict(),
        "obs_dim": obs_dim, "n_actions": n_actions,
    }, os.path.join(LOG_DIR, f"{tag}_seed{seed}_checkpoint.pt"))
