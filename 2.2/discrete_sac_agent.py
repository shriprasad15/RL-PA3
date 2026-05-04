"""Discrete Soft Actor-Critic for PA3 2.2 LunarLander.

Categorical actor + twin Q networks over discrete actions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from sac_core import (
    ReplayBuffer,
    CategoricalActor,
    TwinDiscreteQ,
    soft_update,
    get_device,
    EvalLog,
    evaluate_policy,
)


@dataclass
class DiscreteSACConfig:
    hidden: tuple = (256, 256)
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    buffer_size: int = 1_000_000
    start_steps: int = 10_000
    update_after: int = 10_000
    update_every: int = 1
    grad_steps_per_update: int = 1
    autotune_alpha: bool = True
    init_alpha: float = 0.2
    target_entropy_ratio: float = 0.98


class DiscreteSACAgent:
    def __init__(self, obs_dim: int, n_actions: int, config: DiscreteSACConfig,
                 device: torch.device | None = None):
        self.cfg = config
        self.device = device if device is not None else get_device()
        self.obs_dim = obs_dim
        self.n_actions = n_actions

        self.actor = CategoricalActor(obs_dim, n_actions, config.hidden).to(self.device)
        self.critic = TwinDiscreteQ(obs_dim, n_actions, config.hidden).to(self.device)
        self.critic_target = TwinDiscreteQ(obs_dim, n_actions, config.hidden).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        if config.autotune_alpha:
            self.target_entropy = float(config.target_entropy_ratio) * np.log(n_actions)
            self.log_alpha = torch.tensor(np.log(config.init_alpha), device=self.device, requires_grad=True)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=config.alpha_lr)
        else:
            self.target_entropy = None
            self.log_alpha = torch.tensor(np.log(config.init_alpha), device=self.device)
            self.alpha_opt = None

        self.buffer = ReplayBuffer(config.buffer_size, obs_dim, act_dim=1, discrete=True)
        self.total_env_steps = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        probs, _ = self.actor(o)
        if deterministic:
            return int(probs.argmax(dim=-1).item())
        dist = torch.distributions.Categorical(probs=probs)
        return int(dist.sample().item())

    def random_action(self, rng: np.random.Generator) -> int:
        return int(rng.integers(0, self.n_actions))

    def update(self) -> dict:
        batch = self.buffer.sample(self.cfg.batch_size, self.device)
        o = batch["obs"].float()
        a = batch["acts"].long()
        r = batch["rews"].float()
        o2 = batch["next_obs"].float()
        d = batch["dones"].float()

        with torch.no_grad():
            probs_next, logp_next = self.actor(o2)
            q1_t, q2_t = self.critic_target(o2)
            q_min = torch.min(q1_t, q2_t) - self.alpha.detach() * logp_next
            v_next = (probs_next * q_min).sum(dim=-1)
            y = r + self.cfg.gamma * (1.0 - d) * v_next

        q1_all, q2_all = self.critic(o)
        q1 = q1_all.gather(1, a.view(-1, 1)).squeeze(-1)
        q2 = q2_all.gather(1, a.view(-1, 1)).squeeze(-1)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        for p in self.critic.parameters():
            p.requires_grad_(False)
        probs, logp = self.actor(o)
        with torch.no_grad():
            q1_now, q2_now = self.critic(o)
            q_now = torch.min(q1_now, q2_now)
        actor_loss = (probs * (self.alpha.detach() * logp - q_now)).sum(dim=-1).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        for p in self.critic.parameters():
            p.requires_grad_(True)

        info = {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.alpha.detach().cpu()),
        }

        if self.cfg.autotune_alpha:
            with torch.no_grad():
                entropy = -(probs * logp).sum(dim=-1)
            alpha_loss = -(self.log_alpha * (self.target_entropy - entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            info["alpha_loss"] = float(alpha_loss.item())
            info["entropy"] = float(entropy.mean().cpu())

        soft_update(self.critic_target, self.critic, self.cfg.tau)
        return info

    def checkpoint(self) -> dict:
        return {
            "agent_type": "discrete_sac",
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_opt": self.alpha_opt.state_dict() if self.alpha_opt is not None else None,
            "config": self.cfg.__dict__,
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
            "total_env_steps": self.total_env_steps,
        }


def train_discrete_sac(env_fn: Callable, agent: DiscreteSACAgent, *,
                       total_steps: int,
                       eval_env_fn: Callable,
                       eval_every: int = 10_000,
                       eval_episodes: int = 20,
                       log_stdout: bool = True,
                       seed: int = 0,
                       on_eval: Callable | None = None,
                       train_log_every: int = 1000):
    rng = np.random.default_rng(seed)
    env = env_fn()
    obs, _ = env.reset(seed=seed)

    eval_log = EvalLog()
    train_rows: list[dict] = []
    ep_ret, ep_len = 0.0, 0

    def _eval_now(step: int):
        out = evaluate_policy(
            eval_env_fn,
            act_fn=lambda o: agent.act(o, deterministic=True),
            n_episodes=eval_episodes,
        )
        eval_log.append(step, out["return"], std_ret=out.get("return_std"))
        if on_eval is not None:
            on_eval(step, out, agent)
        if log_stdout:
            print(f"[step {step:>7}] eval_return={out['return']:.2f}")

    _eval_now(0)
    last_update_info = {}

    for t in range(1, total_steps + 1):
        a = agent.random_action(rng) if t <= agent.cfg.start_steps else agent.act(obs, deterministic=False)
        next_obs, r, term, trunc, _ = env.step(a)
        agent.buffer.add(obs, a, r, next_obs, float(term))
        obs = next_obs
        ep_ret += float(r)
        ep_len += 1
        agent.total_env_steps = t

        if term or trunc:
            train_rows.append({
                "global_step": t,
                "event": "episode_end",
                "episode_return": ep_ret,
                "episode_length": ep_len,
                "alpha": float(agent.alpha.detach().cpu()),
            })
            obs, _ = env.reset()
            ep_ret, ep_len = 0.0, 0

        if t >= agent.cfg.update_after and t % agent.cfg.update_every == 0:
            for _ in range(agent.cfg.grad_steps_per_update):
                last_update_info = agent.update()

        if t % train_log_every == 0:
            row = {
                "global_step": t,
                "event": "train_step",
                "replay_size": agent.buffer.size,
                "alpha": float(agent.alpha.detach().cpu()),
            }
            row.update(last_update_info)
            train_rows.append(row)

        if t % eval_every == 0:
            _eval_now(t)

    env.close()
    return eval_log, train_rows
