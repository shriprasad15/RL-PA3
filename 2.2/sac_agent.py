"""Continuous-action Soft Actor-Critic for PA3 2.2 LunarLander.

Assignment-aligned details:
- squashed Gaussian tanh policy
- twin Q critics with clipped double-Q target
- reparameterization trick
- automatic alpha tuning or fixed alpha
- no separate state-value network
- 10K random-action warmup before updates begin
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

from sac_core import (
    ReplayBuffer,
    SquashedGaussianActor,
    TwinQCritic,
    soft_update,
    get_device,
    EvalLog,
    evaluate_policy,
)


@dataclass
class SACConfig:
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
    target_entropy: Optional[float] = None

    reward_scale: float = 1.0


class SACAgent:
    def __init__(self, obs_dim: int, act_dim: int, act_limit: float,
                 config: SACConfig, device: torch.device | None = None):
        self.cfg = config
        self.device = device if device is not None else get_device()
        self.obs_dim, self.act_dim, self.act_limit = obs_dim, act_dim, float(act_limit)

        self.actor = SquashedGaussianActor(obs_dim, act_dim, self.act_limit, config.hidden).to(self.device)
        self.critic = TwinQCritic(obs_dim, act_dim, config.hidden).to(self.device)
        self.critic_target = TwinQCritic(obs_dim, act_dim, config.hidden).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        if config.autotune_alpha:
            self.target_entropy = -float(act_dim) if config.target_entropy is None else float(config.target_entropy)
            self.log_alpha = torch.tensor(np.log(config.init_alpha), device=self.device, requires_grad=True)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=config.alpha_lr)
        else:
            self.target_entropy = None
            self.log_alpha = torch.tensor(np.log(config.init_alpha), device=self.device)
            self.alpha_opt = None

        self.buffer = ReplayBuffer(config.buffer_size, obs_dim, act_dim, discrete=False)
        self.total_env_steps = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        a, _ = self.actor(o, deterministic=deterministic, with_logprob=False)
        return a.cpu().numpy()[0]

    def random_action(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(-self.act_limit, self.act_limit, size=self.act_dim).astype(np.float32)

    def update(self) -> dict:
        batch = self.buffer.sample(self.cfg.batch_size, self.device)
        o = batch["obs"].float()
        a = batch["acts"].float()
        r = batch["rews"].float() * self.cfg.reward_scale
        o2 = batch["next_obs"].float()
        d = batch["dones"].float()

        # Critic target
        with torch.no_grad():
            a2, logp2 = self.actor(o2)
            q1_t, q2_t = self.critic_target(o2, a2)
            q_t = torch.min(q1_t, q2_t) - self.alpha.detach() * logp2
            y = r + self.cfg.gamma * (1.0 - d) * q_t

        q1, q2 = self.critic(o, a)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Actor update
        for p in self.critic.parameters():
            p.requires_grad_(False)
        a_pi, logp_pi = self.actor(o)
        q1_pi, q2_pi = self.critic(o, a_pi)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha.detach() * logp_pi - q_pi).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        for p in self.critic.parameters():
            p.requires_grad_(True)

        info = {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.alpha.detach().cpu()),
            "logp": float(logp_pi.detach().mean().cpu()),
        }

        if self.cfg.autotune_alpha:
            alpha_loss = -(self.log_alpha * (logp_pi.detach() + self.target_entropy)).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            info["alpha_loss"] = float(alpha_loss.item())

        soft_update(self.critic_target, self.critic, self.cfg.tau)
        return info

    def checkpoint(self) -> dict:
        return {
            "agent_type": "continuous_sac",
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_opt": self.alpha_opt.state_dict() if self.alpha_opt is not None else None,
            "config": self.cfg.__dict__,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "act_limit": self.act_limit,
            "total_env_steps": self.total_env_steps,
        }


def train_sac(env_fn: Callable, agent: SACAgent, *,
              total_steps: int,
              eval_env_fn: Callable,
              eval_every: int = 10_000,
              eval_episodes: int = 20,
              eval_extra_reward_fns: dict | None = None,
              log_stdout: bool = True,
              seed: int = 0,
              on_eval: Callable | None = None,
              train_log_every: int = 1000):
    """Train SAC and return (eval_log, train_rows)."""
    rng = np.random.default_rng(seed)
    env = env_fn()
    obs, _ = env.reset(seed=seed)
    eval_log = EvalLog()
    train_rows: list[dict] = []
    ep_ret, ep_len = 0.0, 0

    def _eval_now(step: int):
        eval_out = evaluate_policy(
            eval_env_fn,
            act_fn=lambda o: agent.act(o, deterministic=True),
            n_episodes=eval_episodes,
            extra_reward_fns=eval_extra_reward_fns,
        )
        extras = {k: v for k, v in eval_out.items()
                  if k not in ("return", "return_std", "episode_returns")}
        eval_log.append(step, eval_out["return"], extras, std_ret=eval_out.get("return_std"))
        if on_eval is not None:
            on_eval(step, eval_out, agent)
        if log_stdout:
            print(f"[step {step:>7}] eval_return={eval_out['return']:.2f}")

    _eval_now(0)

    last_update_info = {}
    for t in range(1, total_steps + 1):
        a = agent.random_action(rng) if t <= agent.cfg.start_steps else agent.act(obs, deterministic=False)
        next_obs, r, term, trunc, _info = env.step(a)
        # Important: bootstrap done only on true termination, not time-limit truncation.
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
