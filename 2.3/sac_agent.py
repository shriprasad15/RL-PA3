"""Continuous-action Soft Actor-Critic for PA3 Section 2.3 Reacher.

Implements the assignment SAC requirements:
- squashed Gaussian policy using tanh,
- clipped double Q-learning,
- reparameterization trick,
- automatic temperature tuning,
- 10K random-action phase before learning starts,
- no separate state-value network.

This version returns eval/train logs and best/final checkpoints for per-seed artifacts.
"""
from __future__ import annotations

import copy
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
    evaluate_policy,
    EvalLog,
    TrainLog,
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

    # Assignment-safe default: collect 10K random transitions before learning.
    start_steps: int = 10_000
    update_after: int = 10_000
    update_every: int = 1
    grad_steps_per_update: int = 1

    autotune_alpha: bool = True
    init_alpha: float = 0.2
    target_entropy: float | None = None  # defaults to -act_dim
    reward_scale: float = 1.0


class SACAgent:
    def __init__(self, obs_dim: int, act_dim: int, act_limit: float,
                 config: SACConfig, device: torch.device | None = None):
        self.cfg = config
        self.device = device if device is not None else get_device()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.act_limit = float(act_limit)

        self.actor = SquashedGaussianActor(obs_dim, act_dim, act_limit, config.hidden).to(self.device)
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
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, _ = self.actor(obs_t, deterministic=deterministic, with_logprob=False)
        return action.cpu().numpy()[0]

    def random_action(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(-self.act_limit, self.act_limit, size=self.act_dim).astype(np.float32)

    def checkpoint(self) -> dict:
        """Serializable full checkpoint for final/best artifacts."""
        return {
            "actor": copy.deepcopy(self.actor.state_dict()),
            "critic": copy.deepcopy(self.critic.state_dict()),
            "critic_target": copy.deepcopy(self.critic_target.state_dict()),
            "log_alpha": self.log_alpha.detach().cpu().clone(),
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "act_limit": self.act_limit,
            "hidden": tuple(self.cfg.hidden),
            "total_env_steps": int(self.total_env_steps),
            "alpha": float(self.alpha.detach().cpu()),
        }

    def update(self) -> dict:
        batch = self.buffer.sample(self.cfg.batch_size, self.device)
        obs = batch["obs"]
        acts = batch["acts"]
        rews = batch["rews"] * self.cfg.reward_scale
        next_obs = batch["next_obs"]
        dones = batch["dones"]

        # Critic update.
        with torch.no_grad():
            next_action, next_logp = self.actor(next_obs)
            q1_t, q2_t = self.critic_target(next_obs, next_action)
            q_target_min = torch.min(q1_t, q2_t) - self.alpha.detach() * next_logp
            target = rews + self.cfg.gamma * (1.0 - dones) * q_target_min

        q1, q2 = self.critic(obs, acts)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Actor update.
        for p in self.critic.parameters():
            p.requires_grad_(False)
        pi_action, logp_pi = self.actor(obs)
        q1_pi, q2_pi = self.critic(obs, pi_action)
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

        # Temperature update.
        if self.cfg.autotune_alpha:
            alpha_loss = -(self.log_alpha * (logp_pi.detach() + self.target_entropy)).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            info["alpha_loss"] = float(alpha_loss.item())

        soft_update(self.critic_target, self.critic, self.cfg.tau)
        return info


def train_sac(env_fn: Callable, agent: SACAgent, *,
              total_steps: int,
              eval_every: int = 10_000,
              eval_episodes: int = 20,
              seed: int = 0,
              log_stdout: bool = True,
              train_log_every: int = 1000,
              # legacy single-eval path
              eval_env_fn: Optional[Callable] = None,
              eval_extra_reward_fns: Optional[dict] = None,
              # preferred Reacher path: true evaluation in separate envs
              eval_env_fns: Optional[dict[str, Callable]] = None,
              primary_eval_key: Optional[str] = None):
    """Train SAC with periodic deterministic offline evaluation.

    Returns:
        eval_log, train_log, best_checkpoint, final_checkpoint
    """
    rng = np.random.default_rng(seed)
    env = env_fn()
    obs, _ = env.reset(seed=seed)
    eval_log = EvalLog()
    train_log = TrainLog()

    best_return = -np.inf
    best_ckpt = agent.checkpoint()
    last_update_info = {}
    episode_return = 0.0
    episode_length = 0
    last_episode_return = np.nan
    last_episode_length = np.nan

    def _eval_now(step: int):
        nonlocal best_return, best_ckpt
        if eval_env_fns is not None:
            assert primary_eval_key in eval_env_fns, "primary_eval_key must be one of eval_env_fns"
            evals = {}
            primary_std = 0.0
            for name, fn in eval_env_fns.items():
                out = evaluate_policy(
                    fn,
                    act_fn=lambda o: agent.act(o, deterministic=True),
                    n_episodes=eval_episodes,
                )
                evals[f"{name}_return_mean"] = out["return"]
                evals[f"{name}_return_std"] = out["return_std"]
                if name == primary_eval_key:
                    primary_std = out["return_std"]
            primary_return = evals[f"{primary_eval_key}_return_mean"]
            eval_log.append_eval(
                step,
                primary_return,
                primary_std,
                alpha=float(agent.alpha.detach().cpu()),
                **evals,
            )
            if primary_return > best_return:
                best_return = primary_return
                best_ckpt = agent.checkpoint()
            if log_stdout:
                eval_s = " ".join(f"{k}={v:.2f}" for k, v in evals.items() if k.endswith("mean"))
                print(f"[step {step:>7}] {eval_s}")
            return

        assert eval_env_fn is not None, "Need either eval_env_fn or eval_env_fns"
        out = evaluate_policy(
            eval_env_fn,
            act_fn=lambda o: agent.act(o, deterministic=True),
            n_episodes=eval_episodes,
            extra_reward_fns=eval_extra_reward_fns,
        )
        extras = {k: v for k, v in out.items() if k not in ("return", "return_std")}
        eval_log.append_eval(
            step,
            out["return"],
            out["return_std"],
            alpha=float(agent.alpha.detach().cpu()),
            **extras,
        )
        if out["return"] > best_return:
            best_return = out["return"]
            best_ckpt = agent.checkpoint()
        if log_stdout:
            extras_s = " ".join(f"{k}={v:.2f}" for k, v in extras.items())
            print(f"[step {step:>7}] eval_return={out['return']:.2f} {extras_s}")

    # initial eval at step 0, as required.
    _eval_now(0)

    for t in range(1, total_steps + 1):
        if t <= agent.cfg.start_steps:
            action = agent.random_action(rng)
        else:
            action = agent.act(obs, deterministic=False)

        next_obs, reward, terminated, truncated, _info = env.step(action)
        # Bootstrap mask uses true termination, not timeout truncation.
        agent.buffer.add(obs, action, reward, next_obs, float(terminated))
        obs = next_obs
        agent.total_env_steps = t
        episode_return += float(reward)
        episode_length += 1

        if terminated or truncated:
            last_episode_return = episode_return
            last_episode_length = episode_length
            obs, _ = env.reset()
            episode_return = 0.0
            episode_length = 0

        if t >= agent.cfg.update_after and t % agent.cfg.update_every == 0:
            for _ in range(agent.cfg.grad_steps_per_update):
                last_update_info = agent.update()

        if t % train_log_every == 0:
            train_log.append(
                global_step=t,
                alpha=float(agent.alpha.detach().cpu()),
                replay_size=agent.buffer.size,
                last_episode_return=last_episode_return,
                last_episode_length=last_episode_length,
                **last_update_info,
            )

        if t % eval_every == 0:
            _eval_now(t)

    env.close()
    final_ckpt = agent.checkpoint()
    return eval_log, train_log, best_ckpt, final_ckpt
