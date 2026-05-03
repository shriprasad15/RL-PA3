"""Continuous-action Soft Actor-Critic.

Implements the SAC version with:
- Squashed Gaussian policy (tanh).
- Clipped double-Q learning (twin critics + min over targets).
- Reparameterisation trick for policy gradient.
- Optional automatic temperature tuning (alpha) via a log_alpha parameter optimised
  against a target entropy (= -act_dim by default, as in Haarnoja et al. 2018).
- Manual alpha mode: alpha held fixed.
- 10K random-action warmup before learning starts.

No state value network — only Q networks, per assignment spec.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from sac_core import (
    ReplayBuffer,
    SquashedGaussianActor,
    TwinQCritic,
    soft_update,
    get_device,
)


@dataclass
class SACConfig:
    # architecture / optimization
    hidden: tuple = (256, 256)
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    buffer_size: int = 1_000_000

    # SAC-specific
    start_steps: int = 10_000       # random-action warmup
    update_after: int = 10_000       # first gradient update step
    update_every: int = 1           # env steps between gradient steps
    grad_steps_per_update: int = 1  # gradient updates per "update_every" env step

    # temperature
    autotune_alpha: bool = True
    init_alpha: float = 0.2
    target_entropy: float = None     # defaults to -act_dim if None

    # reward scaling (for the 10x / 0.1x experiment)
    reward_scale: float = 1.0


class SACAgent:
    def __init__(self, obs_dim: int, act_dim: int, act_limit: float,
                 config: SACConfig, device: torch.device = None):
        self.cfg = config
        self.device = device if device is not None else get_device()
        self.obs_dim, self.act_dim, self.act_limit = obs_dim, act_dim, act_limit

        self.actor = SquashedGaussianActor(obs_dim, act_dim, act_limit, config.hidden).to(self.device)
        self.critic = TwinQCritic(obs_dim, act_dim, config.hidden).to(self.device)
        self.critic_target = TwinQCritic(obs_dim, act_dim, config.hidden).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        # temperature
        if config.autotune_alpha:
            self.target_entropy = (
                -float(act_dim) if config.target_entropy is None else float(config.target_entropy)
            )
            # parameterise log_alpha so alpha stays positive
            self.log_alpha = torch.tensor(np.log(config.init_alpha), device=self.device, requires_grad=True)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=config.alpha_lr)
        else:
            self.log_alpha = torch.tensor(np.log(config.init_alpha), device=self.device)
            self.alpha_opt = None
            self.target_entropy = None

        self.buffer = ReplayBuffer(config.buffer_size, obs_dim, act_dim, discrete=False)
        self.total_env_steps = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    # ---------------- action selection ----------------
    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        a, _ = self.actor(o, deterministic=deterministic, with_logprob=False)
        return a.cpu().numpy()[0]

    def random_action(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(-self.act_limit, self.act_limit, size=self.act_dim).astype(np.float32)

    # ---------------- core updates ----------------
    def update(self) -> dict:
        batch = self.buffer.sample(self.cfg.batch_size, self.device)
        o, a, r, o2, d = batch["obs"], batch["acts"], batch["rews"], batch["next_obs"], batch["dones"]
        r = r * self.cfg.reward_scale

        # ----- critic loss -----
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

        # ----- actor loss -----
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

        # ----- temperature loss -----
        info = {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": float(self.alpha.detach().cpu()),
            "logp": float(logp_pi.detach().mean().cpu()),
        }
        if self.cfg.autotune_alpha:
            alpha_loss = -(self.log_alpha * (logp_pi.detach() + self.target_entropy)).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            info["alpha_loss"] = alpha_loss.item()

        soft_update(self.critic_target, self.critic, self.cfg.tau)
        return info


# ---------------------------------------------------------------------------
# Main training loop (with periodic offline eval)
# ---------------------------------------------------------------------------
def train_sac(env_fn: Callable, agent: SACAgent, *,
              total_steps: int,
              eval_env_fn: Callable,
              eval_every: int = 10_000,
              eval_episodes: int = 20,
              eval_extra_reward_fns: dict = None,
              log_stdout: bool = True,
              seed: int = 0):
    """Standard SAC training loop.

    Returns an EvalLog with periodic (step, mean_return) records.
    """
    from sac_core import EvalLog
    rng = np.random.default_rng(seed)
    env = env_fn()
    obs, _ = env.reset(seed=seed)

    log = EvalLog()

    # initial eval at step 0
    def _eval_now(step):
        eval_out = _evaluate(agent, eval_env_fn, eval_episodes, eval_extra_reward_fns)
        extras = {k: v for k, v in eval_out.items() if k != "return"}
        log.append(step, eval_out["return"], extras)
        if log_stdout:
            extras_s = " ".join(f"{k}={v:.2f}" for k, v in extras.items())
            print(f"[step {step:>7}] eval return = {eval_out['return']:.2f}  {extras_s}")

    _eval_now(0)

    for t in range(1, total_steps + 1):
        if t <= agent.cfg.start_steps:
            a = agent.random_action(rng)
        else:
            a = agent.act(obs, deterministic=False)

        next_obs, r, term, trunc, _info = env.step(a)
        # "done" for bootstrap should be term (true termination), not trunc
        agent.buffer.add(obs, a, r, next_obs, float(term))
        obs = next_obs
        agent.total_env_steps = t
        if term or trunc:
            obs, _ = env.reset()

        if t >= agent.cfg.update_after and t % agent.cfg.update_every == 0:
            for _ in range(agent.cfg.grad_steps_per_update):
                agent.update()

        if t % eval_every == 0:
            _eval_now(t)

    env.close()
    return log


def _evaluate(agent, eval_env_fn, n_episodes, extra_reward_fns):
    from sac_core import evaluate_policy
    return evaluate_policy(
        eval_env_fn,
        act_fn=lambda o: agent.act(o, deterministic=True),
        n_episodes=n_episodes,
        extra_reward_fns=extra_reward_fns,
    )
