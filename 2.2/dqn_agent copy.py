"""Vanilla DQN (Mnih et al. 2015): target network + epsilon-greedy + replay."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sac_core import ReplayBuffer, mlp, soft_update, get_device


@dataclass
class DQNConfig:
    hidden: tuple = (256, 256)
    lr: float = 3e-4
    gamma: float = 0.99
    batch_size: int = 256
    buffer_size: int = 1_000_000
    start_steps: int = 10_000
    update_after: int = 10_000
    update_every: int = 1
    grad_steps_per_update: int = 1
    target_update_every: int = 1_000   # hard update
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 100_000


class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=(256, 256)):
        super().__init__()
        self.net = mlp([obs_dim, *hidden, n_actions])

    def forward(self, obs):
        return self.net(obs)


class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, cfg: DQNConfig, device=None):
        self.cfg = cfg
        self.device = device if device is not None else get_device()
        self.n_actions = n_actions

        self.q = QNetwork(obs_dim, n_actions, cfg.hidden).to(self.device)
        self.q_target = QNetwork(obs_dim, n_actions, cfg.hidden).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        for p in self.q_target.parameters():
            p.requires_grad_(False)

        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.buffer = ReplayBuffer(cfg.buffer_size, obs_dim, act_dim=1, discrete=True)
        self.total_env_steps = 0

    def epsilon(self) -> float:
        frac = min(1.0, self.total_env_steps / self.cfg.epsilon_decay_steps)
        return self.cfg.epsilon_start + frac * (self.cfg.epsilon_end - self.cfg.epsilon_start)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        if (not deterministic) and np.random.rand() < self.epsilon():
            return int(np.random.randint(self.n_actions))
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return int(self.q(o).argmax(dim=-1).item())

    def random_action(self, rng: np.random.Generator) -> int:
        return int(rng.integers(0, self.n_actions))

    def update(self) -> dict:
        batch = self.buffer.sample(self.cfg.batch_size, self.device)
        o, a, r, o2, d = batch["obs"], batch["acts"], batch["rews"], batch["next_obs"], batch["dones"]
        with torch.no_grad():
            q_next = self.q_target(o2).max(dim=-1).values
            y = r + self.cfg.gamma * (1 - d) * q_next
        q_all = self.q(o)
        q = q_all.gather(1, a.view(-1, 1)).squeeze(-1)
        loss = F.mse_loss(q, y)
        self.opt.zero_grad(); loss.backward(); self.opt.step()

        if self.total_env_steps % self.cfg.target_update_every == 0:
            self.q_target.load_state_dict(self.q.state_dict())
        return {"loss": float(loss.item()), "epsilon": float(self.epsilon())}


def train_dqn(env_fn: Callable, agent: DQNAgent, *,
              total_steps: int,
              eval_env_fn: Callable,
              eval_every: int = 10_000,
              eval_episodes: int = 20,
              log_stdout: bool = True,
              seed: int = 0):
    from sac_core import EvalLog, evaluate_policy
    rng = np.random.default_rng(seed)
    env = env_fn()
    obs, _ = env.reset(seed=seed)
    log = EvalLog()

    def _eval_now(step):
        out = evaluate_policy(eval_env_fn,
                              act_fn=lambda o: agent.act(o, deterministic=True),
                              n_episodes=eval_episodes)
        log.append(step, out["return"])
        if log_stdout:
            print(f"[step {step:>7}] eval return = {out['return']:.2f}")

    _eval_now(0)
    for t in range(1, total_steps + 1):
        agent.total_env_steps = t
        if t <= agent.cfg.start_steps:
            a = agent.random_action(rng)
        else:
            a = agent.act(obs, deterministic=False)
        next_obs, r, term, trunc, _ = env.step(a)
        agent.buffer.add(obs, a, r, next_obs, float(term))
        obs = next_obs
        if term or trunc:
            obs, _ = env.reset()
        if t >= agent.cfg.update_after and t % agent.cfg.update_every == 0:
            for _ in range(agent.cfg.grad_steps_per_update):
                agent.update()
        if t % eval_every == 0:
            _eval_now(t)
    env.close()
    return log
