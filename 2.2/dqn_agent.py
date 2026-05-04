"""Vanilla DQN for PA3 2.2 discrete LunarLander."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sac_core import ReplayBuffer, mlp, get_device, EvalLog, evaluate_policy


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
    target_update_every: int = 1_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 100_000


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden=(256, 256)):
        super().__init__()
        self.net = mlp([obs_dim, *hidden, n_actions])

    def forward(self, obs):
        return self.net(obs)


class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, cfg: DQNConfig,
                 device: torch.device | None = None):
        self.cfg = cfg
        self.device = device if device is not None else get_device()
        self.obs_dim = obs_dim
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
        o = batch["obs"].float()
        a = batch["acts"].long()
        r = batch["rews"].float()
        o2 = batch["next_obs"].float()
        d = batch["dones"].float()

        with torch.no_grad():
            q_next = self.q_target(o2).max(dim=-1).values
            y = r + self.cfg.gamma * (1.0 - d) * q_next

        q_all = self.q(o)
        q = q_all.gather(1, a.view(-1, 1)).squeeze(-1)
        loss = F.mse_loss(q, y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        if self.total_env_steps % self.cfg.target_update_every == 0:
            self.q_target.load_state_dict(self.q.state_dict())

        return {"q_loss": float(loss.item()), "epsilon": float(self.epsilon())}

    def checkpoint(self) -> dict:
        return {
            "agent_type": "dqn",
            "q": self.q.state_dict(),
            "q_target": self.q_target.state_dict(),
            "optimizer": self.opt.state_dict(),
            "config": self.cfg.__dict__,
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
            "total_env_steps": self.total_env_steps,
        }


def train_dqn(env_fn: Callable, agent: DQNAgent, *,
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
        agent.total_env_steps = t
        a = agent.random_action(rng) if t <= agent.cfg.start_steps else agent.act(obs, deterministic=False)
        next_obs, r, term, trunc, _ = env.step(a)
        agent.buffer.add(obs, a, r, next_obs, float(term))
        obs = next_obs
        ep_ret += float(r)
        ep_len += 1

        if term or trunc:
            train_rows.append({
                "global_step": t,
                "event": "episode_end",
                "episode_return": ep_ret,
                "episode_length": ep_len,
                "epsilon": float(agent.epsilon()),
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
                "epsilon": float(agent.epsilon()),
            }
            row.update(last_update_info)
            train_rows.append(row)

        if t % eval_every == 0:
            _eval_now(t)

    env.close()
    return eval_log, train_rows
