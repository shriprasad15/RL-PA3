"""Shared SAC utilities for PA3 Section 2.3 Reacher experiments.

Artifact-friendly version:
- table-style CSV/JSON logs,
- deterministic evaluation with mean/std return,
- rollout collection for later trajectory/t-SNE/video analysis,
- legacy JSON saving for compatibility with older notebooks/run_all.py.
"""
from __future__ import annotations

import csv
import json
import os
import pickle
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Reproducibility / device
# ---------------------------------------------------------------------------
def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str | os.PathLike):
    os.makedirs(path, exist_ok=True)


def _json_safe(x: Any):
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


def save_json(obj: dict, path: str | os.PathLike):
    ensure_dir(os.path.dirname(str(path)) or ".")
    with open(path, "w") as f:
        json.dump(_json_safe(obj), f, indent=2)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int, discrete: bool = False):
        self.capacity = int(capacity)
        self.ptr = 0
        self.size = 0
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        if discrete:
            self.acts = np.zeros((self.capacity,), dtype=np.int64)
        else:
            self.acts = np.zeros((self.capacity, act_dim), dtype=np.float32)
        self.rews = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)
        self.discrete = bool(discrete)

    def add(self, obs, act, rew, next_obs, done):
        self.obs[self.ptr] = obs
        self.next_obs[self.ptr] = next_obs
        self.acts[self.ptr] = act
        self.rews[self.ptr] = rew
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device):
        if self.size < batch_size:
            raise RuntimeError(f"ReplayBuffer size={self.size}, cannot sample batch_size={batch_size}")
        idx = np.random.randint(0, self.size, size=batch_size)
        to_t = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
        batch = {
            "obs": to_t(self.obs[idx]),
            "rews": to_t(self.rews[idx]),
            "next_obs": to_t(self.next_obs[idx]),
            "dones": to_t(self.dones[idx]),
        }
        if self.discrete:
            batch["acts"] = torch.as_tensor(self.acts[idx], dtype=torch.long, device=device)
        else:
            batch["acts"] = to_t(self.acts[idx])
        return batch


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------
def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, act_limit: float = 1.0,
                 hidden=(256, 256)):
        super().__init__()
        self.net = mlp([obs_dim, *hidden], activation=nn.ReLU, output_activation=nn.ReLU)
        self.mu_head = nn.Linear(hidden[-1], act_dim)
        self.log_std_head = nn.Linear(hidden[-1], act_dim)
        self.act_limit = float(act_limit)

    def forward(self, obs, deterministic: bool = False, with_logprob: bool = True):
        h = self.net(obs)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)

        if deterministic:
            raw_action = mu
        else:
            raw_action = dist.rsample()  # reparameterization trick

        if with_logprob:
            logp = dist.log_prob(raw_action).sum(dim=-1)
            # Stable tanh correction from SAC.
            logp -= (2 * (np.log(2) - raw_action - F.softplus(-2 * raw_action))).sum(dim=-1)
        else:
            logp = None

        action = torch.tanh(raw_action) * self.act_limit
        return action, logp


class TwinQCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256)):
        super().__init__()
        self.q1 = mlp([obs_dim + act_dim, *hidden, 1])
        self.q2 = mlp([obs_dim + act_dim, *hidden, 1])

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    with torch.no_grad():
        for p_t, p in zip(target.parameters(), source.parameters()):
            p_t.data.mul_(1.0 - tau).add_(p.data, alpha=tau)


# ---------------------------------------------------------------------------
# Table logs
# ---------------------------------------------------------------------------
@dataclass
class TableLog:
    rows: list[dict] = field(default_factory=list)

    def append(self, **kwargs):
        self.rows.append({k: _json_safe(v) for k, v in kwargs.items()})

    def to_csv(self, path: str | os.PathLike):
        ensure_dir(os.path.dirname(str(path)) or ".")
        if not self.rows:
            with open(path, "w", newline="") as f:
                f.write("")
            return
        fieldnames = []
        seen = set()
        for row in self.rows:
            for k in row.keys():
                if k not in seen:
                    fieldnames.append(k)
                    seen.add(k)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row)

    def to_json(self, path: str | os.PathLike):
        save_json({"rows": self.rows}, path)


class EvalLog(TableLog):
    def append_eval(self, step: int, primary_return_mean: float, primary_return_std: float = 0.0, **extras):
        self.append(
            global_step=int(step),
            eval_return_mean=float(primary_return_mean),
            eval_return_std=float(primary_return_std),
            **extras,
        )

    def to_legacy_dict(self) -> dict:
        """Legacy shape: {steps: [...], returns: [...], extra_key: [...]}.

        This keeps old plotting notebooks and run_all.py-style checks usable.
        """
        d = {
            "steps": [int(r.get("global_step", 0)) for r in self.rows],
            "returns": [float(r.get("eval_return_mean", np.nan)) for r in self.rows],
        }
        for row in self.rows:
            for k, v in row.items():
                if k in ("global_step", "eval_return_mean", "eval_return_std"):
                    continue
                d.setdefault(k, []).append(_json_safe(v))
        return d


class TrainLog(TableLog):
    pass


# ---------------------------------------------------------------------------
# Evaluation / rollout
# ---------------------------------------------------------------------------
def evaluate_policy(env_fn: Callable, act_fn: Callable, n_episodes: int = 20,
                    max_steps: Optional[int] = None, seed_offset: int = 10_000,
                    extra_reward_fns: Optional[dict] = None) -> dict:
    """Average undiscounted return over deterministic episodes."""
    env = env_fn()
    returns = []
    extras = {k: [] for k in (extra_reward_fns or {})}

    for ep in range(n_episodes):
        obs, _info = env.reset(seed=seed_offset + ep)
        ep_return = 0.0
        ep_extra = {k: 0.0 for k in extras}
        done = False
        steps = 0
        while not done:
            action = act_fn(obs)
            next_obs, reward, term, trunc, info = env.step(action)
            ep_return += float(reward)
            for k, fn in (extra_reward_fns or {}).items():
                ep_extra[k] += float(fn(env, obs, action, reward, info))
            obs = next_obs
            steps += 1
            done = bool(term or trunc)
            if max_steps is not None and steps >= max_steps:
                break
        returns.append(ep_return)
        for k in extras:
            extras[k].append(ep_extra[k])

    env.close()
    out = {"return": float(np.mean(returns)), "return_std": float(np.std(returns))}
    for k, values in extras.items():
        out[k] = float(np.mean(values))
    return out


def collect_rollout(env_fn: Callable, act_fn: Callable, *, seed: int = 0,
                    max_steps: int = 1000) -> dict:
    """Collect one deterministic rollout for later trajectory/t-SNE/video diagnostics."""
    env = env_fn()
    obs, info = env.reset(seed=seed)
    observations, actions, rewards, dones, infos = [], [], [], [], []
    total_return = 0.0

    for _ in range(max_steps):
        action = act_fn(obs)
        next_obs, reward, term, trunc, step_info = env.step(action)
        observations.append(np.asarray(obs, dtype=np.float32))
        actions.append(np.asarray(action, dtype=np.float32))
        rewards.append(float(reward))
        dones.append(bool(term or trunc))
        infos.append({k: _json_safe(v) for k, v in step_info.items()})
        total_return += float(reward)
        obs = next_obs
        if term or trunc:
            break

    env.close()
    return {
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "dones": np.asarray(dones, dtype=bool),
        "infos": infos,
        "return": float(total_return),
        "length": int(len(rewards)),
        "seed": int(seed),
    }


# ---------------------------------------------------------------------------
# Legacy save/load/plot helpers
# ---------------------------------------------------------------------------
def save_log(log: EvalLog, path: str | os.PathLike, config: Optional[dict] = None):
    """Legacy JSON save: logs/<tag>_seedX.json and .config.json."""
    ensure_dir(os.path.dirname(str(path)) or ".")
    with open(path, "w") as f:
        json.dump(log.to_legacy_dict(), f)
    if config is not None:
        cfg_path = str(path).rsplit(".json", 1)[0] + ".config.json"
        save_json(config, cfg_path)


def load_log(path: str | os.PathLike) -> dict:
    with open(path) as f:
        return json.load(f)


def aggregate_runs(paths: list, key: str = "returns"):
    """Return steps, mean, lower CI, upper CI with 95% CI across seeds."""
    runs = [load_log(p) for p in paths]
    T = min(len(r["steps"]) for r in runs)
    steps = np.asarray(runs[0]["steps"][:T])
    values = np.asarray([r[key][:T] for r in runs], dtype=np.float32)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    n = values.shape[0]
    ci = 1.96 * std / np.sqrt(max(n, 1))
    return steps, mean, mean - ci, mean + ci


def plot_curves(ax, paths_by_label: dict, key: str = "returns",
                xlabel: str = "Env steps",
                ylabel: str = "Avg undiscounted return",
                title: str | None = None):
    for label, paths in paths_by_label.items():
        steps, mean, lo, hi = aggregate_runs(paths, key=key)
        ax.plot(steps, mean, label=label)
        ax.fill_between(steps, lo, hi, alpha=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
