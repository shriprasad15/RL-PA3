"""Shared utilities for PA3 2.2 LunarLander.

This version is artifact-friendly:
- fixed-size replay buffer
- actor/critic networks
- deterministic evaluation helper
- CSV/JSON artifact writers
- rollout collection helper
"""
from __future__ import annotations

import csv
import json
import os
import pickle
import random
from dataclasses import dataclass, field
from typing import Callable, Optional, Any

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


# ---------------------------------------------------------------------------
# JSON / CSV helpers
# ---------------------------------------------------------------------------
def _json_safe(x: Any):
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


def save_json(obj: Any, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(_json_safe(obj), f, indent=2)


def save_rows_csv(rows: list[dict], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        with open(path, "w") as f:
            f.write("")
        return
    # Stable column order: union keys in first-seen order.
    keys = []
    for row in rows:
        for k in row.keys():
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def write_done(run_dir: str, text: str = "done"):
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "DONE.txt"), "w") as f:
        f.write(text.rstrip() + "\n")


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
        self.discrete = discrete

    def add(self, o, a, r, o2, d):
        self.obs[self.ptr] = o
        self.next_obs[self.ptr] = o2
        self.acts[self.ptr] = a
        self.rews[self.ptr] = float(r)
        self.dones[self.ptr] = float(d)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device):
        if self.size <= 0:
            raise RuntimeError("Cannot sample from an empty replay buffer")
        idx = np.random.randint(0, self.size, size=batch_size)
        to_t = lambda x: torch.as_tensor(x, device=device)
        return dict(
            obs=to_t(self.obs[idx]),
            acts=to_t(self.acts[idx]),
            rews=to_t(self.rews[idx]),
            next_obs=to_t(self.next_obs[idx]),
            dones=to_t(self.dones[idx]),
        )


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------
def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, act_limit: float = 1.0, hidden=(256, 256)):
        super().__init__()
        self.net = mlp([obs_dim, *hidden], activation=nn.ReLU, output_activation=nn.ReLU)
        self.mu_head = nn.Linear(hidden[-1], act_dim)
        self.log_std_head = nn.Linear(hidden[-1], act_dim)
        self.act_limit = float(act_limit)

    def forward(self, obs, deterministic=False, with_logprob=True):
        h = self.net(obs)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        u = mu if deterministic else dist.rsample()  # reparameterization trick

        if with_logprob:
            logp = dist.log_prob(u).sum(-1)
            # tanh-squash log-prob correction
            logp -= (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(-1)
        else:
            logp = None

        a = torch.tanh(u) * self.act_limit
        return a, logp


class TwinQCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256)):
        super().__init__()
        self.q1 = mlp([obs_dim + act_dim, *hidden, 1])
        self.q2 = mlp([obs_dim + act_dim, *hidden, 1])

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class CategoricalActor(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden=(256, 256)):
        super().__init__()
        self.net = mlp([obs_dim, *hidden, n_actions])

    def forward(self, obs):
        logits = self.net(obs)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return probs, log_probs


class TwinDiscreteQ(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden=(256, 256)):
        super().__init__()
        self.q1 = mlp([obs_dim, *hidden, n_actions])
        self.q2 = mlp([obs_dim, *hidden, n_actions])

    def forward(self, obs):
        return self.q1(obs), self.q2(obs)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    with torch.no_grad():
        for p_t, p in zip(target.parameters(), source.parameters()):
            p_t.data.mul_(1.0 - tau).add_(p.data, alpha=tau)


# ---------------------------------------------------------------------------
# Evaluation logs
# ---------------------------------------------------------------------------
@dataclass
class EvalLog:
    steps: list = field(default_factory=list)
    returns: list = field(default_factory=list)
    stds: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def append(self, step: int, mean_ret: float, extras: Optional[dict] = None, std_ret: float | None = None):
        self.steps.append(int(step))
        self.returns.append(float(mean_ret))
        self.stds.append(float(std_ret) if std_ret is not None else float("nan"))
        if extras:
            for k, v in extras.items():
                self.extra.setdefault(k, []).append(float(v))

    def to_dict(self):
        d = {"steps": self.steps, "returns": self.returns, "return_stds": self.stds}
        d.update(self.extra)
        return d

    def to_rows(self, base: Optional[dict] = None):
        base = base or {}
        rows = []
        for i, step in enumerate(self.steps):
            row = dict(base)
            row.update({
                "global_step": int(step),
                "eval_return_mean": float(self.returns[i]),
                "eval_return_std": float(self.stds[i]) if i < len(self.stds) else "",
            })
            for k, vs in self.extra.items():
                if i < len(vs):
                    row[k] = float(vs[i])
            rows.append(row)
        return rows


def save_log(log: EvalLog, path: str, config: Optional[dict] = None):
    """Legacy JSON writer kept for compatibility with existing plotting/run_all code."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(_json_safe(log.to_dict()), f)
    if config is not None:
        cfg_path = path.rsplit(".json", 1)[0] + ".config.json"
        save_json(config, cfg_path)


def load_log(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def aggregate_runs(paths: list, key: str = "returns"):
    runs = [load_log(p) for p in paths]
    T = min(len(r["steps"]) for r in runs)
    steps = np.array(runs[0]["steps"][:T])
    vals = np.array([r[key][:T] for r in runs], dtype=float)
    mean = vals.mean(axis=0)
    std = vals.std(axis=0, ddof=1) if vals.shape[0] > 1 else np.zeros_like(mean)
    n = vals.shape[0]
    ci = 1.96 * std / np.sqrt(max(n, 1))
    return steps, mean, mean - ci, mean + ci


def plot_curves(ax, paths_by_label: dict, key: str = "returns",
                xlabel="Env steps", ylabel="Avg undiscounted return", title=None):
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


# ---------------------------------------------------------------------------
# Deterministic evaluation / rollout collection
# ---------------------------------------------------------------------------
def evaluate_policy(env_fn: Callable, act_fn: Callable, n_episodes: int = 20,
                    max_steps: Optional[int] = None, seed_offset: int = 10_000,
                    extra_reward_fns: Optional[dict] = None) -> dict:
    """Average undiscounted return over deterministic eval episodes."""
    env = env_fn()
    returns = []
    extras = {k: [] for k in (extra_reward_fns or {})}
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed_offset + ep)
        ep_ret = 0.0
        ep_extra = {k: 0.0 for k in extras}
        ep_len = 0
        done = False
        while not done:
            a = act_fn(obs)
            next_obs, r, term, trunc, step_info = env.step(a)
            ep_ret += float(r)
            for k, fn in (extra_reward_fns or {}).items():
                ep_extra[k] += float(fn(env, obs, a, r, step_info))
            obs = next_obs
            ep_len += 1
            done = bool(term or trunc)
            if max_steps is not None and ep_len >= max_steps:
                break
        returns.append(float(ep_ret))
        for k in extras:
            extras[k].append(float(ep_extra[k]))
    env.close()

    ret_arr = np.asarray(returns, dtype=float)
    out = {
        "return": float(ret_arr.mean()),
        "return_std": float(ret_arr.std(ddof=1)) if len(ret_arr) > 1 else 0.0,
        "episode_returns": returns,
    }
    for k, vs in extras.items():
        out[k] = float(np.mean(vs))
    return out


def _safe_info(info: dict):
    safe = {}
    for k, v in info.items():
        if isinstance(v, (bool, int, float, str)) or v is None:
            safe[k] = v
        elif isinstance(v, np.ndarray):
            safe[k] = v.tolist()
        else:
            safe[k] = str(v)
    return safe


def collect_rollout(env_fn: Callable, act_fn: Callable, *,
                    n_episodes: int = 1,
                    max_steps: Optional[int] = None,
                    seed_offset: int = 50_000):
    env = env_fn()
    episodes = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed_offset + ep)
        ep_obs, ep_actions, ep_rewards, ep_dones, ep_infos = [], [], [], [], []
        done = False
        t = 0
        while not done:
            a = act_fn(obs)
            next_obs, r, term, trunc, step_info = env.step(a)
            ep_obs.append(np.asarray(obs, dtype=np.float32))
            ep_actions.append(np.asarray(a).copy())
            ep_rewards.append(float(r))
            ep_dones.append(bool(term or trunc))
            ep_infos.append(_safe_info(step_info))
            obs = next_obs
            t += 1
            done = bool(term or trunc)
            if max_steps is not None and t >= max_steps:
                break
        episodes.append({
            "observations": np.asarray(ep_obs, dtype=np.float32),
            "actions": np.asarray(ep_actions),
            "rewards": np.asarray(ep_rewards, dtype=np.float32),
            "dones": np.asarray(ep_dones, dtype=bool),
            "infos": ep_infos,
            "return": float(np.sum(ep_rewards)),
            "length": int(len(ep_rewards)),
        })
    env.close()
    return {"episodes": episodes}


def save_pickle(obj: Any, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
