"""Shared building blocks: replay buffer, networks, seed utils, eval loop, plotting helpers.

This file is duplicated in 2.1/, 2.2/, 2.3/, 3/ so each section folder is self-contained.
Keep all copies in sync if editing.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Replay buffer (fixed-size numpy circular buffer)
# ---------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int, discrete: bool = False):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        if discrete:
            self.acts = np.zeros((capacity,), dtype=np.int64)
        else:
            self.acts = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rews = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.discrete = discrete

    def add(self, o, a, r, o2, d):
        self.obs[self.ptr] = o
        self.next_obs[self.ptr] = o2
        self.acts[self.ptr] = a
        self.rews[self.ptr] = r
        self.dones[self.ptr] = float(d)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device):
        idx = np.random.randint(0, self.size, size=batch_size)
        to_t = lambda x: torch.as_tensor(x, device=device)
        return dict(
            obs=to_t(self.obs[idx]),
            acts=to_t(self.acts[idx]),
            rews=to_t(self.rews[idx]),
            next_obs=to_t(self.next_obs[idx]),
            dones=to_t(self.dones[idx]),
        )

    def sample_segments(self, n_segments: int, seg_len: int):
        """Sample `n_segments` contiguous (obs, act) slices of length `seg_len`.

        Used by PEBBLE for preference queries. Returns numpy arrays, no tensor conversion.
        """
        assert self.size >= seg_len + 1, "Buffer too small for segment sampling"
        max_start = self.size - seg_len
        starts = np.random.randint(0, max_start, size=n_segments)
        obs = np.stack([self.obs[s:s + seg_len] for s in starts])
        acts = np.stack([self.acts[s:s + seg_len] for s in starts])
        return obs, acts, starts


# ---------------------------------------------------------------------------
# MLP factory
# ---------------------------------------------------------------------------
def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Continuous squashed-gaussian actor
# ---------------------------------------------------------------------------
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
        if deterministic:
            u = mu
        else:
            u = dist.rsample()  # reparameterised

        if with_logprob:
            logp = dist.log_prob(u).sum(-1)
            # numerically stable log(1 - tanh(u)^2) correction
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


# ---------------------------------------------------------------------------
# Discrete categorical actor + twin Q over discrete actions
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Soft target update
# ---------------------------------------------------------------------------
def soft_update(target: nn.Module, source: nn.Module, tau: float):
    with torch.no_grad():
        for p_t, p in zip(target.parameters(), source.parameters()):
            p_t.data.mul_(1 - tau).add_(p.data, alpha=tau)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_policy(env_fn: Callable, act_fn: Callable, n_episodes: int = 20,
                    max_steps: Optional[int] = None, seed_offset: int = 10_000,
                    extra_reward_fns: Optional[dict] = None) -> dict:
    """Average undiscounted return over `n_episodes` deterministic episodes.

    Returns {"return": mean, **{name: mean_under_that_reward for name in extra_reward_fns}}.
    extra_reward_fns: {name: fn(env, obs, action, reward, info) -> scalar} for cross-eval.
    """
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
            done = term or trunc
            if max_steps is not None and ep_len >= max_steps:
                break
        returns.append(ep_ret)
        for k in extras:
            extras[k].append(ep_extra[k])
    env.close()
    out = {"return": float(np.mean(returns))}
    for k, vs in extras.items():
        out[k] = float(np.mean(vs))
    return out


# ---------------------------------------------------------------------------
# Logging / plotting helpers
# ---------------------------------------------------------------------------
@dataclass
class EvalLog:
    steps: list = field(default_factory=list)
    returns: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)  # extra_key -> list[float]

    def append(self, step: int, ret: float, extras: Optional[dict] = None):
        self.steps.append(int(step))
        self.returns.append(float(ret))
        if extras:
            for k, v in extras.items():
                self.extra.setdefault(k, []).append(float(v))

    def to_dict(self):
        d = {"steps": self.steps, "returns": self.returns}
        d.update(self.extra)
        return d


def save_log(log: EvalLog, path: str, config: Optional[dict] = None):
    """Save eval log (and optionally a config dict) to `path`. Config is saved to
    `<path>.config.json` next to the log file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(log.to_dict(), f)
    if config is not None:
        cfg_path = path.rsplit(".json", 1)[0] + ".config.json"
        with open(cfg_path, "w") as f:
            # cast non-serialisable values (tuples, dataclasses) to primitives
            def _san(v):
                if isinstance(v, (list, tuple)):
                    return [_san(x) for x in v]
                if isinstance(v, dict):
                    return {k: _san(x) for k, x in v.items()}
                if isinstance(v, (str, int, float, bool)) or v is None:
                    return v
                return str(v)
            json.dump({k: _san(v) for k, v in config.items()}, f, indent=2)


def load_log(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def aggregate_runs(paths: list, key: str = "returns"):
    """Return (steps, mean, lo, hi) with 95% CI across runs."""
    runs = [load_log(p) for p in paths]
    # align length to shortest if seeds ran different # of evals
    T = min(len(r["steps"]) for r in runs)
    steps = np.array(runs[0]["steps"][:T])
    vals = np.array([r[key][:T] for r in runs])
    mean = vals.mean(axis=0)
    std = vals.std(axis=0)
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
