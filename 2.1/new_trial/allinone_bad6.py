#!/usr/bin/env python3
"""
All-in-one Pendulum SAC rerun for bad angles with torque limit ±6.

Runs:
1. Auto-alpha SAC for bad targets: -60, -90, 90, 120
2. Manual-alpha search for: -60, 90, 120
3. Selects best manual alpha using final return + AUC
4. Manual final 15-seed runs
5. Reward scaling at target 90: scales 10 and 0.1, auto + manual
6. Saves CSV logs, best/final models, summary CSVs, and plots

This is intentionally one file. No manifest scripts needed.

Default hyperparameters:
- total steps: 300000
- seeds: 0-14
- random exploration: 10000
- learning starts: 10000
- eval every: 10000
- eval episodes: 20
- gamma: 0.99
- tau: 0.005
- lr: 3e-4
- hidden: 256,256
- batch: 256
- replay: 1_000_000
- torque limit/action limit: ±6
"""

from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import argparse
import copy
import json
import math
import multiprocessing as mp
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm


# =========================
# Global defaults
# =========================

AUTO_TARGETS = [-60.0, -90.0, 90.0, 120.0]
MANUAL_TARGETS = [-60.0, 90.0, 120.0]
ALPHA_GRID = [0.003, 0.005, 0.01, 0.03, 0.05, 0.1, 0.2]
REWARD_SCALES = [10.0, 0.1]

TOTAL_STEPS = 300_000
EPISODE_LENGTH = 1000
EVAL_EVERY = 10_000
EVAL_EPISODES = 20
RANDOM_STEPS = 10_000
LEARNING_STARTS = 10_000
UPDATES_PER_STEP = 1

GAMMA = 0.99
TAU = 0.005
LR = 3e-4
BATCH_SIZE = 256
REPLAY_SIZE = 1_000_000
HIDDEN = 256
INIT_ALPHA = 0.2
TARGET_ENTROPY = -1.0

ANGLE_W = 1.0
OMEGA_W = 0.1
TORQUE_W = 0.001

TORQUE_LIMIT = 6.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Utilities
# =========================

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_angle_label(theta: float) -> str:
    sign = "p" if theta >= 0 else "m"
    return f"{sign}{abs(int(theta)):03d}"


def condition_name(mode: str, target: float, reward_scale: float = 1.0, manual_alpha: Optional[float] = None) -> str:
    t = safe_angle_label(target)
    if mode.startswith("auto"):
        return f"auto_target_{t}_scale_{reward_scale:g}"
    a = "none" if manual_alpha is None else str(manual_alpha).replace(".", "p")
    return f"manual_target_{t}_alpha_{a}_scale_{reward_scale:g}"


def wrap_to_pi(x: float) -> float:
    return ((x + math.pi) % (2 * math.pi)) - math.pi


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def write_json(obj: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


# =========================
# Custom Pendulum with torque ±6
# =========================

class TargetPendulumTorque6(gym.Env):
    """
    Uses Gymnasium Pendulum-v1 dynamics but overrides max_torque to ±6
    and uses target-angle reward.

    Reward:
        r = -reward_scale * (
              angle_error^2 + 0.1 * theta_dot^2 + 0.001 * torque^2
            )
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        target_angle_deg: float,
        reward_scale: float = 1.0,
        torque_limit: float = TORQUE_LIMIT,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.env = gym.make("Pendulum-v1", render_mode=render_mode, max_episode_steps=EPISODE_LENGTH)
        self.unwrapped_env = self.env.unwrapped

        # Main actuator change.
        self.unwrapped_env.max_torque = float(torque_limit)
        self.action_space = gym.spaces.Box(
            low=np.array([-float(torque_limit)], dtype=np.float32),
            high=np.array([float(torque_limit)], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = self.env.observation_space

        self.target_angle_deg = float(target_angle_deg)
        self.target_angle_rad = math.radians(float(target_angle_deg))
        self.reward_scale = float(reward_scale)
        self.torque_limit = float(torque_limit)
        self.render_mode = render_mode

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info

    def step(self, action):
        # Reward computed from state before transition, same style as original Pendulum.
        theta, theta_dot = self.unwrapped_env.state
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        clipped = np.clip(action, -self.torque_limit, self.torque_limit)
        torque = float(clipped[0])

        angle_error = wrap_to_pi(float(theta) - self.target_angle_rad)
        base_cost = (
            ANGLE_W * angle_error**2
            + OMEGA_W * float(theta_dot) ** 2
            + TORQUE_W * torque**2
        )
        reward = -self.reward_scale * base_cost

        obs, _orig_reward, terminated, truncated, info = self.env.step(np.array([torque], dtype=np.float32))

        info = dict(info)
        info.update({
            "theta_rad": float(wrap_to_pi(float(theta))),
            "theta_deg": float(np.degrees(wrap_to_pi(float(theta)))),
            "target_angle_deg": self.target_angle_deg,
            "angle_error_rad": float(angle_error),
            "angle_error_deg": float(np.degrees(angle_error)),
            "abs_angle_error_deg": abs(float(np.degrees(angle_error))),
            "theta_dot": float(theta_dot),
            "torque": torque,
            "torque_sq": torque**2,
            "base_cost": float(base_cost),
            "custom_reward": float(reward),
        })
        return obs, float(reward), terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


def make_env(target_angle_deg: float, reward_scale: float = 1.0, render_mode: Optional[str] = None):
    return TargetPendulumTorque6(target_angle_deg, reward_scale, TORQUE_LIMIT, render_mode)


# =========================
# Replay buffer
# =========================

class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, size: int, seed: int):
        self.max_size = int(size)
        self.rng = np.random.default_rng(seed)
        self.obs = np.zeros((self.max_size, obs_dim), dtype=np.float32)
        self.obs2 = np.zeros((self.max_size, obs_dim), dtype=np.float32)
        self.act = np.zeros((self.max_size, act_dim), dtype=np.float32)
        self.rew = np.zeros((self.max_size, 1), dtype=np.float32)
        self.done = np.zeros((self.max_size, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, o, a, r, o2, d):
        self.obs[self.ptr] = o
        self.obs2[self.ptr] = o2
        self.act[self.ptr] = a
        self.rew[self.ptr] = float(r)
        self.done[self.ptr] = float(d)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int):
        idx = self.rng.integers(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.obs[idx], device=DEVICE),
            torch.as_tensor(self.act[idx], device=DEVICE),
            torch.as_tensor(self.rew[idx], device=DEVICE),
            torch.as_tensor(self.obs2[idx], device=DEVICE),
            torch.as_tensor(self.done[idx], device=DEVICE),
        )

    def __len__(self):
        return self.size


# =========================
# SAC networks
# =========================

LOG_STD_MIN = -20
LOG_STD_MAX = 2
EPS = 1e-6


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, act_limit: float):
        super().__init__()
        self.act_limit = float(act_limit)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
        )
        self.mu = nn.Linear(HIDDEN, act_dim)
        self.log_std = nn.Linear(HIDDEN, act_dim)

    def forward(self, obs):
        h = self.net(obs)
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs):
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        x = dist.rsample()
        y = torch.tanh(x)
        action = y * self.act_limit

        logp = dist.log_prob(x)
        logp -= torch.log(self.act_limit * (1 - y.pow(2)) + EPS)
        logp = logp.sum(dim=-1, keepdim=True)

        deterministic = torch.tanh(mu) * self.act_limit
        return action, logp, deterministic


class QNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, 1),
        )

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1))


class SACAgent:
    def __init__(self, obs_dim: int, act_dim: int, act_limit: float, auto_alpha: bool = True, alpha: float = INIT_ALPHA):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = float(act_limit)
        self.auto_alpha = bool(auto_alpha)

        self.actor = Actor(obs_dim, act_dim, act_limit).to(DEVICE)
        self.q1 = QNet(obs_dim, act_dim).to(DEVICE)
        self.q2 = QNet(obs_dim, act_dim).to(DEVICE)
        self.q1_targ = copy.deepcopy(self.q1).to(DEVICE)
        self.q2_targ = copy.deepcopy(self.q2).to(DEVICE)

        for p in self.q1_targ.parameters():
            p.requires_grad = False
        for p in self.q2_targ.parameters():
            p.requires_grad = False

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=LR)
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=LR)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=LR)

        if self.auto_alpha:
            self.log_alpha = torch.tensor(math.log(alpha), requires_grad=True, device=DEVICE)
            self.alpha_opt = optim.Adam([self.log_alpha], lr=LR)
        else:
            self.log_alpha = None
            self.alpha_opt = None
            self.fixed_alpha = float(alpha)

    @property
    def alpha(self):
        if self.auto_alpha:
            return self.log_alpha.exp()
        return torch.tensor(self.fixed_alpha, device=DEVICE)

    @torch.no_grad()
    def act(self, obs, deterministic: bool = False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        a, _logp, det = self.actor.sample(obs_t)
        out = det if deterministic else a
        return out.cpu().numpy()[0]

    def update(self, replay: ReplayBuffer):
        o, a, r, o2, d = replay.sample(BATCH_SIZE)

        with torch.no_grad():
            a2, logp2, _ = self.actor.sample(o2)
            q1_t = self.q1_targ(o2, a2)
            q2_t = self.q2_targ(o2, a2)
            q_t = torch.min(q1_t, q2_t) - self.alpha.detach() * logp2
            backup = r + GAMMA * (1 - d) * q_t

        q1 = self.q1(o, a)
        q2 = self.q2(o, a)
        q1_loss = F.mse_loss(q1, backup)
        q2_loss = F.mse_loss(q2, backup)

        self.q1_opt.zero_grad(set_to_none=True)
        q1_loss.backward()
        self.q1_opt.step()

        self.q2_opt.zero_grad(set_to_none=True)
        q2_loss.backward()
        self.q2_opt.step()

        pi, logp, _ = self.actor.sample(o)
        q_pi = torch.min(self.q1(o, pi), self.q2(o, pi))
        actor_loss = (self.alpha.detach() * logp - q_pi).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (logp + TARGET_ENTROPY).detach()).mean()
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()
            alpha_loss_value = float(alpha_loss.item())
        else:
            alpha_loss_value = 0.0

        with torch.no_grad():
            for p, pt in zip(self.q1.parameters(), self.q1_targ.parameters()):
                pt.data.mul_(1 - TAU).add_(TAU * p.data)
            for p, pt in zip(self.q2.parameters(), self.q2_targ.parameters()):
                pt.data.mul_(1 - TAU).add_(TAU * p.data)

        return {
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": alpha_loss_value,
            "alpha": float(self.alpha.detach().cpu().item()),
        }

    def save(self, path: str | Path):
        torch.save({
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "auto_alpha": self.auto_alpha,
            "alpha": float(self.alpha.detach().cpu().item()),
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "act_limit": self.act_limit,
        }, path)


# =========================
# Evaluation / training
# =========================

def evaluate(agent: SACAgent, target: float, reward_scale: float, seed: int):
    env = make_env(target, reward_scale, render_mode=None)
    returns = []
    abs_errs = []
    omegas = []
    torque_sqs = []
    occupancies = []
    tol_deg = 10.0

    for ep in range(EVAL_EPISODES):
        obs, _ = env.reset(seed=seed + 10000 + ep)
        done = False
        ret = 0.0

        while not done:
            action = agent.act(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            done = bool(term or trunc)
            ret += float(reward)

            abs_errs.append(float(info["abs_angle_error_deg"]))
            omegas.append(abs(float(info["theta_dot"])))
            torque_sqs.append(float(info["torque_sq"]))
            occupancies.append(float(info["abs_angle_error_deg"] < tol_deg))

        returns.append(ret)

    env.close()
    return {
        "eval_return_mean": float(np.mean(returns)),
        "eval_return_std_over_episodes": float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0,
        "mean_abs_angle_error_deg": float(np.mean(abs_errs)),
        "mean_abs_angular_velocity": float(np.mean(omegas)),
        "mean_squared_torque": float(np.mean(torque_sqs)),
        "target_occupancy": float(np.mean(occupancies)),
    }


def train_one(job: dict):
    torch.set_num_threads(int(job.get("torch_num_threads", 1)))

    target = float(job["target"])
    seed = int(job["seed"])
    mode = str(job["mode"])
    reward_scale = float(job.get("reward_scale", 1.0))
    manual_alpha = job.get("manual_alpha", None)
    total_steps = int(job.get("total_steps", TOTAL_STEPS))
    out_root = Path(job["out_root"])

    auto_alpha = mode.startswith("auto")
    alpha = INIT_ALPHA if auto_alpha else float(manual_alpha)

    cond = condition_name(mode, target, reward_scale, manual_alpha)
    seed_dir = ensure_dir(out_root / cond / f"seed_{seed:02d}")
    done_marker = seed_dir / "DONE.txt"

    if done_marker.exists() and (seed_dir / "best_model.pt").exists() and (seed_dir / "eval_log.csv").exists():
        return {"status": "skipped_done", "condition": cond, "seed": seed, "seed_dir": str(seed_dir)}

    set_seed(seed)

    env = make_env(target, reward_scale, render_mode=None)
    env.action_space.seed(seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    agent = SACAgent(obs_dim, act_dim, act_limit, auto_alpha=auto_alpha, alpha=alpha)
    replay = ReplayBuffer(obs_dim, act_dim, REPLAY_SIZE, seed)

    eval_rows = []
    train_rows = []
    update_acc = {"q1_loss": [], "q2_loss": [], "actor_loss": [], "alpha_loss": [], "alpha": []}

    best_return = -float("inf")
    start_time = time.time()

    def save_status(step, phase, extra=None):
        payload = {
            "condition": cond,
            "seed": seed,
            "target": target,
            "mode": mode,
            "reward_scale": reward_scale,
            "manual_alpha": manual_alpha,
            "step": step,
            "total_steps": total_steps,
            "phase": phase,
            "elapsed_seconds": time.time() - start_time,
            "alpha": float(agent.alpha.detach().cpu().item()),
            "replay_size": len(replay),
        }
        if extra:
            payload.update(extra)
        write_json(payload, seed_dir / "status.json")

    # Initial eval
    stats = evaluate(agent, target, reward_scale, seed)
    stats.update({
        "timestep": 0,
        "seed": seed,
        "target_angle_deg": target,
        "reward_scale": reward_scale,
        "mode": mode,
        "manual_alpha": np.nan if manual_alpha is None else float(manual_alpha),
        "alpha": float(agent.alpha.detach().cpu().item()),
        "condition": cond,
    })
    eval_rows.append(stats)
    pd.DataFrame(eval_rows).to_csv(seed_dir / "eval_log.csv", index=False)
    best_return = stats["eval_return_mean"]
    agent.save(seed_dir / "best_model.pt")

    obs, _ = env.reset(seed=seed)
    ep_ret = 0.0
    ep_len = 0

    save_status(0, "started")

    for step in range(1, total_steps + 1):
        if step <= RANDOM_STEPS:
            action = env.action_space.sample()
        else:
            action = agent.act(obs, deterministic=False)

        next_obs, reward, term, trunc, info = env.step(action)
        done_for_buffer = float(term)  # Pendulum truncation should not be terminal for bootstrapping.
        replay.add(obs, action, reward, next_obs, done_for_buffer)

        obs = next_obs
        ep_ret += float(reward)
        ep_len += 1

        if step >= LEARNING_STARTS and len(replay) >= BATCH_SIZE:
            for _ in range(UPDATES_PER_STEP):
                losses = agent.update(replay)
                for k, v in losses.items():
                    update_acc[k].append(v)

        if term or trunc:
            train_rows.append({
                "timestep": step,
                "seed": seed,
                "target_angle_deg": target,
                "reward_scale": reward_scale,
                "mode": mode,
                "manual_alpha": np.nan if manual_alpha is None else float(manual_alpha),
                "condition": cond,
                "episode_return": ep_ret,
                "episode_length": ep_len,
                "alpha": float(agent.alpha.detach().cpu().item()),
            })
            obs, _ = env.reset(seed=seed + step)
            ep_ret = 0.0
            ep_len = 0

        if step % 5000 == 0:
            save_status(step, "training")

        if step % EVAL_EVERY == 0:
            save_status(step, "evaluating")
            stats = evaluate(agent, target, reward_scale, seed + step)
            mean_updates = {
                f"mean_{k}_since_last_eval": float(np.mean(v)) if len(v) else np.nan
                for k, v in update_acc.items()
            }
            update_acc = {"q1_loss": [], "q2_loss": [], "actor_loss": [], "alpha_loss": [], "alpha": []}

            stats.update({
                "timestep": step,
                "seed": seed,
                "target_angle_deg": target,
                "reward_scale": reward_scale,
                "mode": mode,
                "manual_alpha": np.nan if manual_alpha is None else float(manual_alpha),
                "alpha": float(agent.alpha.detach().cpu().item()),
                "condition": cond,
                "wallclock_seconds": time.time() - start_time,
                **mean_updates,
            })
            eval_rows.append(stats)

            pd.DataFrame(eval_rows).to_csv(seed_dir / "eval_log.csv", index=False)
            pd.DataFrame(train_rows).to_csv(seed_dir / "train_log.csv", index=False)

            if stats["eval_return_mean"] > best_return:
                best_return = stats["eval_return_mean"]
                agent.save(seed_dir / "best_model.pt")

            save_status(step, "eval_done", {
                "last_eval_return": stats["eval_return_mean"],
                "best_eval_return": best_return,
                "mean_abs_angle_error_deg": stats["mean_abs_angle_error_deg"],
                "target_occupancy": stats["target_occupancy"],
            })

    agent.save(seed_dir / "final_model.pt")
    pd.DataFrame(eval_rows).to_csv(seed_dir / "eval_log.csv", index=False)
    pd.DataFrame(train_rows).to_csv(seed_dir / "train_log.csv", index=False)

    with open(done_marker, "w") as f:
        f.write(f"completed_at={time.time()}\n")

    env.close()

    save_status(total_steps, "completed", {
        "best_eval_return": best_return,
    })

    return {
        "status": "completed",
        "condition": cond,
        "seed": seed,
        "seed_dir": str(seed_dir),
        "best_eval_return": best_return,
        "wallclock_seconds": time.time() - start_time,
    }


# =========================
# Job management
# =========================

def run_jobs(jobs: List[dict], workers: int, result_path: Path):
    result_path.parent.mkdir(parents=True, exist_ok=True)
    workers = max(1, min(int(workers), len(jobs)))

    print(f"Running {len(jobs)} jobs with {workers} workers")
    print(f"Results -> {result_path}")

    ctx = mp.get_context("fork")
    completed = 0
    failed = 0
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futs = [ex.submit(train_one, job) for job in jobs]

        with tqdm(total=len(futs), desc="jobs", unit="job") as pbar:
            for fut in as_completed(futs):
                try:
                    r = fut.result()
                    r["ok"] = True
                    completed += 1
                except Exception as e:
                    r = {"ok": False, "error": repr(e)}
                    failed += 1

                r["elapsed_total_seconds"] = time.time() - t0
                with open(result_path, "a") as f:
                    f.write(json.dumps(r) + "\n")

                if not r.get("ok"):
                    print("FAILED:", r)

                pbar.update(1)
                pbar.set_postfix(completed=completed, failed=failed)

    print(f"Batch done. completed={completed}, failed={failed}, elapsed={(time.time()-t0)/3600:.2f}h")


def build_auto_jobs(out_root: Path, seeds: List[int], total_steps: int):
    return [
        {"out_root": str(out_root), "target": t, "seed": s, "mode": "auto", "reward_scale": 1.0,
         "manual_alpha": None, "total_steps": total_steps, "torch_num_threads": 1}
        for t in AUTO_TARGETS
        for s in seeds
    ]


def build_manual_search_jobs(out_root: Path, tuning_seeds: List[int], total_steps: int):
    return [
        {"out_root": str(out_root), "target": t, "seed": s, "mode": "manual", "reward_scale": 1.0,
         "manual_alpha": a, "total_steps": total_steps, "torch_num_threads": 1}
        for t in MANUAL_TARGETS
        for a in ALPHA_GRID
        for s in tuning_seeds
    ]


def build_manual_final_jobs(out_root: Path, seeds: List[int], selected: Dict[float, float], total_steps: int):
    return [
        {"out_root": str(out_root), "target": t, "seed": s, "mode": "manual", "reward_scale": 1.0,
         "manual_alpha": a, "total_steps": total_steps, "torch_num_threads": 1}
        for t, a in selected.items()
        for s in seeds
    ]


def build_scaling_jobs(out_root: Path, seeds: List[int], alpha90: float, total_steps: int):
    jobs = []
    for scale in REWARD_SCALES:
        for s in seeds:
            jobs.append({"out_root": str(out_root), "target": 90.0, "seed": s, "mode": "auto_scaled",
                         "reward_scale": scale, "manual_alpha": None, "total_steps": total_steps,
                         "torch_num_threads": 1})
            jobs.append({"out_root": str(out_root), "target": 90.0, "seed": s, "mode": "manual_scaled",
                         "reward_scale": scale, "manual_alpha": alpha90, "total_steps": total_steps,
                         "torch_num_threads": 1})
    return jobs


# =========================
# Alpha selection + plots
# =========================

def load_all_eval_logs(out_root: Path) -> pd.DataFrame:
    rows = []
    for f in out_root.rglob("eval_log.csv"):
        try:
            df = pd.read_csv(f)
            rows.append(df)
        except Exception:
            pass
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def normalize(s: pd.Series):
    lo, hi = float(s.min()), float(s.max())
    if np.isclose(lo, hi):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def select_alphas(out_root: Path, tuning_seeds: List[int]) -> Dict[float, float]:
    df = load_all_eval_logs(out_root)
    manual = df[
        (df["mode"] == "manual")
        & np.isclose(df["reward_scale"].astype(float), 1.0)
        & df["seed"].astype(int).isin(tuning_seeds)
    ].copy()

    if manual.empty:
        raise RuntimeError("No manual search logs found.")

    selected = {}
    all_rows = []

    for target in MANUAL_TARGETS:
        sub_t = manual[np.isclose(manual["target_angle_deg"].astype(float), target)].copy()
        rows = []

        for alpha, sub_a in sub_t.groupby("manual_alpha"):
            curve = sub_a.groupby("timestep", as_index=False)["eval_return_mean"].mean().sort_values("timestep")
            final_return = float(curve["eval_return_mean"].iloc[-1])
            auc = float(np.trapezoid(curve["eval_return_mean"], curve["timestep"]))
            rows.append({
                "target_angle_deg": target,
                "manual_alpha": float(alpha),
                "n_seeds": int(sub_a["seed"].nunique()),
                "final_return": final_return,
                "auc": auc,
            })

        score = pd.DataFrame(rows)
        score["final_norm"] = normalize(score["final_return"])
        score["auc_norm"] = normalize(score["auc"])
        score["selection_score"] = 0.5 * score["final_norm"] + 0.5 * score["auc_norm"]
        score = score.sort_values(["selection_score", "final_return", "auc"], ascending=False)

        best_alpha = float(score.iloc[0]["manual_alpha"])
        selected[target] = best_alpha
        all_rows.append(score)

    summary = pd.concat(all_rows, ignore_index=True)
    summary.to_csv(out_root / "manual_alpha_selection_table.csv", index=False)

    write_json({
        "selected_alphas": {str(k): v for k, v in selected.items()},
        "selection_rule": "0.5*normalized_final_return + 0.5*normalized_AUC",
        "tuning_seeds": tuning_seeds,
        "torque_limit": TORQUE_LIMIT,
    }, out_root / "selected_manual_alphas.json")

    print("Selected alphas:")
    for k, v in selected.items():
        print(f"  {k:g} -> {v:g}")

    return selected


def mean_ci(df: pd.DataFrame, group_cols: List[str], metric: str):
    per_seed = df.groupby(group_cols + ["seed"], as_index=False)[metric].mean()
    g = per_seed.groupby(group_cols)[metric]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out["ci95"] = 1.96 * out["std"].fillna(0.0) / np.sqrt(out["count"].clip(lower=1))
    return out


def plot_ci(ax, agg, x, label):
    agg = agg.sort_values(x)
    ax.plot(agg[x], agg["mean"], marker="o", markersize=2, linewidth=1.5, label=label)
    ax.fill_between(
        agg[x].to_numpy(),
        (agg["mean"] - agg["ci95"]).to_numpy(),
        (agg["mean"] + agg["ci95"]).to_numpy(),
        alpha=0.18,
    )


def make_plots(out_root: Path, selected: Dict[float, float]):
    df = load_all_eval_logs(out_root)
    if df.empty:
        print("No logs for plots.")
        return

    plots = ensure_dir(out_root / "plots")

    # Auto targets.
    sub = df[(df["mode"] == "auto") & np.isclose(df["reward_scale"].astype(float), 1.0)].copy()
    if not sub.empty:
        agg = mean_ci(sub, ["target_angle_deg", "timestep"], "eval_return_mean")
        fig, ax = plt.subplots(figsize=(10, 6))
        for t in AUTO_TARGETS:
            one = agg[np.isclose(agg["target_angle_deg"].astype(float), t)]
            if not one.empty:
                plot_ci(ax, one, "timestep", f"{t:g}°")
        ax.set_title("Auto α SAC bad targets, torque ±6")
        ax.set_xlabel("Environment timesteps")
        ax.set_ylabel("Mean eval return")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots / "auto_bad_targets_return.png", dpi=200)
        plt.close(fig)

    # Manual vs auto.
    for metric in ["eval_return_mean", "mean_abs_angle_error_deg", "target_occupancy", "alpha"]:
        if metric not in df.columns:
            continue
        fig, axes = plt.subplots(1, len(MANUAL_TARGETS), figsize=(15, 5))
        for ax, t in zip(axes, MANUAL_TARGETS):
            auto = df[
                (df["mode"] == "auto")
                & np.isclose(df["target_angle_deg"].astype(float), t)
                & np.isclose(df["reward_scale"].astype(float), 1.0)
            ]
            if not auto.empty:
                plot_ci(ax, mean_ci(auto, ["timestep"], metric), "timestep", "auto α")

            a = selected.get(t)
            man = df[
                (df["mode"] == "manual")
                & np.isclose(df["target_angle_deg"].astype(float), t)
                & np.isclose(df["manual_alpha"].astype(float), a)
                & np.isclose(df["reward_scale"].astype(float), 1.0)
            ]
            if not man.empty:
                plot_ci(ax, mean_ci(man, ["timestep"], metric), "timestep", f"manual α={a:g}")

            ax.set_title(f"target {t:g}°")
            ax.set_xlabel("timesteps")
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.3)
            ax.legend()

        fig.tight_layout()
        fig.savefig(plots / f"manual_vs_auto_{metric}.png", dpi=200)
        plt.close(fig)

    # Final summary.
    final_rows = []
    for (mode, target, scale, alpha), sub in df.groupby(["mode", "target_angle_deg", "reward_scale", "manual_alpha"], dropna=False):
        step = sub["timestep"].max()
        last = sub[sub["timestep"] == step]
        final_rows.append({
            "mode": mode,
            "target_angle_deg": target,
            "reward_scale": scale,
            "manual_alpha": alpha,
            "final_step": step,
            "n_seeds": int(last["seed"].nunique()),
            "return_mean": float(last["eval_return_mean"].mean()),
            "return_std": float(last["eval_return_mean"].std(ddof=1)),
            "angle_error_mean_deg": float(last["mean_abs_angle_error_deg"].mean()),
            "target_occupancy_mean": float(last["target_occupancy"].mean()),
            "alpha_mean": float(last["alpha"].mean()),
        })

    pd.DataFrame(final_rows).to_csv(out_root / "final_summary.csv", index=False)
    print("Plots and summary saved to", out_root)


# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", default="outputs/pendulum_bad_angles_torque6")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seeds", default="0-14")
    ap.add_argument("--tuning-seeds", default="0,1,2")
    ap.add_argument("--total-steps", type=int, default=TOTAL_STEPS)
    args = ap.parse_args()

    def parse_seeds(s):
        if "-" in s and "," not in s:
            a, b = s.split("-", 1)
            return list(range(int(a), int(b) + 1))
        return [int(x.strip()) for x in s.split(",") if x.strip()]

    seeds = parse_seeds(args.seeds)
    tuning_seeds = parse_seeds(args.tuning_seeds)
    out_root = ensure_dir(args.output_root)

    config = {
        "auto_targets": AUTO_TARGETS,
        "manual_targets": MANUAL_TARGETS,
        "alpha_grid": ALPHA_GRID,
        "reward_scales": REWARD_SCALES,
        "total_steps": args.total_steps,
        "seeds": seeds,
        "tuning_seeds": tuning_seeds,
        "torque_limit": TORQUE_LIMIT,
        "random_steps": RANDOM_STEPS,
        "learning_starts": LEARNING_STARTS,
        "eval_every": EVAL_EVERY,
        "eval_episodes": EVAL_EPISODES,
        "gamma": GAMMA,
        "tau": TAU,
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "replay_size": REPLAY_SIZE,
        "hidden": HIDDEN,
        "init_alpha": INIT_ALPHA,
        "target_entropy": TARGET_ENTROPY,
        "reward": {
            "angle_weight": ANGLE_W,
            "angular_velocity_weight": OMEGA_W,
            "torque_weight": TORQUE_W,
        },
    }
    write_json(config, out_root / "run_config.json")

    print("========== BAD ANGLE TORQUE ±6 RUN ==========")
    print("Output:", out_root)
    print("Workers:", args.workers)
    print("Device:", DEVICE)
    print("Total steps:", args.total_steps)
    print("Seeds:", seeds)
    print("Tuning seeds:", tuning_seeds)
    print("Torque limit:", TORQUE_LIMIT)

    phase1 = build_auto_jobs(out_root, seeds, args.total_steps) + build_manual_search_jobs(out_root, tuning_seeds, args.total_steps)
    print("\n========== PHASE 1: auto + manual alpha search ==========")
    print("Phase 1 jobs:", len(phase1))
    run_jobs(phase1, args.workers, out_root / "phase1_results.jsonl")

    print("\n========== SELECTING ALPHAS ==========")
    selected = select_alphas(out_root, tuning_seeds)

    print("\n========== PHASE 2: manual final + reward scaling ==========")
    phase2 = build_manual_final_jobs(out_root, seeds, selected, args.total_steps)

    if 90.0 not in selected:
        raise RuntimeError("No selected alpha for 90.")
    phase2 += build_scaling_jobs(out_root, seeds, selected[90.0], args.total_steps)

    print("Phase 2 jobs:", len(phase2))
    run_jobs(phase2, args.workers, out_root / "phase2_results.jsonl")

    print("\n========== PLOTS ==========")
    make_plots(out_root, selected)

    print("\nDONE.")
    print("Output saved to:", out_root)


if __name__ == "__main__":
    main()