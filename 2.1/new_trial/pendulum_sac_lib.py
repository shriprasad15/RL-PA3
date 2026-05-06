"""
Self-contained Pendulum-v1 SAC utilities for PA3.

This is a compact, documented SAC implementation matching the assignment requirements:
- squashed Gaussian policy with tanh
- clipped double Q-learning
- reparameterization trick
- no separate state-value function
- Adam optimizers
- optional automatic entropy temperature tuning

The code is written as a module so multiprocessing works from notebooks on Linux,
macOS, and Windows.
"""
from __future__ import annotations

import copy
import json
import math
import os
import pickle
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import gymnasium as gym
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from tqdm.auto import tqdm


# -----------------------------
# General utilities
# -----------------------------


def set_global_seeds(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Determinism is nice for reporting, but can reduce speed a little.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def wrap_to_pi(angle: np.ndarray | float) -> np.ndarray | float:
    """Map angle(s) to [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def safe_angle_label(theta_deg: float | int) -> str:
    sign = "p" if theta_deg >= 0 else "m"
    return f"{sign}{abs(int(theta_deg)):03d}"


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_yaml(obj: Dict[str, Any], path: str | Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def atomic_write_json(obj: Dict[str, Any], path: str | Path) -> None:
    """Write JSON safely so the parent monitor never sees a half-written file."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


# -----------------------------
# Modified Pendulum environment
# -----------------------------


class TargetPendulumRewardWrapper(gym.Wrapper):
    """
    Pendulum-v1 reward wrapper for arbitrary target angle.

    Gymnasium Pendulum's internal state angle theta=0 corresponds to upright.
    We keep environment dynamics and default torque limits unchanged, and only
    replace the reward with:
        r = -scale * (w_theta * wrap(theta - target)^2
                      + w_omega * theta_dot^2
                      + w_u * torque^2)
    """

    def __init__(
        self,
        env: gym.Env,
        target_angle_deg: float,
        angle_weight: float = 1.0,
        angular_velocity_weight: float = 0.1,
        torque_weight: float = 0.001,
        reward_scale: float = 1.0,
    ) -> None:
        super().__init__(env)
        self.target_angle_deg = float(target_angle_deg)
        self.target_angle_rad = math.radians(float(target_angle_deg))
        self.angle_weight = float(angle_weight)
        self.angular_velocity_weight = float(angular_velocity_weight)
        self.torque_weight = float(torque_weight)
        self.reward_scale = float(reward_scale)

    def step(self, action):
        # Mirror original Pendulum reward timing: compute reward from state before transition.
        theta, theta_dot = self.unwrapped.state
        clipped_action = np.clip(action, self.action_space.low, self.action_space.high)
        torque = float(np.asarray(clipped_action).reshape(-1)[0])
        angle_error = float(wrap_to_pi(theta - self.target_angle_rad))
        base_cost = (
            self.angle_weight * angle_error**2
            + self.angular_velocity_weight * float(theta_dot) ** 2
            + self.torque_weight * torque**2
        )
        reward = -self.reward_scale * base_cost
        obs, _orig_reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info.update(
            {
                "theta_rad": float(wrap_to_pi(theta)),
                "theta_deg": float(np.degrees(wrap_to_pi(theta))),
                "target_angle_rad": self.target_angle_rad,
                "target_angle_deg": self.target_angle_deg,
                "angle_error_rad": angle_error,
                "angle_error_deg": float(np.degrees(angle_error)),
                "abs_angle_error_rad": abs(angle_error),
                "abs_angle_error_deg": abs(float(np.degrees(angle_error))),
                "theta_dot": float(theta_dot),
                "torque": torque,
                "torque_sq": torque**2,
                "base_cost": base_cost,
                "custom_reward": reward,
            }
        )
        return obs, float(reward), terminated, truncated, info


def make_target_pendulum_env(
    cfg: Dict[str, Any],
    target_angle_deg: float,
    reward_scale: float = 1.0,
    seed: Optional[int] = None,
    render_mode: Optional[str] = None,
) -> gym.Env:
    env = gym.make(
        cfg["experiment"]["env_id"],
        max_episode_steps=int(cfg["experiment"]["episode_length"]),
        render_mode=render_mode,
    )
    if seed is not None:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    r_cfg = cfg["reward"]
    env = TargetPendulumRewardWrapper(
        env,
        target_angle_deg=target_angle_deg,
        angle_weight=float(r_cfg["angle_weight"]),
        angular_velocity_weight=float(r_cfg["angular_velocity_weight"]),
        torque_weight=float(r_cfg["torque_weight"]),
        reward_scale=float(reward_scale),
    )
    return env


# -----------------------------
# Replay buffer
# -----------------------------


class ReplayBuffer:
    def __init__(self, obs_dim: int, action_dim: int, capacity: int, seed: int = 0):
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.idx = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.idx] = np.asarray(obs, dtype=np.float32)
        self.actions[self.idx] = np.asarray(action, dtype=np.float32)
        self.rewards[self.idx] = float(reward)
        self.next_obs[self.idx] = np.asarray(next_obs, dtype=np.float32)
        self.dones[self.idx] = float(done)
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device):
        indices = self.rng.integers(0, self.size, size=int(batch_size))
        return (
            torch.as_tensor(self.obs[indices], device=device),
            torch.as_tensor(self.actions[indices], device=device),
            torch.as_tensor(self.rewards[indices], device=device),
            torch.as_tensor(self.next_obs[indices], device=device),
            torch.as_tensor(self.dones[indices], device=device),
        )

    def __len__(self):
        return self.size


# -----------------------------
# SAC neural networks
# -----------------------------


LOG_STD_MIN_DEFAULT = -20
LOG_STD_MAX_DEFAULT = 2
EPS = 1e-6


def mlp_body(input_dim: int, hidden_layers: Iterable[int], activation=nn.ReLU) -> Tuple[nn.Sequential, int]:
    layers: List[nn.Module] = []
    last_dim = input_dim
    for hidden_dim in hidden_layers:
        layers.append(nn.Linear(last_dim, int(hidden_dim)))
        layers.append(activation())
        last_dim = int(hidden_dim)
    return nn.Sequential(*layers), last_dim


def mlp(input_dim: int, hidden_layers: Iterable[int], output_dim: int, activation=nn.ReLU) -> nn.Sequential:
    body, last_dim = mlp_body(input_dim, hidden_layers, activation=activation)
    return nn.Sequential(*list(body.children()), nn.Linear(last_dim, output_dim))


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_layers: List[int]):
        super().__init__()
        self.net = mlp(obs_dim + action_dim, hidden_layers, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1))


class SquashedGaussianPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_layers: List[int],
        action_space: gym.Space,
        log_std_min: float = LOG_STD_MIN_DEFAULT,
        log_std_max: float = LOG_STD_MAX_DEFAULT,
    ):
        super().__init__()
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.backbone, last_dim = mlp_body(obs_dim, hidden_layers)
        self.mean_linear = nn.Linear(last_dim, action_dim)
        self.log_std_linear = nn.Linear(last_dim, action_dim)

        action_scale = (action_space.high - action_space.low) / 2.0
        action_bias = (action_space.high + action_space.low) / 2.0
        self.register_buffer("action_scale", torch.as_tensor(action_scale, dtype=torch.float32))
        self.register_buffer("action_bias", torch.as_tensor(action_bias, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(obs)
        mean = self.mean_linear(h)
        log_std = self.log_std_linear(h)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        # Change-of-variables correction for tanh plus affine action scaling.
        log_prob = normal.log_prob(x_t)
        correction = torch.log(self.action_scale * (1.0 - y_t.pow(2)) + EPS)
        log_prob = (log_prob - correction).sum(dim=-1, keepdim=True)

        deterministic = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, deterministic


class SACAgent:
    def __init__(self, obs_dim: int, action_dim: int, action_space: gym.Space, cfg: Dict[str, Any], device: str | torch.device):
        self.cfg = cfg
        self.device = torch.device(device)
        h = list(cfg["network"]["hidden_layers"])
        sac_cfg = cfg["sac"]
        opt_cfg = cfg["optimizer"]
        ent_cfg = cfg["entropy"]

        self.gamma = float(sac_cfg["gamma"])
        self.tau = float(sac_cfg["tau"])
        self.automatic_alpha_tuning = bool(ent_cfg["automatic_alpha_tuning"])
        self.target_entropy = float(ent_cfg.get("target_entropy", -float(action_dim)))

        self.actor = SquashedGaussianPolicy(
            obs_dim,
            action_dim,
            h,
            action_space,
            log_std_min=float(cfg["network"]["actor_log_std_min"]),
            log_std_max=float(cfg["network"]["actor_log_std_max"]),
        ).to(self.device)
        self.q1 = QNetwork(obs_dim, action_dim, h).to(self.device)
        self.q2 = QNetwork(obs_dim, action_dim, h).to(self.device)
        self.q1_target = copy.deepcopy(self.q1).to(self.device)
        self.q2_target = copy.deepcopy(self.q2).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=float(opt_cfg["actor_lr"]))
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=float(opt_cfg["critic_lr"]))
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=float(opt_cfg["critic_lr"]))

        if self.automatic_alpha_tuning:
            init_alpha = float(ent_cfg["initial_alpha"])
            self.log_alpha = torch.tensor(math.log(init_alpha), requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=float(opt_cfg["alpha_lr"]))
        else:
            self.log_alpha = None
            self.alpha_optimizer = None
            self.fixed_alpha = float(ent_cfg["initial_alpha"])

        self.total_updates = 0

    @property
    def alpha(self) -> torch.Tensor:
        if self.automatic_alpha_tuning:
            return self.log_alpha.exp()
        return torch.tensor(self.fixed_alpha, device=self.device)

    def select_action(self, obs: np.ndarray, evaluate: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _logp, deterministic = self.actor.sample(obs_t)
        out = deterministic if evaluate else action
        return out.cpu().numpy()[0]

    def update(self, replay_buffer: ReplayBuffer, batch_size: int) -> Dict[str, float]:
        obs, actions, rewards, next_obs, dones = replay_buffer.sample(batch_size, self.device)

        with torch.no_grad():
            next_actions, next_logp, _ = self.actor.sample(next_obs)
            target_q1 = self.q1_target(next_obs, next_actions)
            target_q2 = self.q2_target(next_obs, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha.detach() * next_logp
            y = rewards + (1.0 - dones) * self.gamma * target_q

        q1_pred = self.q1(obs, actions)
        q2_pred = self.q2(obs, actions)
        q1_loss = F.mse_loss(q1_pred, y)
        q2_loss = F.mse_loss(q2_pred, y)

        self.q1_optimizer.zero_grad(set_to_none=True)
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad(set_to_none=True)
        q2_loss.backward()
        self.q2_optimizer.step()

        new_actions, logp, _ = self.actor.sample(obs)
        q_new = torch.min(self.q1(obs, new_actions), self.q2(obs, new_actions))
        actor_loss = (self.alpha.detach() * logp - q_new).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.automatic_alpha_tuning:
            alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha_loss_value = float(alpha_loss.item())
        else:
            alpha_loss_value = 0.0

        self.soft_update_targets()
        self.total_updates += 1

        return {
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": alpha_loss_value,
            "alpha": float(self.alpha.detach().cpu().item()),
        }

    def soft_update_targets(self) -> None:
        with torch.no_grad():
            for p, tp in zip(self.q1.parameters(), self.q1_target.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
            for p, tp in zip(self.q2.parameters(), self.q2_target.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "q1_optimizer": self.q1_optimizer.state_dict(),
            "q2_optimizer": self.q2_optimizer.state_dict(),
            "automatic_alpha_tuning": self.automatic_alpha_tuning,
            "log_alpha": None if self.log_alpha is None else self.log_alpha.detach().cpu(),
            "fixed_alpha": None if self.automatic_alpha_tuning else self.fixed_alpha,
            "total_updates": self.total_updates,
            "cfg": self.cfg,
        }

    def save(self, path: str | Path) -> None:
        torch.save(self.state_dict(), path)

    def load_actor_only(self, path: str | Path, map_location=None) -> None:
        ckpt = torch.load(path, map_location=map_location or self.device)
        self.actor.load_state_dict(ckpt["actor"])


def build_agent_for_env(env: gym.Env, cfg: Dict[str, Any], device: str | torch.device) -> SACAgent:
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    return SACAgent(obs_dim, action_dim, env.action_space, cfg, device)


# -----------------------------
# Evaluation and rollout
# -----------------------------


def evaluate_policy(
    agent: SACAgent,
    cfg: Dict[str, Any],
    target_angle_deg: float,
    reward_scale: float,
    eval_episodes: int,
    seed: int,
) -> Dict[str, float]:
    env = make_target_pendulum_env(cfg, target_angle_deg, reward_scale, seed=seed, render_mode=None)
    returns, lengths = [], []
    abs_angle_errors, angular_velocities, torque_sqs, occupancies = [], [], [], []
    tol_rad = math.radians(float(cfg.get("analysis", {}).get("target_tolerance_deg", 10.0)))

    for ep in range(int(eval_episodes)):
        obs, _ = env.reset(seed=seed + 10_000 + ep)
        done = False
        ep_return = 0.0
        ep_len = 0
        while not done:
            action = agent.select_action(obs, evaluate=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            ep_return += float(reward)
            ep_len += 1
            abs_angle_errors.append(float(info["abs_angle_error_rad"]))
            angular_velocities.append(abs(float(info["theta_dot"])))
            torque_sqs.append(float(info["torque_sq"]))
            occupancies.append(float(info["abs_angle_error_rad"] < tol_rad))
        returns.append(ep_return)
        lengths.append(ep_len)
    env.close()

    return {
        "eval_return_mean": float(np.mean(returns)),
        "eval_return_std_over_episodes": float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0,
        "eval_length_mean": float(np.mean(lengths)),
        "mean_abs_angle_error_rad": float(np.mean(abs_angle_errors)),
        "mean_abs_angle_error_deg": float(np.degrees(np.mean(abs_angle_errors))),
        "mean_abs_angular_velocity": float(np.mean(angular_velocities)),
        "mean_squared_torque": float(np.mean(torque_sqs)),
        "target_occupancy": float(np.mean(occupancies)),
    }


def collect_rollout(
    agent: SACAgent,
    cfg: Dict[str, Any],
    target_angle_deg: float,
    reward_scale: float,
    seed: int,
    deterministic: bool = True,
) -> Dict[str, Any]:
    env = make_target_pendulum_env(cfg, target_angle_deg, reward_scale, seed=seed, render_mode=None)
    obs, _ = env.reset(seed=seed)
    rollout: Dict[str, Any] = {"obs": [], "actions": [], "rewards": [], "infos": []}
    done = False
    cumulative_reward = 0.0
    while not done:
        action = agent.select_action(obs, evaluate=deterministic)
        next_obs, reward, terminated, truncated, info = env.step(action)
        cumulative_reward += float(reward)
        info = dict(info)
        info["cumulative_reward"] = cumulative_reward
        rollout["obs"].append(np.asarray(obs, dtype=np.float32))
        rollout["actions"].append(np.asarray(action, dtype=np.float32))
        rollout["rewards"].append(float(reward))
        rollout["infos"].append(info)
        obs = next_obs
        done = bool(terminated or truncated)
    env.close()
    rollout["return"] = cumulative_reward
    rollout["length"] = len(rollout["rewards"])
    return rollout


# -----------------------------
# Training one seed/run
# -----------------------------


def make_run_cfg(
    base_cfg: Dict[str, Any],
    target_angle_deg: float,
    seed: int,
    mode: str,
    reward_scale: float = 1.0,
    manual_alpha: Optional[float] = None,
    total_timesteps: Optional[int] = None,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg["run"] = {
        "target_angle_deg": float(target_angle_deg),
        "seed": int(seed),
        "mode": mode,
        "reward_scale": float(reward_scale),
        "manual_alpha": None if manual_alpha is None else float(manual_alpha),
    }
    cfg["reward"]["reward_scale"] = float(reward_scale)
    if total_timesteps is not None:
        cfg["experiment"]["total_timesteps"] = int(total_timesteps)

    if mode.startswith("auto"):
        cfg["entropy"]["automatic_alpha_tuning"] = True
    else:
        if manual_alpha is None:
            raise ValueError("manual_alpha must be provided for manual mode")
        cfg["entropy"]["automatic_alpha_tuning"] = False
        cfg["entropy"]["initial_alpha"] = float(manual_alpha)
    return cfg


def condition_name(mode: str, target_angle_deg: float, reward_scale: float = 1.0, manual_alpha: Optional[float] = None) -> str:
    target = safe_angle_label(target_angle_deg)
    if mode.startswith("auto"):
        return f"auto_target_{target}_scale_{reward_scale:g}"
    alpha_txt = str(manual_alpha).replace(".", "p") if manual_alpha is not None else "unknown"
    return f"manual_target_{target}_alpha_{alpha_txt}_scale_{reward_scale:g}"


def train_one_run(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train exactly one (target, seed, condition) run.

    job keys:
        base_cfg, output_root, target_angle_deg, seed, mode, reward_scale, manual_alpha, device
    """
    base_cfg = job["base_cfg"]
    output_root = Path(job["output_root"])
    target_angle_deg = float(job["target_angle_deg"])
    seed = int(job["seed"])
    mode = str(job["mode"])
    reward_scale = float(job.get("reward_scale", 1.0))
    manual_alpha = job.get("manual_alpha", None)
    device = job.get("device", "cpu")
    total_timesteps = job.get("total_timesteps", None)

    cfg = make_run_cfg(base_cfg, target_angle_deg, seed, mode, reward_scale, manual_alpha, total_timesteps)
    cond = condition_name(mode, target_angle_deg, reward_scale, manual_alpha)
    seed_dir = ensure_dir(output_root / cond / f"seed_{seed:02d}")
    save_yaml(cfg, seed_dir / "config.yaml")
    checkpoints_dir = ensure_dir(seed_dir / "checkpoints")

    total_steps_for_status = int(cfg["experiment"]["total_timesteps"])
    status_path = seed_dir / "status.json"
    status_every_steps = int(job.get("status_every_steps", cfg.get("runtime", {}).get("status_every_steps", 2000)))
    status_every_steps = max(1, status_every_steps)
    pid = os.getpid()
    last_eval_return_for_status = None
    best_eval_return_for_status = None
    run_started_for_status = time.time()

    def write_status(step: int, phase: str, extra: Optional[Dict[str, Any]] = None) -> None:
        payload: Dict[str, Any] = {
            "condition": cond,
            "seed": seed,
            "target_angle_deg": target_angle_deg,
            "reward_scale": reward_scale,
            "mode": mode,
            "manual_alpha": None if manual_alpha is None else float(manual_alpha),
            "current_step": int(step),
            "total_timesteps": int(total_steps_for_status),
            "progress_fraction": float(int(step) / max(1, int(total_steps_for_status))),
            "phase": phase,
            "pid": int(pid),
            "updated_at_unix": time.time(),
            "elapsed_seconds": time.time() - run_started_for_status,
            "last_eval_return": last_eval_return_for_status,
            "best_eval_return": best_eval_return_for_status,
        }
        if extra:
            payload.update(extra)
        atomic_write_json(payload, status_path)

    # If finished marker exists, skip safely. This helps resume interrupted sweeps.
    done_marker = seed_dir / "DONE.txt"
    if done_marker.exists() and (seed_dir / "final_model.pt").exists() and (seed_dir / "eval_log.csv").exists():
        write_status(total_steps_for_status, "skipped_done")
        return {"status": "skipped_done", "seed_dir": str(seed_dir), "condition": cond, "seed": seed}

    # Prevent CPU oversubscription when many SAC runs are launched in parallel.
    # Without this, each worker may use many BLAS/PyTorch threads and the whole PC can lag.
    torch_threads = int(job.get("torch_num_threads", 1))
    try:
        torch.set_num_threads(max(1, torch_threads))
    except Exception:
        pass

    write_status(0, "initializing")
    set_global_seeds(seed)
    env = make_target_pendulum_env(cfg, target_angle_deg, reward_scale, seed=seed, render_mode=None)
    agent = build_agent_for_env(env, cfg, device=device)
    replay = ReplayBuffer(
        obs_dim=int(np.prod(env.observation_space.shape)),
        action_dim=int(np.prod(env.action_space.shape)),
        capacity=int(cfg["sac"]["replay_buffer_size"]),
        seed=seed,
    )

    total_steps = int(cfg["experiment"]["total_timesteps"])
    eval_every = int(cfg["experiment"]["eval_every"])
    eval_episodes = int(cfg["experiment"]["eval_episodes"])
    batch_size = int(cfg["sac"]["batch_size"])
    random_steps = int(cfg["sac"]["random_exploration_steps"])
    learning_starts = int(cfg["sac"]["learning_starts"])
    updates_per_step = int(cfg["sac"].get("updates_per_step", 1))

    train_rows: List[Dict[str, Any]] = []
    eval_rows: List[Dict[str, Any]] = []
    update_acc: Dict[str, List[float]] = {"q1_loss": [], "q2_loss": [], "actor_loss": [], "alpha_loss": [], "alpha": []}

    # Required: evaluate untrained policy at timestep 0.
    start_eval = evaluate_policy(agent, cfg, target_angle_deg, reward_scale, eval_episodes, seed=seed)
    start_eval.update(
        {
            "timestep": 0,
            "seed": seed,
            "target_angle_deg": target_angle_deg,
            "reward_scale": reward_scale,
            "mode": mode,
            "manual_alpha": np.nan if manual_alpha is None else float(manual_alpha),
            "alpha": float(agent.alpha.detach().cpu().item()),
            "condition": cond,
        }
    )
    eval_rows.append(start_eval)
    pd.DataFrame(eval_rows).to_csv(seed_dir / "eval_log.csv", index=False)
    best_eval_return = start_eval["eval_return_mean"]
    last_eval_return_for_status = float(start_eval["eval_return_mean"])
    best_eval_return_for_status = float(best_eval_return)
    write_status(0, "eval_t0_done", {"alpha": float(agent.alpha.detach().cpu().item())})
    agent.save(seed_dir / "best_model.pt")
    if bool(cfg.get("artifacts", {}).get("save_eval_checkpoints", True)):
        agent.save(checkpoints_dir / "model_step_000000.pt")

    obs, _ = env.reset(seed=seed)
    episode_return, episode_length = 0.0, 0
    run_started = time.time()

    pbar = tqdm(total=total_steps, desc=f"{cond}/seed_{seed:02d}", position=0, leave=False, disable=bool(job.get("disable_tqdm", True)))
    for step in range(1, total_steps + 1):
        if step <= random_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs, evaluate=False)

        next_obs, reward, terminated, truncated, info = env.step(action)
        # Pendulum has no true terminal state. Do not treat time-limit truncation as terminal for bootstrapping.
        done_for_buffer = float(terminated)
        replay.add(obs, action, reward, next_obs, done_for_buffer)

        obs = next_obs
        episode_return += float(reward)
        episode_length += 1

        if step >= learning_starts and len(replay) >= batch_size:
            for _ in range(updates_per_step):
                losses = agent.update(replay, batch_size)
                for k, v in losses.items():
                    update_acc[k].append(v)

        if terminated or truncated:
            train_rows.append(
                {
                    "timestep": step,
                    "seed": seed,
                    "target_angle_deg": target_angle_deg,
                    "reward_scale": reward_scale,
                    "mode": mode,
                    "manual_alpha": np.nan if manual_alpha is None else float(manual_alpha),
                    "condition": cond,
                    "episode_return": episode_return,
                    "episode_length": episode_length,
                    "alpha": float(agent.alpha.detach().cpu().item()),
                }
            )
            obs, _ = env.reset(seed=seed + step)
            episode_return, episode_length = 0.0, 0

        if step == 1 or step == random_steps or step == learning_starts or step % status_every_steps == 0:
            write_status(
                step,
                "training",
                {
                    "alpha": float(agent.alpha.detach().cpu().item()),
                    "episodes_logged": len(train_rows),
                    "replay_size": len(replay),
                },
            )

        if step % eval_every == 0:
            write_status(step, "evaluating", {"alpha": float(agent.alpha.detach().cpu().item()), "episodes_logged": len(train_rows), "replay_size": len(replay)})
            eval_stats = evaluate_policy(agent, cfg, target_angle_deg, reward_scale, eval_episodes, seed=seed + step)
            mean_update = {f"mean_{k}_since_last_eval": (float(np.mean(v)) if len(v) else np.nan) for k, v in update_acc.items()}
            update_acc = {"q1_loss": [], "q2_loss": [], "actor_loss": [], "alpha_loss": [], "alpha": []}
            eval_stats.update(
                {
                    "timestep": step,
                    "seed": seed,
                    "target_angle_deg": target_angle_deg,
                    "reward_scale": reward_scale,
                    "mode": mode,
                    "manual_alpha": np.nan if manual_alpha is None else float(manual_alpha),
                    "alpha": float(agent.alpha.detach().cpu().item()),
                    "condition": cond,
                    "wallclock_seconds": time.time() - run_started,
                    **mean_update,
                }
            )
            eval_rows.append(eval_stats)
            last_eval_return_for_status = float(eval_stats["eval_return_mean"])
            pd.DataFrame(eval_rows).to_csv(seed_dir / "eval_log.csv", index=False)
            pd.DataFrame(train_rows).to_csv(seed_dir / "train_log.csv", index=False)
            write_status(
                step,
                "eval_done",
                {
                    "alpha": float(agent.alpha.detach().cpu().item()),
                    "episodes_logged": len(train_rows),
                    "replay_size": len(replay),
                    "last_eval_return": last_eval_return_for_status,
                    "mean_abs_angle_error_deg": float(eval_stats.get("mean_abs_angle_error_deg", np.nan)),
                    "target_occupancy": float(eval_stats.get("target_occupancy", np.nan)),
                },
            )

            if bool(cfg.get("artifacts", {}).get("save_eval_checkpoints", True)):
                agent.save(checkpoints_dir / f"model_step_{step:06d}.pt")

            if eval_stats["eval_return_mean"] > best_eval_return:
                best_eval_return = eval_stats["eval_return_mean"]
                best_eval_return_for_status = float(best_eval_return)
                agent.save(seed_dir / "best_model.pt")

        pbar.update(1)

    pbar.close()
    agent.save(seed_dir / "final_model.pt")
    pd.DataFrame(train_rows).to_csv(seed_dir / "train_log.csv", index=False)
    pd.DataFrame(eval_rows).to_csv(seed_dir / "eval_log.csv", index=False)

    final_rollout = collect_rollout(agent, cfg, target_angle_deg, reward_scale, seed=seed + 999_999, deterministic=True)
    with open(seed_dir / "final_rollout.pkl", "wb") as f:
        pickle.dump(final_rollout, f)

    with open(done_marker, "w") as f:
        f.write(f"completed_at_unix={time.time()}\n")

    write_status(total_steps_for_status, "completed", {"alpha": float(agent.alpha.detach().cpu().item()), "best_eval_return": float(best_eval_return)})

    env.close()
    return {
        "status": "completed",
        "seed_dir": str(seed_dir),
        "condition": cond,
        "seed": seed,
        "best_eval_return": best_eval_return,
        "wallclock_seconds": time.time() - run_started,
    }


# -----------------------------
# Job builders
# -----------------------------


def build_auto_jobs(base_cfg: Dict[str, Any], output_root: str | Path, target_angles: List[float], device: str = "cpu") -> List[Dict[str, Any]]:
    jobs = []
    for theta in target_angles:
        for seed in base_cfg["experiment"]["seeds"]:
            jobs.append(
                {
                    "base_cfg": base_cfg,
                    "output_root": str(output_root),
                    "target_angle_deg": float(theta),
                    "seed": int(seed),
                    "mode": "auto",
                    "reward_scale": float(base_cfg["reward"].get("reward_scale", 1.0)),
                    "manual_alpha": None,
                    "device": device,
                }
            )
    return jobs


def build_manual_candidate_jobs(
    base_cfg: Dict[str, Any], output_root: str | Path, target_angles: List[float], candidate_alphas: List[float], device: str = "cpu"
) -> List[Dict[str, Any]]:
    jobs = []
    for theta in target_angles:
        for alpha in candidate_alphas:
            for seed in base_cfg["experiment"]["seeds"]:
                jobs.append(
                    {
                        "base_cfg": base_cfg,
                        "output_root": str(output_root),
                        "target_angle_deg": float(theta),
                        "seed": int(seed),
                        "mode": "manual",
                        "reward_scale": 1.0,
                        "manual_alpha": float(alpha),
                        "device": device,
                    }
                )
    return jobs


def choose_best_manual_alphas(eval_df: pd.DataFrame, selected_targets: List[float]) -> Dict[float, float]:
    """Choose alpha with best mean final return over seeds for each target."""
    manual = eval_df[(eval_df["mode"] == "manual") & (eval_df["reward_scale"] == 1.0)].copy()
    out: Dict[float, float] = {}
    for theta in selected_targets:
        sub = manual[np.isclose(manual["target_angle_deg"], float(theta))]
        if sub.empty:
            continue
        final_step = sub["timestep"].max()
        final = sub[sub["timestep"] == final_step]
        scores = final.groupby("manual_alpha", as_index=False)["eval_return_mean"].mean()
        best_row = scores.sort_values("eval_return_mean", ascending=False).iloc[0]
        out[float(theta)] = float(best_row["manual_alpha"])
    return out


def build_reward_scaling_jobs(
    base_cfg: Dict[str, Any],
    output_root: str | Path,
    target_angle_deg: float,
    reward_scales: List[float],
    manual_alpha: float,
    device: str = "cpu",
) -> List[Dict[str, Any]]:
    jobs = []
    for scale in reward_scales:
        for mode, alpha in [("auto_scaled", None), ("manual_scaled", manual_alpha)]:
            for seed in base_cfg["experiment"]["seeds"]:
                jobs.append(
                    {
                        "base_cfg": base_cfg,
                        "output_root": str(output_root),
                        "target_angle_deg": float(target_angle_deg),
                        "seed": int(seed),
                        "mode": mode,
                        "reward_scale": float(scale),
                        "manual_alpha": None if alpha is None else float(alpha),
                        "device": device,
                    }
                )
    return jobs


# -----------------------------
# Log loading and plotting
# -----------------------------


def load_all_eval_logs(output_root: str | Path) -> pd.DataFrame:
    rows = []
    for csv_path in Path(output_root).glob("*/seed_*/eval_log.csv"):
        try:
            df = pd.read_csv(csv_path)
            df["seed_dir"] = str(csv_path.parent)
            rows.append(df)
        except Exception as e:
            print(f"Skipping {csv_path}: {e}")
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def load_all_train_logs(output_root: str | Path) -> pd.DataFrame:
    rows = []
    for csv_path in Path(output_root).glob("*/seed_*/train_log.csv"):
        try:
            df = pd.read_csv(csv_path)
            df["seed_dir"] = str(csv_path.parent)
            rows.append(df)
        except Exception as e:
            print(f"Skipping {csv_path}: {e}")
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def mean_ci_over_seeds(df: pd.DataFrame, group_cols: List[str], metric: str) -> pd.DataFrame:
    # First average duplicate rows per seed if any, then compute CI over seeds.
    per_seed = df.groupby(group_cols + ["seed"], as_index=False)[metric].mean()
    g = per_seed.groupby(group_cols)[metric]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out["ci95"] = 1.96 * out["std"].fillna(0.0) / np.sqrt(out["count"].clip(lower=1))
    return out


def plot_curve_with_ci(ax, df: pd.DataFrame, x: str, y: str, label: str):
    df = df.sort_values(x)
    ax.plot(df[x], df["mean"], label=label)
    ax.fill_between(df[x], df["mean"] - df["ci95"], df["mean"] + df["ci95"], alpha=0.2)


def plot_auto_all_targets(eval_df: pd.DataFrame, plots_dir: str | Path, metric: str = "eval_return_mean") -> Path:
    plots_dir = ensure_dir(plots_dir)
    sub = eval_df[(eval_df["mode"] == "auto") & (np.isclose(eval_df["reward_scale"], 1.0))].copy()
    agg = mean_ci_over_seeds(sub, ["target_angle_deg", "timestep"], metric)
    fig, ax = plt.subplots(figsize=(11, 7))
    for theta in sorted(agg["target_angle_deg"].unique()):
        one = agg[np.isclose(agg["target_angle_deg"], theta)]
        plot_curve_with_ci(ax, one, "timestep", metric, label=f"{theta:g}°")
    ax.set_title(f"Auto-α SAC on modified Pendulum-v1: {metric}")
    ax.set_xlabel("Environment timesteps")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)
    ax.legend(title="target")
    out = Path(plots_dir) / f"auto_all_targets_{metric}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_manual_vs_auto(eval_df: pd.DataFrame, plots_dir: str | Path, selected_targets: List[float], best_manual_alphas: Dict[float, float], metric: str = "eval_return_mean") -> List[Path]:
    plots_dir = ensure_dir(plots_dir)
    outputs = []
    for theta in selected_targets:
        fig, ax = plt.subplots(figsize=(10, 6))
        auto = eval_df[(eval_df["mode"] == "auto") & np.isclose(eval_df["target_angle_deg"], theta) & np.isclose(eval_df["reward_scale"], 1.0)]
        if not auto.empty:
            agg = mean_ci_over_seeds(auto, ["timestep"], metric)
            plot_curve_with_ci(ax, agg, "timestep", metric, label="auto α")
        if theta in best_manual_alphas:
            alpha = best_manual_alphas[theta]
            man = eval_df[(eval_df["mode"] == "manual") & np.isclose(eval_df["target_angle_deg"], theta) & np.isclose(eval_df["manual_alpha"], alpha) & np.isclose(eval_df["reward_scale"], 1.0)]
            if not man.empty:
                agg = mean_ci_over_seeds(man, ["timestep"], metric)
                plot_curve_with_ci(ax, agg, "timestep", metric, label=f"manual α={alpha:g}")
        ax.set_title(f"Manual vs auto α at target {theta:g}°: {metric}")
        ax.set_xlabel("Environment timesteps")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.legend()
        out = Path(plots_dir) / f"manual_vs_auto_target_{safe_angle_label(theta)}_{metric}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=200)
        plt.close(fig)
        outputs.append(out)
    return outputs


def plot_reward_scaling(eval_df: pd.DataFrame, plots_dir: str | Path, target_angle_deg: float = 90.0, metric: str = "eval_return_mean") -> Path:
    plots_dir = ensure_dir(plots_dir)
    sub = eval_df[np.isclose(eval_df["target_angle_deg"], target_angle_deg) & (eval_df["mode"].isin(["auto_scaled", "manual_scaled"]))].copy()
    if sub.empty:
        raise ValueError("No reward-scaling runs found. Run reward scaling jobs first.")
    fig, ax = plt.subplots(figsize=(11, 7))
    group_cols = ["mode", "reward_scale", "timestep"]
    agg = mean_ci_over_seeds(sub, group_cols, metric)
    for (mode, scale), one in agg.groupby(["mode", "reward_scale"]):
        label = f"{mode.replace('_scaled', '')}, scale={scale:g}"
        plot_curve_with_ci(ax, one, "timestep", metric, label=label)
    ax.set_title(f"Reward scaling at target {target_angle_deg:g}°: {metric}")
    ax.set_xlabel("Environment timesteps")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)
    ax.legend()
    out = Path(plots_dir) / f"reward_scaling_target_{safe_angle_label(target_angle_deg)}_{metric}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_extra_metric_family(eval_df: pd.DataFrame, plots_dir: str | Path, mode_filter: str = "auto") -> List[Path]:
    metrics = [
        "mean_abs_angle_error_deg",
        "target_occupancy",
        "mean_abs_angular_velocity",
        "mean_squared_torque",
        "alpha",
    ]
    outputs = []
    for metric in metrics:
        if metric not in eval_df.columns:
            continue
        try:
            outputs.append(plot_auto_all_targets(eval_df, plots_dir, metric=metric))
        except Exception as e:
            print(f"Could not plot {metric}: {e}")
    return outputs


def plot_final_summary_bars(eval_df: pd.DataFrame, plots_dir: str | Path, metric: str = "eval_return_mean") -> Path:
    plots_dir = ensure_dir(plots_dir)
    sub = eval_df[(eval_df["mode"] == "auto") & np.isclose(eval_df["reward_scale"], 1.0)].copy()
    final_step = sub["timestep"].max()
    final = sub[sub["timestep"] == final_step]
    agg = mean_ci_over_seeds(final, ["target_angle_deg"], metric).sort_values("target_angle_deg")
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(agg))
    ax.bar(x, agg["mean"], yerr=agg["ci95"], capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:g}°" for t in agg["target_angle_deg"]])
    ax.set_title(f"Final auto-α performance across target angles: {metric}")
    ax.set_xlabel("Target angle")
    ax.set_ylabel(metric)
    ax.grid(True, axis="y", alpha=0.3)
    out = Path(plots_dir) / f"final_summary_auto_targets_{metric}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# -----------------------------
# Video generation
# -----------------------------


def make_policy_from_checkpoint(seed_dir: str | Path, device: str = "cpu", checkpoint_name: str = "best_model.pt") -> Tuple[SACAgent, Dict[str, Any]]:
    seed_dir = Path(seed_dir)
    cfg = load_yaml(seed_dir / "config.yaml")
    env = make_target_pendulum_env(
        cfg,
        cfg["run"]["target_angle_deg"],
        cfg["run"]["reward_scale"],
        seed=cfg["run"]["seed"],
        render_mode=None,
    )
    agent = build_agent_for_env(env, cfg, device=device)
    agent.load_actor_only(seed_dir / checkpoint_name, map_location=torch.device(device))
    env.close()
    return agent, cfg


def annotate_frame(frame: np.ndarray, lines: List[str]) -> np.ndarray:
    """Annotate RGB frame using PIL if available; silently returns original if PIL missing."""
    try:
        from PIL import Image, ImageDraw, ImageFont

        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 16)
        except Exception:
            font = ImageFont.load_default()
        x, y = 8, 8
        # draw a simple background rectangle for readability
        line_height = 20
        box_h = line_height * len(lines) + 8
        draw.rectangle([0, 0, 360, box_h], fill=(255, 255, 255))
        for line in lines:
            draw.text((x, y), line, fill=(0, 0, 0), font=font)
            y += line_height
        return np.asarray(img)
    except Exception:
        return frame


def generate_rollout_video(
    seed_dir: str | Path,
    out_path: str | Path,
    checkpoint_name: str = "best_model.pt",
    device: str = "cpu",
    fps: int = 30,
    max_frames: Optional[int] = None,
) -> Path:
    seed_dir = Path(seed_dir)
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    agent, cfg = make_policy_from_checkpoint(seed_dir, device=device, checkpoint_name=checkpoint_name)
    target = float(cfg["run"]["target_angle_deg"])
    scale = float(cfg["run"]["reward_scale"])
    seed = int(cfg["run"]["seed"])
    env = make_target_pendulum_env(cfg, target, scale, seed=seed, render_mode="rgb_array")
    obs, _ = env.reset(seed=seed + 777)
    frames = []
    done = False
    cumulative_reward = 0.0
    t = 0
    while not done:
        action = agent.select_action(obs, evaluate=True)
        obs, reward, terminated, truncated, info = env.step(action)
        cumulative_reward += float(reward)
        frame = env.render()
        lines = [
            f"t={t:04d} | target={target:g} deg | theta={info['theta_deg']:+.1f} deg",
            f"error={info['angle_error_deg']:+.1f} deg | |omega|={abs(info['theta_dot']):.2f} | torque={info['torque']:+.2f}",
            f"reward={reward:+.3f} | cumulative={cumulative_reward:+.1f}",
        ]
        frames.append(annotate_frame(frame, lines))
        t += 1
        done = bool(terminated or truncated)
        if max_frames is not None and t >= max_frames:
            break
    imageio.mimsave(out_path, frames, fps=fps, macro_block_size=1)
    env.close()
    return out_path
