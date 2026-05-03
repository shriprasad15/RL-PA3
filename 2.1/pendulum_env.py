"""Modified Pendulum-v1: align to target angle theta_target (in degrees) with the vertical.

Pendulum-v1 observation is (cos(theta), sin(theta), theta_dot) where theta=0 is upright
(pointing along +x in the gym convention). So the unwrapped env exposes `env.state = (theta, theta_dot)`
with theta in radians.

We define the error e = angle_diff(theta - theta_target_rad) wrapped to [-pi, pi]. The reward
is a classic quadratic:

    r = -(e**2 + 0.1 * theta_dot**2 + 0.001 * torque**2)

which matches the gym default reward *shape* but re-centered around theta_target instead of 0.
Maximum per-step reward is 0 (at target, zero velocity, zero action). Episode length 1000.

The environment has no terminal state (only truncation at T=1000). We preserve the original
obs space (cos, sin, theta_dot) so the policy network doesn't need to know theta_target; the
objective is encoded purely through the reward.

NOTE: gym's default Pendulum uses T=200. We explicitly override max_episode_steps=1000 per
the assignment spec.
"""
from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
from gymnasium.envs.classic_control.pendulum import PendulumEnv


def angle_diff(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return ((a + math.pi) % (2 * math.pi)) - math.pi


class PendulumTargetAngle(gym.Wrapper):
    """Wrap Pendulum-v1 with a per-step reward centred on theta_target (degrees)."""

    def __init__(self, theta_target_deg: float = 0.0, max_episode_steps: int = 1000,
                 reward_scale: float = 1.0):
        env = gym.make("Pendulum-v1").unwrapped  # strip the default 200-step TimeLimit
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
        super().__init__(env)
        self.theta_target_rad = math.radians(theta_target_deg)
        self.theta_target_deg = theta_target_deg
        self.reward_scale = float(reward_scale)

    def step(self, action):
        obs, _gym_r, term, trunc, info = self.env.step(action)
        theta, theta_dot = self.unwrapped.state
        u = float(np.clip(action, -2.0, 2.0).item() if np.ndim(action) else action)
        e = angle_diff(theta - self.theta_target_rad)
        cost = e * e + 0.1 * theta_dot * theta_dot + 0.001 * u * u
        r = -cost * self.reward_scale
        info["angle_error_rad"] = e
        info["at_target"] = bool(abs(e) < math.radians(5.0) and abs(theta_dot) < 0.5)
        return obs, float(r), term, trunc, info


def make_pendulum(theta_target_deg: float, reward_scale: float = 1.0,
                  max_episode_steps: int = 1000) -> gym.Env:
    return PendulumTargetAngle(theta_target_deg, max_episode_steps, reward_scale)
