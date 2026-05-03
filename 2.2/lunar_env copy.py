"""LunarLander wrappers.

Two wrappers:
- `make_lunar_continuous(hover_bonus=None)` -> LunarLander-v3 continuous. If `hover_bonus`
  is not None, add that reward (given just once per episode) when the lander enters the
  hover box |x|<0.1, 0.4<|y|<0.6.
- `make_lunar_discrete()` -> LunarLander-v3 discrete.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np


class HoverBonusWrapper(gym.Wrapper):
    """Add a one-shot reward (hover_bonus) when the lander enters the hover box.

    LunarLander observation: (x, y, vx, vy, angle, ang_vel, leg1_contact, leg2_contact).

    The bonus value is stored in a mutable attribute so the training loop can switch it
    (e.g. +200 -> -100) mid-training without rebuilding the env.
    """

    def __init__(self, env: gym.Env, hover_bonus: float):
        super().__init__(env)
        self.hover_bonus = float(hover_bonus)
        self._given = False

    def set_hover_bonus(self, new_bonus: float):
        self.hover_bonus = float(new_bonus)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._given = False
        return obs, info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        x, y = float(obs[0]), float(obs[1])
        in_box = (abs(x) < 0.1) and (0.4 < abs(y) < 0.6)
        if in_box and not self._given:
            r = r + self.hover_bonus
            self._given = True
            info["hover_bonus_given"] = True
        return obs, float(r), term, trunc, info


def make_lunar_continuous(hover_bonus: float | None = None, enable_wind: bool = False) -> gym.Env:
    env = gym.make("LunarLander-v3", continuous=True, enable_wind=enable_wind)
    if hover_bonus is not None:
        env = HoverBonusWrapper(env, hover_bonus)
    return env


def make_lunar_discrete(enable_wind: bool = False) -> gym.Env:
    return gym.make("LunarLander-v3", continuous=False, enable_wind=enable_wind)
