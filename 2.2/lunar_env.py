"""LunarLander-v3 wrappers for PA3 section 2.2.

- make_lunar_continuous(hover_bonus=None): continuous LunarLander.
- make_lunar_discrete(): discrete LunarLander.
- HoverBonusWrapper adds a one-shot hover reward when |x| < 0.1 and 0.4 < |y| < 0.6.
"""
from __future__ import annotations

import gymnasium as gym


class HoverBonusWrapper(gym.Wrapper):
    """Add a one-shot reward when the lander enters the hover box.

    Observation: (x, y, vx, vy, angle, angular_velocity, left_leg_contact, right_leg_contact)
    The hover bonus can be changed mid-training with set_hover_bonus(+200 -> -100).
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

        # Always expose these keys so evaluation metrics are clean.
        info = dict(info)
        info["inside_hover_box"] = bool(in_box)
        info["hover_bonus_given"] = False
        info["hover_bonus_value"] = float(self.hover_bonus)

        if in_box and not self._given:
            r = float(r) + self.hover_bonus
            self._given = True
            info["hover_bonus_given"] = True

        return obs, float(r), term, trunc, info


def make_lunar_continuous(hover_bonus: float | None = None,
                          enable_wind: bool = False) -> gym.Env:
    env = gym.make("LunarLander-v3", continuous=True, enable_wind=enable_wind)
    if hover_bonus is not None:
        env = HoverBonusWrapper(env, hover_bonus)
    return env


def make_lunar_discrete(enable_wind: bool = False) -> gym.Env:
    return gym.make("LunarLander-v3", continuous=False, enable_wind=enable_wind)
