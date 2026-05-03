"""DeepMind Control Suite `reacher-easy` wrappers for PA3 Section 2.3.

This file exposes a Gymnasium-style API over dm_control's reacher-easy task and
implements the three reward formulations from the assignment:

Ra (dense shaped, fixed length T=1000 by default):
    r = 1                                      if fingertip is in target
      = -||x_goal - x_pos|| - ||action||^2     otherwise

Rb (sparse/default-style, fixed length T=1000 by default):
    r = 1 if fingertip is in target else 0

Rc (time-to-goal, variable length):
    r = -1 until the fingertip is in target with near-zero velocity.
    In training mode, timeout at 1000 steps resets only the arm while keeping the
    same target, applies a -20 penalty, and continues the same episode.
    In evaluation mode, timeout at 1000 steps ends the episode with the -20
    penalty applied on the final step.

Also includes the final-policy diagnostic required by the assignment:
    steps_to_goal_and_dwell(..., max_steps=5000)
"""
from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
from dm_control import suite


# ---------------------------------------------------------------------------
# dm_control -> Gymnasium adapter
# ---------------------------------------------------------------------------
class DMCReacherBase(gym.Env):
    """Minimal Gymnasium wrapper around dm_control reacher-easy.

    Observation is the flattened dm_control observation dictionary.
    """

    metadata = {"render_modes": []}

    def __init__(self, dmc_seed: int | None = None):
        super().__init__()
        self._dmc_seed = dmc_seed
        self._env = self._build(dmc_seed)

        action_spec = self._env.action_spec()
        act_low = np.asarray(action_spec.minimum, dtype=np.float32)
        act_high = np.asarray(action_spec.maximum, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=act_low, high=act_high, dtype=np.float32)

        ts = self._env.reset()
        obs = self._flatten_obs(ts.observation)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
        )
        self._np_random, _ = gym.utils.seeding.np_random(dmc_seed or 0)

    @staticmethod
    def _build(seed: int | None):
        kwargs = {"domain_name": "reacher", "task_name": "easy"}
        if seed is not None:
            kwargs["task_kwargs"] = {"random": int(seed)}
        return suite.load(**kwargs)

    @staticmethod
    def _flatten_obs(obs_dict) -> np.ndarray:
        return np.concatenate([
            np.atleast_1d(v).astype(np.float32).ravel() for v in obs_dict.values()
        ])

    def current_observation(self) -> np.ndarray:
        """Return current flattened observation without stepping the simulator."""
        obs_dict = self._env.task.get_observation(self._env.physics)
        return self._flatten_obs(obs_dict)

    # ---- physics helpers used by reward formulations ----
    def finger_to_target(self) -> np.ndarray:
        """Vector between fingertip and target; norm is the distance to target."""
        phys = self._env.physics
        fn = getattr(phys, "finger_to_target", None)
        if callable(fn):
            return np.asarray(fn(), dtype=np.float32).copy()
        try:
            finger = phys.named.data.geom_xpos["finger"][:2]
            target = phys.named.data.geom_xpos["target"][:2]
            return np.asarray(target - finger, dtype=np.float32)
        except Exception:
            return np.zeros(2, dtype=np.float32)

    def joint_velocity(self) -> np.ndarray:
        """MuJoCo qvel for the two reacher joints."""
        return np.asarray(self._env.physics.data.qvel, dtype=np.float32).copy()

    def in_target(self) -> bool:
        """Use dm_control's own target reward as the target-membership signal."""
        return bool(float(self._env.task.get_reward(self._env.physics)) > 0.5)

    def reset_arm_keep_target(self) -> np.ndarray:
        """Randomize arm state while preserving target position; return new obs.

        Used by Rc training after timeout.
        """
        physics = self._env.physics
        target_pos = physics.named.model.geom_pos["target"].copy()
        with physics.reset_context():
            self._env.task.initialize_episode(physics)
            physics.named.model.geom_pos["target"][:] = target_pos
        return self.current_observation()

    # ---- Gymnasium API ----
    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            self._np_random, _ = gym.utils.seeding.np_random(seed)
            # dm_control task RNG is set at construction, so rebuild if seed changes.
            if seed != self._dmc_seed:
                try:
                    self._env.close()
                except Exception:
                    pass
                self._env = self._build(seed)
                self._dmc_seed = seed
        ts = self._env.reset()
        return self._flatten_obs(ts.observation), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32),
                         self.action_space.low, self.action_space.high)
        ts = self._env.step(action)
        obs = self._flatten_obs(ts.observation)
        r_default = float(ts.reward if ts.reward is not None else 0.0)
        info = {"in_target": r_default > 0.5}
        # dm_control reacher-easy is normally continuing; wrappers control termination.
        return obs, r_default, False, False, info

    def close(self):
        self._env.close()


# ---------------------------------------------------------------------------
# Reward wrappers
# ---------------------------------------------------------------------------
class FixedLengthRewardWrapper(gym.Wrapper):
    """Ra or Rb with fixed-length episodes."""

    def __init__(self, env: DMCReacherBase, reward_name: str,
                 max_episode_steps: int = 1000):
        assert reward_name in ("Ra", "Rb")
        super().__init__(env)
        self.reward_name = reward_name
        self.max_episode_steps = int(max_episode_steps)
        self.elapsed = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.elapsed = 0
        return obs, info

    def step(self, action):
        obs, _r_default, term, trunc, info = self.env.step(action)
        self.elapsed += 1
        in_target = bool(info.get("in_target", False))

        if self.reward_name == "Ra":
            if in_target:
                reward = 1.0
            else:
                dist = float(np.linalg.norm(self.env.finger_to_target()))
                a = np.asarray(action, dtype=np.float32)
                reward = -dist - float(np.dot(a, a))
        else:  # Rb
            reward = 1.0 if in_target else 0.0

        trunc = bool(trunc or self.elapsed >= self.max_episode_steps)
        return obs, float(reward), False, trunc, info


class RcTrainingWrapper(gym.Wrapper):
    """Rc training mode.

    Per-step reward is -1. Episode terminates only when the target is reached
    with near-zero joint velocity. Every 1000 steps without termination, reset
    arm while keeping the same target, add -20 penalty, and continue.
    """

    TIMEOUT = 1000
    TIMEOUT_PENALTY = -20.0
    GOAL_VEL_THRESHOLD = 0.5

    def __init__(self, env: DMCReacherBase):
        super().__init__(env)
        self.steps_since_reset = 0
        self.cumulative_length = 0
        self.cumulative_return = 0.0
        self.n_timeouts = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.steps_since_reset = 0
        self.cumulative_length = 0
        self.cumulative_return = 0.0
        self.n_timeouts = 0
        return obs, info

    def step(self, action):
        obs, _r_default, _term, _trunc, info = self.env.step(action)
        self.steps_since_reset += 1
        self.cumulative_length += 1

        reward = -1.0
        near_zero_vel = float(np.linalg.norm(self.env.joint_velocity())) < self.GOAL_VEL_THRESHOLD
        reached = bool(info.get("in_target", False)) and near_zero_vel

        if reached:
            self.cumulative_return += reward
            info.update({
                "rc_reached_goal": True,
                "rc_episode_length": self.cumulative_length,
                "rc_episode_return": self.cumulative_return,
                "rc_n_timeouts": self.n_timeouts,
            })
            return obs, float(reward), True, False, info

        if self.steps_since_reset >= self.TIMEOUT:
            reward += self.TIMEOUT_PENALTY
            self.n_timeouts += 1
            obs = self.env.reset_arm_keep_target()  # important: return the new state
            self.steps_since_reset = 0
            info.update({
                "rc_timeout_reset": True,
                "rc_n_timeouts": self.n_timeouts,
            })

        self.cumulative_return += reward
        info.update({
            "rc_reached_goal": False,
            "rc_episode_length_so_far": self.cumulative_length,
            "rc_episode_return_so_far": self.cumulative_return,
        })
        return obs, float(reward), False, False, info


class RcEvaluationWrapper(gym.Wrapper):
    """Rc evaluation mode with a 1000-step hard cap.

    If not reached by 1000 steps, episode truncates and receives the -20 timeout
    penalty on the final step.
    """

    TIMEOUT = 1000
    TIMEOUT_PENALTY = -20.0
    GOAL_VEL_THRESHOLD = 0.5

    def __init__(self, env: DMCReacherBase):
        super().__init__(env)
        self.elapsed = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.elapsed = 0
        return obs, info

    def step(self, action):
        obs, _r_default, _term, _trunc, info = self.env.step(action)
        self.elapsed += 1
        reward = -1.0

        near_zero_vel = float(np.linalg.norm(self.env.joint_velocity())) < self.GOAL_VEL_THRESHOLD
        reached = bool(info.get("in_target", False)) and near_zero_vel
        if reached:
            info["rc_reached_goal"] = True
            return obs, float(reward), True, False, info

        if self.elapsed >= self.TIMEOUT:
            reward += self.TIMEOUT_PENALTY
            info["rc_reached_goal"] = False
            info["rc_timeout"] = True
            return obs, float(reward), False, True, info

        info["rc_reached_goal"] = False
        return obs, float(reward), False, False, info


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def make_reacher(reward_name: str = "Rb", mode: str = "train",
                 seed: Optional[int] = None,
                 max_episode_steps: int = 1000) -> gym.Env:
    """Build reacher-easy with selected reward.

    Args:
        reward_name: one of {"Ra", "Rb", "Rc"}.
        mode: "train" or "eval". Only affects Rc.
        seed: optional dm_control task seed.
        max_episode_steps: used for Ra/Rb fixed-length wrappers. Set to 5000 for
            the final diagnostic required in Q3(a).
    """
    assert mode in ("train", "eval")
    base = DMCReacherBase(dmc_seed=seed)
    if reward_name in ("Ra", "Rb"):
        return FixedLengthRewardWrapper(base, reward_name,
                                        max_episode_steps=max_episode_steps)
    if reward_name == "Rc":
        return RcTrainingWrapper(base) if mode == "train" else RcEvaluationWrapper(base)
    raise ValueError(f"Unknown reward_name={reward_name!r}")


# ---------------------------------------------------------------------------
# Final-policy behavior metrics for Q3(a)
# ---------------------------------------------------------------------------
def steps_to_goal_and_dwell(env_fn, act_fn, n_episodes: int = 500,
                            max_steps: int = 5000,
                            seed_offset: int = 20_000):
    """Compute steps-to-goal and steps-in-target for deterministic rollouts.

    If the agent never reaches the target, steps_to_goal=max_steps and
    steps_in_target=0. Use an Ra/Rb fixed-length env with max_episode_steps=5000
    so this diagnostic is not truncated at 1000.
    """
    steps_to_goal = []
    steps_in_target = []
    env = env_fn()
    for ep in range(n_episodes):
        obs, _info = env.reset(seed=seed_offset + ep)
        first_reach = None
        dwell = 0
        for t in range(max_steps):
            action = act_fn(obs)
            obs, _r, term, trunc, info = env.step(action)
            in_target = bool(info.get("in_target", False))
            if in_target and first_reach is None:
                first_reach = t
            if first_reach is not None and in_target:
                dwell += 1
            if term or trunc:
                break
        steps_to_goal.append(first_reach if first_reach is not None else max_steps)
        steps_in_target.append(dwell if first_reach is not None else 0)
    env.close()
    return np.asarray(steps_to_goal), np.asarray(steps_in_target)
