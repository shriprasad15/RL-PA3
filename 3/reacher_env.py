"""DeepMind Control Suite `reacher-easy` wrappers for the three reward formulations.

We call dm_control directly (no shimmy dependency) so we can precisely control timeouts and
arm-resets for the Rc formulation. The resulting env presents a Gymnasium-style API.

Reward formulations (PDF Fig. 3):

    Ra (dense):
        r = 1                                  if fingertip in target
          = -||x_goal - x_pos|| - ||a||^2      otherwise
        Episode: fixed length 1000. Truncated on time-limit.

    Rb (sparse, dm_control default):
        r = 1 if in target else 0
        Episode: fixed length 1000. Truncated on time-limit.

    Rc (time-to-reach, per TA clarification):
        r_per_step = -1
        Episode terminates ONLY when the fingertip is in the target with near-zero velocity.
        Timeout is NOT episode completion:
          - after 1000 steps without completion, RESET THE ARM (randomise joints & velocity)
            but keep the SAME target.
          - add a reset penalty of -20 to the reward at that step.
          - episode continues with the same counters for return and length.
        In eval mode only, timeout IS treated as episode end; the eval return for a
        timed-out episode is -1020 (= -1000 for the steps + -20 reset penalty).

TA also said:
- Ra / Rb / Rc should all train for 500K steps for a clean comparison.
- Confidence intervals for the final-policy 500-ep eval should be over SEEDS, not episodes.
"""
from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
from dm_control import suite


# ---------------------------------------------------------------------------
# dm_control -> gymnasium adapter (minimal; only what we need)
# ---------------------------------------------------------------------------
class DMCReacherBase(gym.Env):
    """Minimal Gymnasium wrapper around dm_control `reacher-easy` exposing flattened obs."""

    metadata = {"render_modes": []}

    def __init__(self, dmc_seed: int | None = None):
        super().__init__()
        # dm_control's Task seed must be supplied at construction via task_kwargs.
        # Its `.random` attribute is a read-only property, so we can't reseed after init.
        self._dmc_seed = dmc_seed
        self._env = self._build(dmc_seed)
        self._action_spec = self._env.action_spec()
        act_low = np.asarray(self._action_spec.minimum, dtype=np.float32)
        act_high = np.asarray(self._action_spec.maximum, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=act_low, high=act_high, dtype=np.float32)
        ts = self._env.reset()
        obs = self._flatten_obs(ts.observation)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                                shape=obs.shape, dtype=np.float32)
        self._np_random, _ = gym.utils.seeding.np_random(dmc_seed or 0)

    @staticmethod
    def _build(seed: int | None):
        kwargs = {"domain_name": "reacher", "task_name": "easy"}
        if seed is not None:
            kwargs["task_kwargs"] = {"random": int(seed)}
        return suite.load(**kwargs)

    @staticmethod
    def _flatten_obs(obs_dict) -> np.ndarray:
        return np.concatenate([np.atleast_1d(v).astype(np.float32).ravel()
                                for v in obs_dict.values()])

    # ---- raw physics helpers used by the reward formulations ----
    def finger_to_target(self) -> np.ndarray:
        return np.asarray(self._env.physics.finger_to_target(), dtype=np.float32)

    def joint_velocity(self) -> np.ndarray:
        return np.asarray(self._env.physics.angular_velocity(), dtype=np.float32)

    def in_target(self) -> bool:
        return bool(float(self._env.task.get_reward(self._env.physics)) > 0.5)

    # ---- arm reset preserving target ----
    def reset_arm_keep_target(self):
        """Randomise joint angles/velocities via dm_control's internal rng but preserve the
        target position. Used by Rc on timeout."""
        physics = self._env.physics
        target_pos = physics.named.model.geom_pos["target"].copy()
        # dm_control reacher Task.initialize_episode randomises both arm and target;
        # we call it, then restore the target.
        with physics.reset_context():
            self._env.task.initialize_episode(physics)
            physics.named.model.geom_pos["target"][:] = target_pos

    # ---- gym API ----
    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            self._np_random, _ = gym.utils.seeding.np_random(seed)
            # dm_control's Task.random is a read-only property. To change the stream
            # we must rebuild the underlying env with the new seed. This only fires on
            # an explicit reset(seed=...), so the cost is amortised across episodes.
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
        # default (sparse) reward reported by dm_control
        r_default = float(ts.reward if ts.reward is not None else 0.0)
        info = {"in_target": r_default > 0.5}
        term = False  # dm_control reacher-easy never terminates by itself
        trunc = False
        return obs, r_default, term, trunc, info

    def close(self):
        self._env.close()


# ---------------------------------------------------------------------------
# Reward wrappers
# ---------------------------------------------------------------------------
class FixedLengthRewardWrapper(gym.Wrapper):
    """Ra or Rb — fixed-length 1000-step episodes, truncated at time limit."""

    def __init__(self, env: DMCReacherBase, reward_name: str, max_episode_steps: int = 1000):
        assert reward_name in ("Ra", "Rb")
        super().__init__(env)
        self.reward_name = reward_name
        self._elapsed = 0
        self._max_steps = int(max_episode_steps)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._elapsed = 0
        return obs, info

    def step(self, action):
        obs, r_default, term, trunc, info = self.env.step(action)
        self._elapsed += 1
        in_t = info["in_target"]

        if self.reward_name == "Ra":
            if in_t:
                r = 1.0
            else:
                dist = float(np.linalg.norm(self.env.finger_to_target()))
                a = np.asarray(action, dtype=np.float32)
                r = -dist - float(np.dot(a, a))
        else:  # Rb
            r = 1.0 if in_t else 0.0

        trunc = trunc or (self._elapsed >= self._max_steps)
        return obs, r, False, trunc, info


class RcTrainingWrapper(gym.Wrapper):
    """Rc in TRAINING mode — timeouts don't end the episode.

    Per-step reward: -1.
    Episode ends ONLY when fingertip is in target with near-zero velocity.
    After every 1000 steps without termination: reset arm (keep target), add -20 penalty,
    continue the same episode (return/length keep accumulating).
    """

    TIMEOUT = 1000
    TIMEOUT_PENALTY = -20.0
    GOAL_VEL_THRESHOLD = 0.5

    def __init__(self, env: DMCReacherBase):
        super().__init__(env)
        self._steps_since_reset = 0  # counts toward the next timeout
        self._cumulative_length = 0
        self._cumulative_return = 0.0
        self._n_timeouts = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._steps_since_reset = 0
        self._cumulative_length = 0
        self._cumulative_return = 0.0
        self._n_timeouts = 0
        return obs, info

    def step(self, action):
        obs, _r_default, _t, _tr, info = self.env.step(action)
        self._steps_since_reset += 1
        self._cumulative_length += 1

        r = -1.0
        vel = self.env.joint_velocity()
        near_zero_vel = float(np.linalg.norm(vel)) < self.GOAL_VEL_THRESHOLD
        reached = info["in_target"] and near_zero_vel

        if reached:
            # episode ends — true goal completion
            self._cumulative_return += r
            info.update({
                "rc_episode_length": self._cumulative_length,
                "rc_episode_return": self._cumulative_return,
                "rc_n_timeouts": self._n_timeouts,
                "rc_reached_goal": True,
            })
            return obs, r, True, False, info

        # timeout handling: reset arm, keep target, apply penalty, continue episode
        if self._steps_since_reset >= self.TIMEOUT:
            r += self.TIMEOUT_PENALTY
            self._n_timeouts += 1
            self.env.reset_arm_keep_target()
            self._steps_since_reset = 0
            # observe the new state after arm reset so the policy sees it next step
            # The spec doesn't require us to emit the reset-state as obs here — the next
            # step() will act from it naturally. We still return the current `obs`.
            info.update({"rc_timeout_reset": True, "rc_n_timeouts": self._n_timeouts})

        self._cumulative_return += r
        info["rc_episode_length_so_far"] = self._cumulative_length
        info["rc_episode_return_so_far"] = self._cumulative_return
        return obs, r, False, False, info


class RcEvaluationWrapper(gym.Wrapper):
    """Rc in EVALUATION mode — 1000-step cap; timeout ends the episode.

    If the agent does not reach the goal within 1000 steps, episode ends with cumulative
    return = -1020 (i.e. -1 per step for 1000 steps, plus a -20 terminal penalty).
    If it reaches the goal, return is (-1 per step) up to termination step.
    """

    TIMEOUT = 1000
    TIMEOUT_PENALTY = -20.0
    GOAL_VEL_THRESHOLD = 0.5

    def __init__(self, env: DMCReacherBase):
        super().__init__(env)
        self._elapsed = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._elapsed = 0
        return obs, info

    def step(self, action):
        obs, _r_default, _t, _tr, info = self.env.step(action)
        self._elapsed += 1

        r = -1.0
        vel = self.env.joint_velocity()
        reached = info["in_target"] and float(np.linalg.norm(vel)) < self.GOAL_VEL_THRESHOLD
        if reached:
            return obs, r, True, False, info
        if self._elapsed >= self.TIMEOUT:
            r += self.TIMEOUT_PENALTY  # -1 (this step) + -20 = -21 on the last step
            return obs, r, False, True, info
        return obs, r, False, False, info


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def make_reacher(reward_name: str = "Rb", mode: str = "train",
                  seed: Optional[int] = None) -> gym.Env:
    """Build a reacher env with the chosen reward.

    mode='train' vs 'eval' only matters for Rc:
      train -> RcTrainingWrapper (timeout-reset-continuation)
      eval  -> RcEvaluationWrapper (1000-step hard cap, -1020 on timeout)
    For Ra and Rb the two modes are identical (fixed length 1000).

    `seed`: optional dm_control task seed (passed at construction). Most callers leave
    it None — the Gym env.reset(seed=...) path will rebuild the env with the seed on
    the first reset.
    """
    base = DMCReacherBase(dmc_seed=seed)
    if reward_name in ("Ra", "Rb"):
        return FixedLengthRewardWrapper(base, reward_name)
    if reward_name == "Rc":
        return RcTrainingWrapper(base) if mode == "train" else RcEvaluationWrapper(base)
    raise ValueError(f"Unknown reward_name {reward_name!r}")


# ---------------------------------------------------------------------------
# Cross-evaluation reward functions (no mutation of env state)
# ---------------------------------------------------------------------------
def reward_fn_Ra(env, obs, action, r, info):
    # env here is whatever was passed to evaluate_policy(); we reach the base via .unwrapped.env chain
    base = env
    while hasattr(base, "env") and not isinstance(base, DMCReacherBase):
        base = base.env
    if info.get("in_target", False):
        return 1.0
    dist = float(np.linalg.norm(base.finger_to_target()))
    a = np.asarray(action, dtype=np.float32)
    return -dist - float(np.dot(a, a))


def reward_fn_Rb(env, obs, action, r, info):
    return 1.0 if info.get("in_target", False) else 0.0


def reward_fn_Rc(env, obs, action, r, info):
    # running Rc metric during evaluation of an Ra/Rb policy:
    # -1 per step; we do not have a live cumulative state for termination here, so this
    # just reports the cumulative per-step cost (the "time cost"). Combined with Rb_eval
    # as the success indicator, the reader can infer time-to-reach.
    return -1.0


CROSS_EVAL_FNS = {
    "Ra_eval": reward_fn_Ra,
    "Rb_eval": reward_fn_Rb,
    "Rc_eval": reward_fn_Rc,
}


# ---------------------------------------------------------------------------
# Policy-quality metrics for Q3(a)
# ---------------------------------------------------------------------------
def steps_to_goal_and_dwell(env_fn, act_fn, n_episodes: int = 500, max_steps: int = 5000,
                            seed_offset: int = 20_000):
    """For each of `n_episodes` deterministic rollouts of length `max_steps`,
    return (steps_to_first_reach, steps_spent_in_target). If the agent never reaches,
    steps_to_goal = max_steps and steps_in_target = 0.

    This function expects a fixed-length env (Ra/Rb style); for Rc pass an Ra or Rb env
    so the arm is not silently reset mid-episode.
    """
    s2g, sit = [], []
    env = env_fn()
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed_offset + ep)
        reached_at = None
        dwell = 0
        for t in range(max_steps):
            a = act_fn(obs)
            obs, r, term, trunc, info = env.step(a)
            if info.get("in_target", False):
                if reached_at is None:
                    reached_at = t
                dwell += 1
            if term or trunc:
                break
        s2g.append(reached_at if reached_at is not None else max_steps)
        sit.append(dwell)
    env.close()
    return np.asarray(s2g), np.asarray(sit)
