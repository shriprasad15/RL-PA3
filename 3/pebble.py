"""PEBBLE (Lee et al. 2021): preference-based SAC.

High-level algorithm (paper):
1. *Unsupervised pre-training*: run SAC using an *intrinsic* reward (we use k-NN state-entropy
   bonus, an approximation of the paper's `APT` reward) for `unsup_steps` env steps.
2. *Reward-model pre-training*: collect `n_init_queries` preference labels from the simulated
   teacher and train the reward model for `reward_epochs_init` epochs on them.
3. *Main loop* (interleaved):
   - every K env steps, sample `n_queries_per_batch` new preference queries, get labels,
     add to the preference buffer, train the reward model for `reward_epochs_per_batch` epochs.
   - relabel all transitions in the SAC replay buffer with the current reward model.
   - run one SAC gradient update per env step (same as vanilla SAC).
   - stop sampling preference queries once `total_feedback` labels have been collected.

Reward model:
- ensemble of 3 MLPs r_theta(s, a) -> scalar (per-timestep reward). Training minimises the
  cross-entropy of the Bradley-Terry preference model: given two segments sigma^0, sigma^1
  with teacher label y in {0,1}, the predicted probability of sigma^0 being preferred is
      P[sigma^0 > sigma^1] = exp(sum r_theta(sigma^0)) / (exp(sum r_theta(sigma^0)) + exp(sum r_theta(sigma^1)))
  cross-entropy loss = -( y log P_0 + (1-y) log P_1 ).

Simulated teacher: uses ground-truth cumulative reward; labels 0 if segment 0 has strictly
higher ground-truth return, else 1. Ties broken to 0.5 (soft label).

This module is intentionally self-contained so it can sit next to Pendulum or Reacher helper
modules. It operates on a generic env_fn and a ground-truth reward function (in the Pendulum
case this is just the env's own reward; in the Reacher case the user picks which of Ra/Rb/Rc
the simulated teacher optimises).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sac_core import (
    ReplayBuffer, SquashedGaussianActor, TwinQCritic, mlp, soft_update, get_device,
    EvalLog, evaluate_policy,
)


# =============================================================================
# Reward model ensemble
# =============================================================================
class RewardModel(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256), n_ensemble: int = 3):
        super().__init__()
        self.nets = nn.ModuleList([mlp([obs_dim + act_dim, *hidden, 1]) for _ in range(n_ensemble)])
        self.n_ensemble = n_ensemble

    def forward_single(self, i, obs, act):
        return self.nets[i](torch.cat([obs, act], dim=-1)).squeeze(-1)

    def predict(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """Mean prediction across the ensemble. obs, act can be batched with any leading dims."""
        rs = torch.stack([self.forward_single(i, obs, act) for i in range(self.n_ensemble)], dim=0)
        return rs.mean(dim=0)

    def predict_all(self, obs, act):
        return torch.stack([self.forward_single(i, obs, act) for i in range(self.n_ensemble)], dim=0)


# =============================================================================
# Preference buffer
# =============================================================================
@dataclass
class PrefBuffer:
    capacity: int = 2000
    seg_obs0: list = field(default_factory=list)
    seg_act0: list = field(default_factory=list)
    seg_obs1: list = field(default_factory=list)
    seg_act1: list = field(default_factory=list)
    labels:   list = field(default_factory=list)  # 0 or 1 or 0.5

    def add(self, o0, a0, o1, a1, y):
        self.seg_obs0.append(o0); self.seg_act0.append(a0)
        self.seg_obs1.append(o1); self.seg_act1.append(a1)
        self.labels.append(y)
        # simple FIFO eviction
        if len(self.labels) > self.capacity:
            self.seg_obs0.pop(0); self.seg_act0.pop(0)
            self.seg_obs1.pop(0); self.seg_act1.pop(0)
            self.labels.pop(0)

    def __len__(self):
        return len(self.labels)

    def sample_batch(self, batch_size: int, device):
        n = len(self)
        idx = np.random.randint(0, n, size=min(batch_size, n))
        o0 = torch.as_tensor(np.stack([self.seg_obs0[i] for i in idx]), device=device, dtype=torch.float32)
        a0 = torch.as_tensor(np.stack([self.seg_act0[i] for i in idx]), device=device, dtype=torch.float32)
        o1 = torch.as_tensor(np.stack([self.seg_obs1[i] for i in idx]), device=device, dtype=torch.float32)
        a1 = torch.as_tensor(np.stack([self.seg_act1[i] for i in idx]), device=device, dtype=torch.float32)
        y  = torch.as_tensor(np.array([self.labels[i] for i in idx]), device=device, dtype=torch.float32)
        return o0, a0, o1, a1, y


# =============================================================================
# k-NN state-entropy intrinsic reward (simple APT approximation)
# =============================================================================
def knn_state_entropy(states: np.ndarray, k: int = 5) -> np.ndarray:
    """For each row of `states`, return log(distance_to_kth_NN + 1) as a state-novelty reward."""
    N = states.shape[0]
    if N <= k + 1:
        return np.zeros(N, dtype=np.float32)
    # Pairwise Euclidean distances (N, N); ok at a few thousand states
    d = np.linalg.norm(states[:, None, :] - states[None, :, :], axis=-1)
    d.sort(axis=1)
    kth = d[:, k]  # skip self at index 0
    return np.log(kth + 1.0).astype(np.float32)


# =============================================================================
# PEBBLE trainer
# =============================================================================
@dataclass
class PebbleConfig:
    # SAC side (identical to SACConfig, held inline to keep the module self-contained)
    hidden: tuple = (256, 256)
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    buffer_size: int = 500_000
    start_steps: int = 10_000
    update_after: int = 1_000
    autotune_alpha: bool = True
    init_alpha: float = 0.2
    # PEBBLE side
    unsup_steps: int = 9_000        # steps of intrinsic-reward SAC before the first preference query
    segment_len: int = 50           # length of each trajectory segment in a preference query
    n_init_queries: int = 100       # preference queries sampled before the first reward-model training
    n_queries_per_batch: int = 32   # new queries per batch in the main loop
    query_every: int = 5_000        # env steps between preference-sampling batches
    reward_epochs_init: int = 200
    reward_epochs_per_batch: int = 20
    total_feedback: int = 500       # budget (max preference labels from the teacher)
    reward_lr: float = 3e-4
    n_ensemble: int = 3
    reward_batch_size: int = 128
    pref_buffer_cap: int = 4000
    # exploration sampling for preference queries: "uniform" (pure random) or "disagreement"
    query_strategy: str = "uniform"


class PebbleAgent:
    """SAC agent learning from a reward model trained on preferences.

    The replay buffer additionally stores the *ground-truth reward* alongside each transition
    (in a parallel array `gt_rews`). The simulated teacher reads from this array to rate
    trajectory segments — avoids reconstructing the ground-truth reward from obs/action, which
    may not always be possible (e.g. environments where state info isn't in the obs vector).
    """

    def __init__(self, obs_dim, act_dim, act_limit, cfg: PebbleConfig, device=None):
        self.cfg = cfg
        self.device = device if device is not None else get_device()
        self.obs_dim, self.act_dim, self.act_limit = obs_dim, act_dim, act_limit

        self.actor = SquashedGaussianActor(obs_dim, act_dim, act_limit, cfg.hidden).to(self.device)
        self.critic = TwinQCritic(obs_dim, act_dim, cfg.hidden).to(self.device)
        self.critic_target = TwinQCritic(obs_dim, act_dim, cfg.hidden).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        self.target_entropy = -float(act_dim)
        self.log_alpha = torch.tensor(np.log(cfg.init_alpha), device=self.device, requires_grad=cfg.autotune_alpha)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr) if cfg.autotune_alpha else None

        self.buffer = ReplayBuffer(cfg.buffer_size, obs_dim, act_dim, discrete=False)
        # parallel GT-reward array used only by the simulated teacher
        self.gt_rews = np.zeros(cfg.buffer_size, dtype=np.float32)
        self.reward_model = RewardModel(obs_dim, act_dim, cfg.hidden, cfg.n_ensemble).to(self.device)
        self.reward_opt = torch.optim.Adam(self.reward_model.parameters(), lr=cfg.reward_lr)
        self.pref = PrefBuffer(capacity=cfg.pref_buffer_cap)
        self.total_env_steps = 0
        self.total_feedback_used = 0

    def add_transition(self, o, a, r_stored, o2, done, gt_r):
        """Insert a transition. `r_stored` is the reward saved in the SAC replay (either
        intrinsic, predicted, or later relabelled); `gt_r` is the true env reward for the
        teacher's use.
        """
        ptr = self.buffer.ptr
        self.buffer.add(o, a, r_stored, o2, done)
        self.gt_rews[ptr] = float(gt_r)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        a, _ = self.actor(o, deterministic=deterministic, with_logprob=False)
        return a.cpu().numpy()[0]

    def random_action(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(-self.act_limit, self.act_limit, size=self.act_dim).astype(np.float32)

    # ---------- reward model training ----------
    def train_reward_model(self, n_epochs: int):
        if len(self.pref) < 8:
            return
        for _ in range(n_epochs):
            o0, a0, o1, a1, y = self.pref.sample_batch(self.cfg.reward_batch_size, self.device)
            total_loss = 0.0
            self.reward_opt.zero_grad()
            for i in range(self.cfg.n_ensemble):
                # bootstrap: random 80% of each batch for this ensemble member
                idx = torch.randperm(o0.shape[0], device=self.device)[: max(1, int(0.8 * o0.shape[0]))]
                r0 = self.reward_model.forward_single(i, o0[idx], a0[idx]).sum(dim=-1)  # (B,)
                r1 = self.reward_model.forward_single(i, o1[idx], a1[idx]).sum(dim=-1)
                # P[seg0 > seg1] = sigma(r0 - r1)  (equivalent to softmax over 2 logits)
                logits = torch.stack([r0, r1], dim=-1)  # (B,2)
                log_probs = F.log_softmax(logits, dim=-1)
                # y==0 -> seg0 preferred -> target index 0
                target = torch.stack([1.0 - y[idx], y[idx]], dim=-1)
                loss = -(target * log_probs).sum(dim=-1).mean()
                total_loss = total_loss + loss
            total_loss = total_loss / self.cfg.n_ensemble
            total_loss.backward()
            self.reward_opt.step()

    # ---------- relabelling replay ----------
    @torch.no_grad()
    def relabel_buffer(self, chunk: int = 8192):
        n = self.buffer.size
        if n == 0:
            return
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            o = torch.as_tensor(self.buffer.obs[start:end], device=self.device)
            a = torch.as_tensor(self.buffer.acts[start:end], device=self.device)
            r = self.reward_model.predict(o, a)
            self.buffer.rews[start:end] = r.cpu().numpy()

    # ---------- SAC update using buffer rewards (already relabelled) ----------
    def sac_update(self) -> dict:
        batch = self.buffer.sample(self.cfg.batch_size, self.device)
        o, a, r, o2, d = batch["obs"], batch["acts"], batch["rews"], batch["next_obs"], batch["dones"]
        with torch.no_grad():
            a2, logp2 = self.actor(o2)
            q1_t, q2_t = self.critic_target(o2, a2)
            q_t = torch.min(q1_t, q2_t) - self.alpha.detach() * logp2
            y = r + self.cfg.gamma * (1.0 - d) * q_t
        q1, q2 = self.critic(o, a)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.critic_opt.zero_grad(); critic_loss.backward(); self.critic_opt.step()

        for p in self.critic.parameters():
            p.requires_grad_(False)
        a_pi, logp_pi = self.actor(o)
        q1_pi, q2_pi = self.critic(o, a_pi)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha.detach() * logp_pi - q_pi).mean()
        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()
        for p in self.critic.parameters():
            p.requires_grad_(True)

        if self.cfg.autotune_alpha:
            alpha_loss = -(self.log_alpha * (logp_pi.detach() + self.target_entropy)).mean()
            self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()

        soft_update(self.critic_target, self.critic, self.cfg.tau)
        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item(),
                "alpha": float(self.alpha.detach().cpu())}

    # ---------- preference sampling ----------
    def sample_preference_query(self):
        """Pick two random segments from the replay buffer and return their GT rewards too.

        Returns (obs0, act0, gt0, obs1, act1, gt1) where gt* are per-step ground-truth reward arrays.
        """
        obs_segs, act_segs, starts = self.buffer.sample_segments(2, self.cfg.segment_len)
        gt_segs = np.stack([self.gt_rews[s:s + self.cfg.segment_len] for s in starts])
        return obs_segs[0], act_segs[0], gt_segs[0], obs_segs[1], act_segs[1], gt_segs[1]


# =============================================================================
# Simulated teacher
# =============================================================================
def simulated_teacher_label_from_gt(gt0: np.ndarray, gt1: np.ndarray) -> float:
    """Return 0 if seg 0 has HIGHER cumulative ground-truth reward, 1 if lower, 0.5 on ties."""
    r0 = float(np.sum(gt0)); r1 = float(np.sum(gt1))
    if abs(r0 - r1) < 1e-6:
        return 0.5
    return 0.0 if r0 > r1 else 1.0


# =============================================================================
# Training loop
# =============================================================================
def train_pebble(env_fn: Callable, agent: PebbleAgent, *,
                 total_steps: int,
                 eval_env_fn: Callable,
                 eval_every: int = 10_000,
                 eval_episodes: int = 20,
                 log_stdout: bool = True,
                 seed: int = 0):
    """Full PEBBLE training loop.

    The training env's reward is what we call the "ground-truth reward" — it's what the
    simulated teacher uses to label preferences. The SAC agent NEVER sees that ground-truth
    reward directly; it learns from the reward-model predictions (except during unsupervised
    pre-training when it uses intrinsic motivation).

    Eval always uses the ground-truth env reward.
    """
    cfg = agent.cfg
    rng = np.random.default_rng(seed)
    env = env_fn()
    obs, _ = env.reset(seed=seed)

    log = EvalLog()

    def _eval_now(step):
        out = evaluate_policy(eval_env_fn,
                              act_fn=lambda o: agent.act(o, deterministic=True),
                              n_episodes=eval_episodes)
        log.append(step, out["return"])
        if log_stdout:
            print(f"[step {step:>7}] gt return = {out['return']:.2f}  "
                  f"feedback={agent.total_feedback_used}")

    # ------------------ phase 1: unsupervised pre-training (intrinsic reward) ------------------
    _eval_now(0)
    for t in range(1, cfg.unsup_steps + 1):
        agent.total_env_steps = t
        if t <= cfg.start_steps:
            a = agent.random_action(rng)
        else:
            a = agent.act(obs)
        next_obs, r_env, term, trunc, _ = env.step(a)
        agent.add_transition(obs, a, 0.0, next_obs, float(term), gt_r=r_env)
        obs = next_obs
        if term or trunc:
            obs, _ = env.reset()
        # refresh intrinsic reward (k-NN novelty) over the most recent chunk
        if t % 1000 == 0 and agent.buffer.size > cfg.start_steps:
            recent = min(agent.buffer.size, 4000)
            idxs = np.arange(agent.buffer.size - recent, agent.buffer.size)
            states = agent.buffer.obs[idxs]
            r_int = knn_state_entropy(states, k=5)
            agent.buffer.rews[idxs] = r_int.astype(np.float32)
        if t >= cfg.update_after:
            agent.sac_update()
        if t % eval_every == 0:
            _eval_now(t)

    # ------------------ phase 2: initial preference queries + reward-model training ------------------
    for _ in range(min(cfg.n_init_queries, max(0, cfg.total_feedback - agent.total_feedback_used))):
        if agent.buffer.size < cfg.segment_len + 1:
            break
        o0, a0, g0, o1, a1, g1 = agent.sample_preference_query()
        y = simulated_teacher_label_from_gt(g0, g1)
        agent.pref.add(o0, a0, o1, a1, y)
        agent.total_feedback_used += 1
    agent.train_reward_model(cfg.reward_epochs_init)
    agent.relabel_buffer()

    # ------------------ phase 3: main PEBBLE loop ------------------
    next_query_step = cfg.unsup_steps + cfg.query_every
    for t in range(cfg.unsup_steps + 1, total_steps + 1):
        agent.total_env_steps = t
        if t <= cfg.start_steps:
            a = agent.random_action(rng)
        else:
            a = agent.act(obs)
        next_obs, r_env, term, trunc, _ = env.step(a)
        with torch.no_grad():
            ot = torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
            at = torch.as_tensor(a, dtype=torch.float32, device=agent.device).unsqueeze(0)
            r_pred = float(agent.reward_model.predict(ot, at).item())
        agent.add_transition(obs, a, r_pred, next_obs, float(term), gt_r=r_env)
        obs = next_obs
        if term or trunc:
            obs, _ = env.reset()
        if t >= cfg.update_after:
            agent.sac_update()
        if t >= next_query_step and agent.total_feedback_used < cfg.total_feedback:
            budget_left = cfg.total_feedback - agent.total_feedback_used
            n_new = min(cfg.n_queries_per_batch, budget_left)
            for _ in range(n_new):
                if agent.buffer.size < cfg.segment_len + 1:
                    break
                o0, a0, g0, o1, a1, g1 = agent.sample_preference_query()
                y = simulated_teacher_label_from_gt(g0, g1)
                agent.pref.add(o0, a0, o1, a1, y)
                agent.total_feedback_used += 1
            agent.train_reward_model(cfg.reward_epochs_per_batch)
            agent.relabel_buffer()
            next_query_step += cfg.query_every
        if t % eval_every == 0:
            _eval_now(t)

    env.close()
    return log
