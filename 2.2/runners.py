"""2.2 LunarLander training entry points with per-seed artifact saving.

Each run writes:
runs/<tag>/seed_XX/
  config.json
  eval_log.csv / eval_log.json
  train_log.csv / train_log.json
  best_model.pt
  final_model.pt
  final_rollout.pkl
  DONE.txt

It also writes a legacy logs/<tag>_seed<seed>.json file so older notebooks/run_all.py
that expect logs/*.json continue to work.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from sac_core import (
    set_global_seed,
    save_log,
    save_json,
    save_rows_csv,
    save_pickle,
    write_done,
    collect_rollout,
    get_device,
)
from sac_agent import SACAgent, SACConfig, train_sac
from discrete_sac_agent import DiscreteSACAgent, DiscreteSACConfig, train_discrete_sac
from dqn_agent import DQNAgent, DQNConfig, train_dqn
from lunar_env import make_lunar_continuous, make_lunar_discrete

LOG_DIR = "logs"        # legacy compatibility
RUNS_DIR = "runs"       # new artifact directory
EVAL_EPISODES = 20


def _seed_dir(tag: str, seed: int, output_root: str | None = None) -> Path:
    root = Path(output_root) if output_root is not None else Path(RUNS_DIR)
    return root / tag / f"seed_{seed:02d}"


def _legacy_log_path(tag: str, seed: int) -> str:
    return os.path.join(LOG_DIR, f"{tag}_seed{seed}.json")


def _save_common_artifacts(*, run_dir: Path, legacy_tag: str, seed: int,
                           config: dict, eval_log, train_rows: list[dict],
                           agent, env_fn, rollout_max_steps: int | None = None):
    run_dir.mkdir(parents=True, exist_ok=True)

    # Config
    save_json(config, str(run_dir / "config.json"))

    # Eval log in both JSON and CSV
    eval_dict = eval_log.to_dict()
    save_json(eval_dict, str(run_dir / "eval_log.json"))
    save_rows_csv(eval_log.to_rows({
        "seed": seed,
        "tag": legacy_tag,
        "agent": config.get("agent", ""),
        "env": config.get("env", ""),
    }), str(run_dir / "eval_log.csv"))

    # Train log in both JSON and CSV
    save_json(train_rows, str(run_dir / "train_log.json"))
    save_rows_csv(train_rows, str(run_dir / "train_log.csv"))

    # Model checkpoints
    final_ckpt = agent.checkpoint()
    torch.save(final_ckpt, run_dir / "final_model.pt")

    # Best model saved during training if callback did not save one for any reason.
    best_path = run_dir / "best_model.pt"
    if not best_path.exists():
        torch.save(final_ckpt, best_path)

    # Small deterministic rollout from the final policy for later behavior / t-SNE diagnostics.
    rollout = collect_rollout(
        env_fn,
        act_fn=lambda o: agent.act(o, deterministic=True),
        n_episodes=1,
        max_steps=rollout_max_steps,
    )
    save_pickle(rollout, str(run_dir / "final_rollout.pkl"))

    # Legacy JSON log for old notebooks/dispatcher compatibility.
    os.makedirs(LOG_DIR, exist_ok=True)
    save_log(eval_log, _legacy_log_path(legacy_tag, seed), config=config)

    write_done(str(run_dir), "completed")


def _make_best_saver(run_dir: Path):
    state = {"best": -float("inf")}

    def _callback(step: int, eval_out: dict, agent):
        score = float(eval_out["return"])
        if score > state["best"]:
            state["best"] = score
            run_dir.mkdir(parents=True, exist_ok=True)
            torch.save(agent.checkpoint(), run_dir / "best_model.pt")
            save_json({"best_step": step, "best_eval_return": score}, str(run_dir / "best.json"))

    return _callback


def _make_ckpt_saver(run_dir: Path, agent):
    """best_ckpt_fn-compatible callback (step, best_return) that closes over agent."""
    saver = _make_best_saver(run_dir)

    def _callback(step: int, best_return: float):
        saver(step, {"return": best_return}, agent)

    return _callback


def _hover_hits_fn(env, obs, action, r, info):
    return 1.0 if info.get("hover_bonus_given", False) else 0.0


# ---------------------------------------------------------------------------
# LL-1: continuous SAC auto-alpha
# ---------------------------------------------------------------------------
def run_continuous(seed: int, tag: str = "cont_auto", *, total_steps: int,
                   eval_every: int, hover_bonus=None, buffer_size=1_000_000,
                   buffer_type="infinite-equivalent (1M)", autotune=True,
                   init_alpha=0.2, device=None, output_root: str | None = None):
    set_global_seed(seed)
    run_dir = _seed_dir(tag, seed, output_root)

    env_fn = lambda: make_lunar_continuous(hover_bonus=hover_bonus)
    probe = env_fn()
    obs_dim = probe.observation_space.shape[0]
    act_dim = probe.action_space.shape[0]
    act_limit = float(probe.action_space.high[0])
    probe.close()

    cfg = SACConfig(
        start_steps=10_000,
        update_after=10_000,
        autotune_alpha=autotune,
        init_alpha=init_alpha,
        buffer_size=buffer_size,
    )
    agent = SACAgent(obs_dim, act_dim, act_limit, cfg, device=device or get_device())

    config = {
        "section": "2.2_lunarlander",
        "experiment_id": tag,
        "agent": "SAC",
        "env": "LunarLander-v3 continuous=True" + (" + hover bonus" if hover_bonus is not None else ""),
        "continuous": True,
        "hover_bonus": hover_bonus,
        "seed": seed,
        "total_steps": total_steps,
        "initial_random_steps": cfg.start_steps,
        "update_after": cfg.update_after,
        "replay_buffer_size": cfg.buffer_size,
        "replay_buffer_type": buffer_type,
        "batch_size": cfg.batch_size,
        "gamma": cfg.gamma,
        "tau": cfg.tau,
        "actor_lr": cfg.actor_lr,
        "critic_lr": cfg.critic_lr,
        "alpha_lr": cfg.alpha_lr,
        "autotune_alpha": cfg.autotune_alpha,
        "init_alpha": cfg.init_alpha,
        "target_entropy": -act_dim if cfg.target_entropy is None else cfg.target_entropy,
        "eval_every": eval_every,
        "eval_episodes": EVAL_EPISODES,
    }

    eval_log, train_rows = train_sac(
        env_fn,
        agent,
        total_steps=total_steps,
        eval_env_fn=env_fn,
        eval_every=eval_every,
        eval_episodes=EVAL_EPISODES,
        log_stdout=True,
        seed=seed,
        best_ckpt_fn=_make_ckpt_saver(run_dir, agent),
    )

    _save_common_artifacts(
        run_dir=run_dir,
        legacy_tag=tag,
        seed=seed,
        config=config,
        eval_log=eval_log,
        train_rows=train_rows,
        agent=agent,
        env_fn=env_fn,
        rollout_max_steps=None,
    )


# ---------------------------------------------------------------------------
# LL-2: hover +200 -> -100, fixed or auto alpha
# ---------------------------------------------------------------------------
def run_hover_switch(seed: int, tag: str, *, autotune: bool, init_alpha: float,
                     steps_pre: int, steps_post: int, eval_every: int,
                     buffer_size: int = 250_000, device=None,
                     output_root: str | None = None):
    set_global_seed(seed)
    run_dir = _seed_dir(tag, seed, output_root)

    live_env = make_lunar_continuous(hover_bonus=+200.0)
    current_bonus = {"value": 200.0}
    eval_env_fn = lambda: make_lunar_continuous(hover_bonus=current_bonus["value"])

    obs_dim = live_env.observation_space.shape[0]
    act_dim = live_env.action_space.shape[0]
    act_limit = float(live_env.action_space.high[0])

    cfg = SACConfig(
        autotune_alpha=autotune,
        init_alpha=init_alpha,
        start_steps=10_000,
        update_after=10_000,
        buffer_size=buffer_size,
    )
    agent = SACAgent(obs_dim, act_dim, act_limit, cfg, device=device or get_device())

    config = {
        "section": "2.2_lunarlander",
        "experiment_id": tag,
        "agent": "SAC",
        "env": "LunarLander-v3 continuous=True + hover box",
        "continuous": True,
        "hover_box": "|x| < 0.1 and 0.4 < |y| < 0.6",
        "hover_reward_before": 200.0,
        "hover_reward_after": -100.0,
        "reward_change_step": steps_pre,
        "fixed_alpha": init_alpha if not autotune else None,
        "autotune_alpha": autotune,
        "seed": seed,
        "total_steps": steps_pre + steps_post,
        "steps_pre": steps_pre,
        "steps_post": steps_post,
        "initial_random_steps": cfg.start_steps,
        "update_after": cfg.update_after,
        "replay_buffer_size": cfg.buffer_size,
        "replay_buffer_type": "fixed-size (TA rule for hover reward switch)",
        "batch_size": cfg.batch_size,
        "gamma": cfg.gamma,
        "tau": cfg.tau,
        "actor_lr": cfg.actor_lr,
        "critic_lr": cfg.critic_lr,
        "alpha_lr": cfg.alpha_lr,
        "init_alpha": cfg.init_alpha,
        "target_entropy": -act_dim if cfg.target_entropy is None else cfg.target_entropy,
        "eval_every": eval_every,
        "eval_episodes": EVAL_EPISODES,
    }

    from sac_core import EvalLog, evaluate_policy
    eval_log = EvalLog()
    train_rows: list[dict] = []
    rng = np.random.default_rng(seed)
    obs, _ = live_env.reset(seed=seed)
    ep_ret, ep_len = 0.0, 0
    total = steps_pre + steps_post
    flipped = False
    best_saver = _make_best_saver(run_dir)
    last_update_info = {}

    def _eval_now(step: int):
        out = evaluate_policy(
            eval_env_fn,
            act_fn=lambda o: agent.act(o, deterministic=True),
            n_episodes=EVAL_EPISODES,
            extra_reward_fns={"hover_hits": _hover_hits_fn},
        )
        extras = {
            "hover_hits": out["hover_hits"],
            "hover_reward_value": current_bonus["value"],
            "alpha": float(agent.alpha.detach().cpu()),
        }
        eval_log.append(step, out["return"], extras, std_ret=out.get("return_std"))
        best_saver(step, out, agent)
        print(f"[step {step:>7}] eval_return={out['return']:.2f} hover_hits={out['hover_hits']:.2f}")

    _eval_now(0)

    for t in range(1, total + 1):
        if (not flipped) and t == steps_pre + 1:
            live_env.set_hover_bonus(-100.0)
            current_bonus["value"] = -100.0
            flipped = True

        a = agent.random_action(rng) if t <= cfg.start_steps else agent.act(obs, deterministic=False)
        next_obs, r, term, trunc, info = live_env.step(a)
        agent.buffer.add(obs, a, r, next_obs, float(term))
        obs = next_obs
        ep_ret += float(r)
        ep_len += 1
        agent.total_env_steps = t

        if term or trunc:
            train_rows.append({
                "global_step": t,
                "event": "episode_end",
                "episode_return": ep_ret,
                "episode_length": ep_len,
                "alpha": float(agent.alpha.detach().cpu()),
                "hover_reward_value": current_bonus["value"],
            })
            obs, _ = live_env.reset()
            ep_ret, ep_len = 0.0, 0

        if t >= cfg.update_after and t % cfg.update_every == 0:
            for _ in range(cfg.grad_steps_per_update):
                last_update_info = agent.update()

        if t % 1000 == 0:
            row = {
                "global_step": t,
                "event": "train_step",
                "replay_size": agent.buffer.size,
                "alpha": float(agent.alpha.detach().cpu()),
                "hover_reward_value": current_bonus["value"],
            }
            row.update(last_update_info)
            train_rows.append(row)

        if t % eval_every == 0:
            _eval_now(t)

    live_env.close()

    # Final rollout should use post-switch reward value.
    final_env_fn = lambda: make_lunar_continuous(hover_bonus=-100.0)
    _save_common_artifacts(
        run_dir=run_dir,
        legacy_tag=tag,
        seed=seed,
        config=config,
        eval_log=eval_log,
        train_rows=train_rows,
        agent=agent,
        env_fn=final_env_fn,
        rollout_max_steps=None,
    )


# ---------------------------------------------------------------------------
# LL-3A: discrete SAC
# ---------------------------------------------------------------------------
def run_disc_sac(seed: int, tag: str = "disc_sac", *, total_steps: int,
                 eval_every: int, device=None, output_root: str | None = None):
    set_global_seed(seed)
    run_dir = _seed_dir(tag, seed, output_root)

    env_fn = lambda: make_lunar_discrete()
    probe = env_fn()
    obs_dim = probe.observation_space.shape[0]
    n_actions = probe.action_space.n
    probe.close()

    cfg = DiscreteSACConfig(start_steps=10_000, update_after=10_000)
    agent = DiscreteSACAgent(obs_dim, n_actions, cfg, device=device or get_device())

    config = {
        "section": "2.2_lunarlander",
        "experiment_id": tag,
        "agent": "DiscreteSAC",
        "env": "LunarLander-v3 continuous=False",
        "continuous": False,
        "seed": seed,
        "total_steps": total_steps,
        "initial_random_steps": cfg.start_steps,
        "update_after": cfg.update_after,
        "replay_buffer_size": cfg.buffer_size,
        "replay_buffer_type": "infinite-equivalent (1M)",
        "batch_size": cfg.batch_size,
        "gamma": cfg.gamma,
        "tau": cfg.tau,
        "actor_lr": cfg.actor_lr,
        "critic_lr": cfg.critic_lr,
        "alpha_lr": cfg.alpha_lr,
        "autotune_alpha": cfg.autotune_alpha,
        "init_alpha": cfg.init_alpha,
        "target_entropy_ratio": cfg.target_entropy_ratio,
        "target_entropy": cfg.target_entropy_ratio * np.log(n_actions),
        "eval_every": eval_every,
        "eval_episodes": EVAL_EPISODES,
    }

    eval_log, train_rows = train_discrete_sac(
        env_fn,
        agent,
        total_steps=total_steps,
        eval_env_fn=env_fn,
        eval_every=eval_every,
        eval_episodes=EVAL_EPISODES,
        log_stdout=True,
        seed=seed,
        best_ckpt_fn=_make_ckpt_saver(run_dir, agent),
    )

    _save_common_artifacts(
        run_dir=run_dir,
        legacy_tag=tag,
        seed=seed,
        config=config,
        eval_log=eval_log,
        train_rows=train_rows,
        agent=agent,
        env_fn=env_fn,
        rollout_max_steps=None,
    )


# ---------------------------------------------------------------------------
# LL-3B: DQN
# ---------------------------------------------------------------------------
def run_dqn(seed: int, tag: str = "dqn", *, total_steps: int,
            eval_every: int, device=None, output_root: str | None = None):
    set_global_seed(seed)
    run_dir = _seed_dir(tag, seed, output_root)

    env_fn = lambda: make_lunar_discrete()
    probe = env_fn()
    obs_dim = probe.observation_space.shape[0]
    n_actions = probe.action_space.n
    probe.close()

    cfg = DQNConfig(start_steps=10_000, update_after=10_000, grad_steps_per_update=1)
    agent = DQNAgent(obs_dim, n_actions, cfg, device=device or get_device())

    config = {
        "section": "2.2_lunarlander",
        "experiment_id": tag,
        "agent": "DQN",
        "env": "LunarLander-v3 continuous=False",
        "continuous": False,
        "seed": seed,
        "total_steps": total_steps,
        "initial_random_steps": cfg.start_steps,
        "update_after": cfg.update_after,
        "replay_buffer_size": cfg.buffer_size,
        "replay_buffer_type": "infinite-equivalent (1M)",
        "batch_size": cfg.batch_size,
        "gamma": cfg.gamma,
        "lr": cfg.lr,
        "target_update_every": cfg.target_update_every,
        "epsilon_start": cfg.epsilon_start,
        "epsilon_end": cfg.epsilon_end,
        "epsilon_decay_steps": cfg.epsilon_decay_steps,
        "eval_every": eval_every,
        "eval_episodes": EVAL_EPISODES,
    }

    eval_log, train_rows = train_dqn(
        env_fn,
        agent,
        total_steps=total_steps,
        eval_env_fn=env_fn,
        eval_every=eval_every,
        eval_episodes=EVAL_EPISODES,
        log_stdout=True,
        seed=seed,
        best_ckpt_fn=_make_ckpt_saver(run_dir, agent),
    )

    _save_common_artifacts(
        run_dir=run_dir,
        legacy_tag=tag,
        seed=seed,
        config=config,
        eval_log=eval_log,
        train_rows=train_rows,
        agent=agent,
        env_fn=env_fn,
        rollout_max_steps=None,
    )
