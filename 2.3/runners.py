"""Training entry points for PA3 Section 2.3 Reacher with per-seed artifacts.

Each run saves:

    runs/<tag>/seed_XX/
        config.json
        eval_log.csv
        eval_log.json
        train_log.csv
        train_log.json
        best_model.pt
        final_model.pt
        final_actor.pt
        final_rollout.pkl
        DONE.txt

For compatibility with existing notebooks/run_all.py, it also writes:

    logs/<tag>_seed<seed>.json
    logs/<tag>_seed<seed>.config.json
    logs/<tag>_seed<seed>_actor.pt
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import torch

from sac_core import (
    set_global_seed,
    save_json,
    save_log,
    ensure_dir,
    collect_rollout,
    get_device,
)
from sac_agent import SACAgent, SACConfig, train_sac
from reacher_env import make_reacher

LOG_DIR = "logs"
DEFAULT_OUTPUT_ROOT = "runs"
EVAL_EPISODES = 20
REWARDS = ("Ra", "Rb", "Rc")


def seed_dir(output_root: str | Path, tag: str, seed: int) -> Path:
    return Path(output_root) / tag / f"seed_{seed:02d}"


def done_path(output_root: str | Path, tag: str, seed: int) -> Path:
    return seed_dir(output_root, tag, seed) / "DONE.txt"


def is_done(output_root: str | Path, tag: str, seed: int) -> bool:
    return done_path(output_root, tag, seed).exists()


def _actor_only_checkpoint(full_ckpt: dict) -> dict:
    return {
        "actor": full_ckpt["actor"],
        "obs_dim": full_ckpt["obs_dim"],
        "act_dim": full_ckpt["act_dim"],
        "act_limit": full_ckpt["act_limit"],
        "hidden": full_ckpt.get("hidden", (256, 256)),
        "total_env_steps": full_ckpt.get("total_env_steps"),
        "alpha": full_ckpt.get("alpha"),
    }


def _save_run_artifacts(
    *,
    out_dir: Path,
    config: dict,
    eval_log,
    train_log,
    best_ckpt: dict,
    final_ckpt: dict,
    rollout: dict | None,
):
    ensure_dir(out_dir)
    save_json(config, out_dir / "config.json")
    eval_log.to_csv(out_dir / "eval_log.csv")
    eval_log.to_json(out_dir / "eval_log.json")
    train_log.to_csv(out_dir / "train_log.csv")
    train_log.to_json(out_dir / "train_log.json")
    torch.save(best_ckpt, out_dir / "best_model.pt")
    torch.save(final_ckpt, out_dir / "final_model.pt")
    torch.save(_actor_only_checkpoint(final_ckpt), out_dir / "final_actor.pt")

    if rollout is not None:
        with open(out_dir / "final_rollout.pkl", "wb") as f:
            pickle.dump(rollout, f)

    with open(out_dir / "DONE.txt", "w") as f:
        f.write("done\n")


def run_reacher(
    seed: int,
    reward_name: str,
    tag: str,
    *,
    total_steps: int,
    eval_every: int,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    device=None,
    save_rollout: bool = True,
):
    """Train SAC on one reward formulation and evaluate on all three rewards."""
    assert reward_name in REWARDS
    set_global_seed(seed)
    out_dir = seed_dir(output_root, tag, seed)

    train_env_fn = lambda: make_reacher(reward_name, mode="train")

    # True cross-evaluation in separate evaluation environments.
    eval_env_fns = {
        "Ra_eval": lambda: make_reacher("Ra", mode="eval"),
        "Rb_eval": lambda: make_reacher("Rb", mode="eval"),
        "Rc_eval": lambda: make_reacher("Rc", mode="eval"),
    }
    primary_eval_key = f"{reward_name}_eval"

    probe_env = train_env_fn()
    obs_dim = probe_env.observation_space.shape[0]
    act_dim = probe_env.action_space.shape[0]
    act_limit = float(probe_env.action_space.high[0])
    probe_env.close()

    cfg = SACConfig(start_steps=10_000, update_after=10_000)
    agent = SACAgent(obs_dim, act_dim, act_limit, cfg, device=device or get_device())
    best_ckpt_path = os.path.join(LOG_DIR, f"{tag}_seed{seed}_best.pt")

    def _save_best(step, ret):
        torch.save({
            "actor": agent.actor.state_dict(), "critic": agent.critic.state_dict(),
            "critic_target": agent.critic_target.state_dict(),
            "log_alpha": agent.log_alpha.detach().cpu(),
            "step": step, "best_return": ret,
            "obs_dim": obs_dim, "act_dim": act_dim, "act_limit": act_limit,
            "reward_formulation": reward_name, "seed": seed,
        }, best_ckpt_path)

    eval_log, train_log, best_ckpt, final_ckpt = train_sac(
        train_env_fn,
        agent,
        total_steps=total_steps,
        eval_every=eval_every,
        eval_episodes=EVAL_EPISODES,
        eval_env_fns=eval_env_fns,
        primary_eval_key=primary_eval_key,
        log_stdout=False,
        seed=seed,
        best_ckpt_fn=_save_best,
    )

    rollout = None
    if save_rollout:
        # Store one deterministic rollout under the training reward, mainly for qualitative checks.
        # Use 1000 steps to keep per-seed artifacts lightweight.
        rollout = collect_rollout(
            lambda: make_reacher(reward_name, mode="eval"),
            lambda o: agent.act(o, deterministic=True),
            seed=50_000 + seed,
            max_steps=1000,
        )

    run_config = {
        "section": "2.3_reacher",
        "tag": tag,
        "env": "dm_control reacher-easy",
        "reward_formulation": reward_name,
        "seed": seed,
        "total_steps": total_steps,
        "initial_random_steps": cfg.start_steps,
        "update_after": cfg.update_after,
        "replay_buffer_size": cfg.buffer_size,
        "replay_buffer_type": "fixed-size circular (1M)",
        "batch_size": cfg.batch_size,
        "gamma": cfg.gamma,
        "tau": cfg.tau,
        "actor_lr": cfg.actor_lr,
        "critic_lr": cfg.critic_lr,
        "alpha_lr": cfg.alpha_lr,
        "autotune_alpha": cfg.autotune_alpha,
        "init_alpha": cfg.init_alpha,
        "target_entropy": "-action_dim",
        "eval_every": eval_every,
        "eval_episodes": EVAL_EPISODES,
        "cross_eval_keys": ["Ra_eval", "Rb_eval", "Rc_eval"],
        "rc_timeout_steps": 1000,
        "rc_timeout_penalty": -20.0,
        "rc_goal_velocity_threshold": 0.5,
        "rc_eval_timeout_return": -1020,
        "final_diagnostic_episode_length": 5000,
        "artifact_files": [
            "config.json", "eval_log.csv", "eval_log.json",
            "train_log.csv", "train_log.json",
            "best_model.pt", "final_model.pt", "final_actor.pt",
            "final_rollout.pkl", "DONE.txt",
        ],
    }

    _save_run_artifacts(
        out_dir=out_dir,
        config=run_config,
        eval_log=eval_log,
        train_log=train_log,
        best_ckpt=best_ckpt,
        final_ckpt=final_ckpt,
        rollout=rollout,
    )

    # Backward compatibility with the existing notebook/run_all.py log convention.
    ensure_dir(LOG_DIR)
    legacy_path = Path(LOG_DIR) / f"{tag}_seed{seed}.json"
    save_log(eval_log, legacy_path, config=run_config)
    torch.save(_actor_only_checkpoint(final_ckpt), Path(LOG_DIR) / f"{tag}_seed{seed}_actor.pt")

    return out_dir
