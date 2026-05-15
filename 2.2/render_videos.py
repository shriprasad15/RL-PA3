"""Generate rollout videos for trained LunarLander agents.

Usage (from 2.2/):
    python render_videos.py                          # all experiments, seed 0
    python render_videos.py --experiments cont_auto  # just one
    python render_videos.py --seeds 0 1 2            # multiple seeds
    python render_videos.py --steps 500 --fps 30
    python render_videos.py --out videos/

Experiments: cont_auto, hover_auto, hover_fixed
(disc_sac and dqn use discrete actions — rendered separately via their own actor)

Outputs: <out_dir>/<experiment>_seed<k>.mp4
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import imageio
import gymnasium as gym

sys.path.insert(0, os.path.dirname(__file__))
from sac_core import SquashedGaussianActor, get_device

RUNS_DIR = "runs"
EXPERIMENTS = ["cont_auto", "hover_auto", "hover_fixed"]


def load_actor(path: str, device):
    payload = torch.load(path, map_location=device)
    actor = SquashedGaussianActor(
        payload["obs_dim"],
        payload["act_dim"],
        payload["act_limit"],
        tuple(payload.get("hidden", (256, 256))),
    ).to(device)
    actor.load_state_dict(payload["actor"])
    actor.eval()
    return actor


@torch.no_grad()
def act_deterministic(actor, obs: np.ndarray, device) -> np.ndarray:
    o = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    a, _ = actor(o, deterministic=True, with_logprob=False)
    return a.cpu().numpy()[0]


def render_rollout(experiment: str, seed: int, actor_path: str, out_path: str, *,
                   max_steps: int = 1000, fps: int = 30,
                   height: int = 480, width: int = 480, device):
    actor = load_actor(actor_path, device)

    env = gym.make("LunarLander-v3", continuous=True, render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)

    frames = []
    total_return = 0.0
    frames.append(env.render())

    for _ in range(max_steps):
        action = act_deterministic(actor, obs, device)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_return += reward
        frames.append(env.render())
        if terminated or truncated:
            break

    env.close()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8)
    for frame in frames:
        writer.append_data(frame)
    writer.close()

    print(f"  {out_path}  ({len(frames)} frames, return={total_return:.2f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="+", default=EXPERIMENTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--checkpoint", default="best_model.pt",
                        help="Checkpoint filename inside runs/<exp>/seed_XX/")
    parser.add_argument("--out", default="videos")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    for exp in args.experiments:
        for seed in args.seeds:
            seed_str = f"seed_{seed:02d}"
            actor_path = os.path.join(RUNS_DIR, exp, seed_str, args.checkpoint)
            if not os.path.exists(actor_path):
                print(f"  [skip] {actor_path} not found")
                continue
            out_path = os.path.join(args.out, f"{exp}_{seed_str}.mp4")
            print(f"Rendering {exp} {seed_str} ...")
            render_rollout(
                experiment=exp, seed=seed,
                actor_path=actor_path,
                out_path=out_path,
                max_steps=args.steps,
                fps=args.fps,
                height=args.height,
                width=args.width,
                device=device,
            )

    print("Done. Videos written to:", args.out)


if __name__ == "__main__":
    main()
