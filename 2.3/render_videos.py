"""Generate rollout videos for the trained Reacher agents.

Usage (from 2.3/):
    python render_videos.py                   # Ra/Rb/Rc seed 0, 200 steps each
    python render_videos.py --rewards Ra Rb   # only Ra and Rb
    python render_videos.py --seeds 0 1 2     # three seeds per reward
    python render_videos.py --steps 500       # longer rollouts
    python render_videos.py --out videos/     # custom output dir

Outputs:  <out_dir>/reacher_<reward>_seed<k>.mp4
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import imageio

# ── make sure we can import from this directory ────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from sac_core import SquashedGaussianActor, get_device
from reacher_env import make_reacher


# ─────────────────────────────────────────────────────────────────────────────
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


def render_rollout(
    reward_name: str,
    seed: int,
    actor_path: str,
    out_path: str,
    *,
    max_steps: int = 200,
    fps: int = 30,
    height: int = 480,
    width: int = 480,
    camera_id: int = 0,
    ignore_done: bool = False,
    device,
):
    actor = load_actor(actor_path, device)

    # Use eval env so Rc behaves with the hard cap (not the training arm-reset loop)
    env = make_reacher(reward_name, mode="eval")

    # Unwrap to DMCReacherBase
    base = env
    while not hasattr(base, "_env") or not hasattr(base._env, "physics"):
        base = base.env

    # reset() may rebuild base._env internally (new dm_control env for new seed),
    # so capture dm_physics AFTER reset to avoid a stale reference.
    obs, _ = env.reset(seed=seed)
    dm_physics = base._env.physics

    frames = []
    total_return = 0.0

    # Render initial state
    frames.append(dm_physics.render(height=height, width=width, camera_id=camera_id))

    for _ in range(max_steps):
        action = act_deterministic(actor, obs, device)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_return += reward
        # Render after step so motion is visible
        frames.append(dm_physics.render(height=height, width=width, camera_id=camera_id))
        if (terminated or truncated) and not ignore_done:
            break
        if terminated or truncated:
            # Reset env but keep recording; refresh physics reference after reset
            obs, _ = env.reset(seed=seed)
            dm_physics = base._env.physics

    env.close()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8)
    for frame in frames:
        writer.append_data(frame)
    writer.close()

    print(f"  {out_path}  ({len(frames)} frames, return={total_return:.2f})")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rewards", nargs="+", default=["Ra", "Rb", "Rc"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--ignore_done", action="store_true",
                        help="Keep recording after episode ends (env resets and continues)")
    parser.add_argument("--logs", default="logs")
    parser.add_argument("--out", default="videos")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    for reward in args.rewards:
        for seed in args.seeds:
            tag = f"reacher_{reward}_seed{seed}"
            actor_path = os.path.join(args.logs, f"{tag}_actor.pt")
            if not os.path.exists(actor_path):
                print(f"  [skip] {actor_path} not found")
                continue
            out_path = os.path.join(args.out, f"{tag}.mp4")
            print(f"Rendering {reward} seed {seed} ...")
            render_rollout(
                reward_name=reward,
                seed=seed,
                actor_path=actor_path,
                out_path=out_path,
                max_steps=args.steps,
                fps=args.fps,
                height=args.height,
                width=args.width,
                ignore_done=args.ignore_done,
                device=device,
            )

    print("Done. Videos written to:", args.out)


if __name__ == "__main__":
    main()
