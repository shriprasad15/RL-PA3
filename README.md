# DA6400 PA3 — Soft Actor-Critic (and PEBBLE bonus)

Implementations of SAC / Discrete-SAC / DQN / PEBBLE on Pendulum-v1, LunarLander-v3, and
dm_control `reacher-easy`.

## Setup

```bash
conda create -n pa3 python=3.11 -y
conda activate pa3
pip install -r requirements.txt
```

If `box2d-py` fails to build (LunarLander), install `swig` first
(`sudo apt-get install swig` / `brew install swig`), then re-install.

For dm_control with a headless machine, set `MUJOCO_GL=egl` before running.

## Layout

```
sac_core.py            # Actor/Critic networks, replay buffer, eval loop, utils
sac_agent.py           # SACAgent (continuous, squashed gaussian, clipped double-Q, auto/manual alpha)
discrete_sac_agent.py  # Discrete-SAC (categorical policy, expected-value updates)
dqn_agent.py           # Vanilla DQN w/ target net + replay
envs.py                # Pendulum theta_target wrapper, LunarLander hover wrapper, DMC Reacher reward wrappers
pebble.py              # PEBBLE: reward model ensemble, preference buffer, simulated teacher, unsup pretraining

2_1_pendulum.ipynb     # Q2.1 — SAC on modified Pendulum-v1
2_2_lunar_lander.ipynb # Q2.2 — LunarLander continuous + hover + discrete SAC vs DQN
2_3_reacher.ipynb      # Q2.3 — DMC Reacher with three reward formulations
3_bonus_pebble.ipynb   # Q3 bonus — PEBBLE on Pendulum and Reacher
```

## Reproducing

Each notebook has a `CONFIG` cell at the top with:
- `SMOKE_TEST = True/False` — if True, runs with 2 seeds and ~20K steps to sanity-check the pipeline.
- `N_SEEDS` — set to 15 for the full assignment run (2 for smoke).
- Per-experiment step budgets (tuned for convergence without waste).

Run order:
1. Set `SMOKE_TEST = True`, execute the notebook, verify curves look sane (~5 min each).
2. Set `SMOKE_TEST = False`, `N_SEEDS = 15`, re-run the "Train" cells.

Logs (per-run CSVs with eval returns vs. env steps) are saved to `logs/<notebook>/<config>_seed<k>.csv`.
Aggregate plots are re-drawn from those CSVs in the "Plots" section.

## Hardware

Code is written for an RTX 5090 (32 GB) but will run on any CUDA GPU or CPU (slower).
All agents auto-detect CUDA.

## Notes on environment versions

- `Pendulum-v1` — standard (gymnasium ≥1.0). We wrap it to implement the
  θ_target objective (reward is recomputed from the unwrapped state).
- `LunarLander-v3` — continuous version uses `continuous=True`.
- DMC `reacher-easy` via `shimmy[dm-control]` — `gym.make("dm_control/reacher-easy-v0")`.
  Target radius 0.05; action ∈ [-1,1]²; observation flattened to R¹¹.
