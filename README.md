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
- `N_SEEDS` — 15 for the full assignment run (2 for smoke).
- Per-experiment step budgets tuned for convergence without waste.

### Option A: notebook-driven (single-process, one section at a time)

```bash
cd 2.1 && jupyter lab   # then run all cells
```

Sequential; skips cached runs. Fine for a sanity check, slow for 15-seed runs.

### Option B: parallel CLI (run everything in parallel)

Top-level `run_all.py` dispatches every `(experiment, seed)` combo as an
independent subprocess. Each worker runs one full training job end-to-end and
writes the exact same `<section>/logs/<tag>_seed<k>.json` the notebook would.

```bash
# Smoke test across all four sections (2 seeds, short budgets)
python run_all.py --smoke --workers 4

# Full assignment run (15 seeds, TA-mandated budgets)
python run_all.py --full --workers 6

# Only section 2.3 with the full budget
python run_all.py --full --sections 2.3 --workers 4

# Enumerate jobs without launching (sanity check)
python run_all.py --full --dry-run
```

Everything is resumable: any job whose output JSON already exists is skipped.
Per-job stdout/stderr goes to `run_logs/<section>__<tag>__seed<k>.log`.

After the CLI finishes, open the matching notebook and just run the "Plot"
cells — they read the same JSONs the CLI wrote. No re-training.

**Workers tuning on the RTX 5090:**
- `--workers 6` is a safe default (each SAC agent uses <1 GB VRAM).
- Reacher (MuJoCo) is more CPU-bound; you can push `--workers 8` on a 24-core CPU.
- If you see CUDA OOM, drop `--workers`.

Full-run catalogue sizes: 212 jobs for 2.1, 75 for 2.2, 45 for 2.3, 240 for 3
(572 total). On an RTX 5090 most jobs run in 10-30 minutes each; expect
end-to-end wall-clock of ~24-48 hours at `--workers 6` for everything.

## Hardware

Code is written for an RTX 5090 (32 GB) but will run on any CUDA GPU or CPU (slower).
All agents auto-detect CUDA.

## Notes on environment versions

- `Pendulum-v1` — standard (gymnasium ≥1.0). We wrap it to implement the
  θ_target objective (reward is recomputed from the unwrapped state).
- `LunarLander-v3` — continuous version uses `continuous=True`.
- DMC `reacher-easy` via `shimmy[dm-control]` — `gym.make("dm_control/reacher-easy-v0")`.
  Target radius 0.05; action ∈ [-1,1]²; observation flattened to R¹¹.
