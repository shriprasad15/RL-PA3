# PA3 Section 2.1 Pendulum SAC — script-based final run

This folder converts the notebook workflow into clean scripts. Training is done by Python scripts; the notebook is only for visualization/report assets.

## What this implements

- Modified `Pendulum-v1` target-angle reward:

  \[
  e_\theta = \mathrm{wrapToPi}(\theta - \theta_{target}), \qquad
  r_t = -\left(e_\theta^2 + 0.1\dot\theta^2 + 0.001u^2\right)
  \]

- Reward scaling experiment uses `r_scaled = scale * r`, with scales `10.0` and `0.1`.
- Default Pendulum torque limits are unchanged.
- SAC uses squashed Gaussian policy, clipped double Q-learning, reparameterization trick, no state-value network, Adam optimizers, and optional automatic entropy tuning.
- Evaluation starts at timestep `0`, then every `10,000` steps, with `20` deterministic episodes.
- Final plots use mean ± 95% CI over seeds.
- Every seed saves config, logs, best/final model, final rollout, and status files.

## File structure

```text
pa3_pendulum_scripts_final/
├── configs/
│   └── pendulum_final_300k.yaml
├── notebooks/
│   └── Pendulum_Visualisation.ipynb
├── scripts/
│   ├── run_01_auto.sh
│   ├── run_02_manual_search.sh
│   ├── run_03_manual_final.sh
│   ├── run_04_reward_scaling.sh
│   └── run_05_plots_and_videos.sh
├── pendulum_sac_lib.py
├── run_pendulum_jobs.py
├── make_pendulum_manifests.py
├── select_manual_alphas.py
├── make_pendulum_plots.py
├── make_pendulum_videos.py
├── requirements.txt
└── README.md
```

## Setup

From inside this folder:

```bash
python -m venv .venv-pa3
source .venv-pa3/bin/activate
pip install -r requirements.txt
```

If you already have your PA3 venv, just activate it and run `pip install -r requirements.txt`.

## Final run order

Use 4 workers first to avoid overheating. You can increase to 6 only if the laptop is stable.

```bash
export WORKERS=4
bash scripts/run_01_auto.sh outputs/pendulum_final_300k_clean
bash scripts/run_02_manual_search.sh outputs/pendulum_final_300k_clean
bash scripts/run_03_manual_final.sh outputs/pendulum_final_300k_clean
bash scripts/run_04_reward_scaling.sh outputs/pendulum_final_300k_clean
bash scripts/run_05_plots_and_videos.sh outputs/pendulum_final_300k_clean
```

## If you want a quick sanity test first

Use a separate pilot folder and fewer steps/seeds:

```bash
python make_pendulum_manifests.py --stage auto --output-root outputs/pendulum_pilot_20k --total-timesteps 20000 --seeds 0,1 --targets 0,-60
python run_pendulum_jobs.py --manifest outputs/pendulum_pilot_20k/auto_jobs.jsonl --workers 2 --start-method spawn --poll-interval 10 --status-every-steps 2000 --torch-num-threads 1
python make_pendulum_plots.py --output-root outputs/pendulum_pilot_20k
```

## Expected output layout

```text
outputs/pendulum_final_300k_clean/
├── auto_jobs.jsonl
├── manual_search_jobs.jsonl
├── manual_final_jobs.jsonl
├── reward_scaling_jobs.jsonl
├── selected_manual_alphas.json
├── manual_alpha_selection_table.csv
├── combined_eval_logs.csv
├── summary_auto_final.csv
├── summary_manual_vs_auto_final.csv
├── summary_reward_scaling_final.csv
├── plots/
│   ├── auto_all_targets_eval_return_mean.png
│   ├── manual_vs_auto_panels_eval_return_mean.png
│   ├── reward_scaling_panels_eval_return_mean.png
│   └── ...
├── videos/
│   └── *.gif
└── <condition>/seed_XX/
    ├── config.yaml
    ├── eval_log.csv
    ├── train_log.csv
    ├── best_model.pt
    ├── final_model.pt
    ├── final_rollout.pkl
    ├── status.json
    └── DONE.txt
```

## Notes

- `manual_search` uses 3 tuning seeds and the broader alpha grid `[0.01, 0.03, 0.05, 0.1, 0.2, 0.5]`.
- `select_manual_alphas.py` chooses alpha using `0.5 × normalized final return + 0.5 × normalized AUC`.
- `manual_final` and `reward_scaling` require `selected_manual_alphas.json`, so run `manual_search` first.
- `make_pendulum_plots.py --smooth-window 3` creates supplementary smoothed plots. Keep raw plots as the main result.
