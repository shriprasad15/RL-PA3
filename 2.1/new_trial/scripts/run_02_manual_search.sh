#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT=${1:-outputs/pendulum_final_300k_clean}
WORKERS=${WORKERS:-4}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python make_pendulum_manifests.py --stage manual_search --output-root "$OUTPUT_ROOT" --total-timesteps 300000 --alpha-grid 0.01,0.03,0.05,0.1,0.2,0.5 --tuning-seeds 0,1,2
python run_pendulum_jobs.py --manifest "$OUTPUT_ROOT/manual_search_jobs.jsonl" --workers "$WORKERS" --start-method spawn --poll-interval 30 --status-every-steps 5000 --torch-num-threads 1
python select_manual_alphas.py --output-root "$OUTPUT_ROOT" --tuning-seeds 0,1,2
