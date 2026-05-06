#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT=${1:-outputs/pendulum_final_300k_clean}
WORKERS=${WORKERS:-4}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python make_pendulum_manifests.py --stage auto --output-root "$OUTPUT_ROOT" --total-timesteps 300000
python run_pendulum_jobs.py --manifest "$OUTPUT_ROOT/auto_jobs.jsonl" --workers "$WORKERS" --start-method spawn --poll-interval 30 --status-every-steps 5000 --torch-num-threads 1
