#!/usr/bin/env bash
# Crunch-mode full run — fits a 2-day deadline on RTX 5090 / Ultra 9.
#
# Choices vs run_full.sh:
#   * --seeds 5 instead of 15 (still gives 95% CI; defensible and fits time budget)
#   * Skip section 3 (PEBBLE bonus) — not required for 100 marks
#   * Reacher on CPU with 16 workers (dm_control is CPU-bound)
#   * GPU sections at 6 workers (5090 saturates around this; more workers just slice thinner)
#
# Expected wall-clock on the described hardware: ~14-18 hours.
# Fully resumable: Ctrl-C then re-run; skipped jobs come from disk.
#
# If wall-clock permits after this finishes, run the bonus separately:
#   python run_all.py --full --seeds 5 --workers 8 --sections 3

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

SEEDS="${SEEDS:-5}"
REACHER_WORKERS="${REACHER_WORKERS:-16}"
GPU_WORKERS="${GPU_WORKERS:-6}"

echo "== Probe =="
python run_all.py --probe

echo
echo "== Crunch plan =="
echo "  seeds:            $SEEDS   (main sections — defensible 95% CI)"
echo "  Reacher workers:  $REACHER_WORKERS   (CPU-bound MuJoCo)"
echo "  GPU workers:      $GPU_WORKERS    (5090 saturation point)"
echo "  sections:         2.1 2.2 2.3   (bonus 3 skipped; run later if time)"
echo

mkdir -p run_logs

# Phase A: Reacher on CPU. Scales with cores.
python run_all.py --full --seeds "$SEEDS" --workers "$REACHER_WORKERS" --sections 2.3 \
    > run_logs/_crunchA_reacher.out 2>&1 &
A_PID=$!
echo "Phase A (Reacher, pid=$A_PID)   -> run_logs/_crunchA_reacher.out"

# Phase B: Pendulum + LunarLander on GPU.
python run_all.py --full --seeds "$SEEDS" --workers "$GPU_WORKERS" --sections 2.1 2.2 \
    > run_logs/_crunchB_gpu.out 2>&1 &
B_PID=$!
echo "Phase B (GPU sects, pid=$B_PID) -> run_logs/_crunchB_gpu.out"

echo
echo "Monitor:"
echo "  tail -f run_logs/_crunchA_reacher.out"
echo "  tail -f run_logs/_crunchB_gpu.out"
echo "  watch -n2 nvidia-smi"
echo "  watch -n5 'find . -path \"*/logs/*.json\" ! -name \"*.config.json\" | wc -l'"

fail=0
wait "$A_PID" || fail=1
wait "$B_PID" || fail=1

if [[ $fail -eq 0 ]]; then
    echo
    echo "CRUNCH COMPLETE. Sections 2.1 / 2.2 / 2.3 done at seeds=$SEEDS."
    echo
    echo "Next (if time remains):"
    echo "  python run_all.py --full --seeds $SEEDS --workers $GPU_WORKERS --sections 3"
    echo "  (or push additional seeds e.g. seeds=10 for main sections — resumable)"
else
    echo "One or more phases reported failures. Inspect run_logs/_crunch*.out"
    exit 1
fi
