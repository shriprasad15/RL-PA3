#!/usr/bin/env bash
# Drive the full PA3 run in the way that actually finishes fastest on an RTX 5090 rig.
#
# Strategy:
#   * Reacher (2.3) runs MuJoCo on CPU -> many workers scale well.
#   * Pendulum / LunarLander / PEBBLE (2.1, 2.2, 3) are GPU-bound once the 5090 saturates
#     -> fewer workers each gets more GPU time slice, throughput is better than cranking up.
#
# We run Reacher in parallel (shell &) with the other three, so both bottlenecks
# are exploited simultaneously.
#
# Kill safely: Ctrl-C propagates to both background jobs. Re-run this script to resume.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

echo "== Probing host =="
python run_all.py --probe

echo
echo "== Launching two parallel phases =="
echo "  Phase A: Reacher (2.3) on CPU, 16 workers"
echo "  Phase B: Pendulum / LunarLander / PEBBLE on GPU, 8 workers"
echo

mkdir -p run_logs

# Reacher: CPU-bound, scale workers high.
python run_all.py --full --workers 16 --sections 2.3 \
    >  run_logs/_phaseA_reacher.out 2>&1 &
PHASE_A_PID=$!
echo "Phase A (Reacher) launched: pid=$PHASE_A_PID -> run_logs/_phaseA_reacher.out"

# GPU-heavy sections: 8 workers gives each job ~2x the GPU slice vs 12/16 workers.
python run_all.py --full --workers 8 --sections 2.1 2.2 3 \
    >  run_logs/_phaseB_gpu.out 2>&1 &
PHASE_B_PID=$!
echo "Phase B (GPU sections) launched: pid=$PHASE_B_PID -> run_logs/_phaseB_gpu.out"

echo
echo "Monitor with:"
echo "  tail -f run_logs/_phaseB_gpu.out"
echo "  tail -f run_logs/_phaseA_reacher.out"
echo "  watch -n2 nvidia-smi"
echo "  watch -n5 'find . -path \"*/logs/*.json\" ! -name \"*.config.json\" | wc -l'"
echo

# Wait for both phases. exit 1 if either fails.
fail=0
wait "$PHASE_A_PID" || fail=1
wait "$PHASE_B_PID" || fail=1

if [[ $fail -eq 0 ]]; then
    echo "All phases completed successfully."
else
    echo "One or more phases reported failures. Inspect run_logs/*.out and run_logs/<section>__*.log"
    exit 1
fi
