#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT=${1:-outputs/pendulum_final_300k_clean}

python make_pendulum_plots.py --output-root "$OUTPUT_ROOT" --smooth-window 1
python make_pendulum_plots.py --output-root "$OUTPUT_ROOT" --smooth-window 3
python make_pendulum_videos.py --output-root "$OUTPUT_ROOT" --include-manual --include-scaling --ext gif --checkpoint best_model.pt
