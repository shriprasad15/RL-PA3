#!/bin/bash
# Auto-redistribute workers when rl-1 finishes
# Watches rl-1 process, when it dies → restart rl-2 with 10 workers, rl-3 with 5 workers

echo "Watching rl-1... will redistribute when it completes."

while true; do
    # Check if allinone_bad6.py is still running
    if ! pgrep -f "allinone_bad6.py" > /dev/null 2>&1; then
        echo ""
        echo "=== rl-1 DONE at $(date '+%H:%M:%S') ==="
        echo "Redistributing workers..."
        sleep 5

        # Kill rl-2 and rl-3
        tmux send-keys -t rl-2 C-c
        sleep 3
        tmux send-keys -t rl-3 C-c
        sleep 3

        # Restart with more workers
        tmux send-keys -t rl-2 "python parallel_run_15seeds.py --workers 10 2>&1 | tee logs_15seeds/parallel_run2.log" ENTER
        tmux send-keys -t rl-3 "python parallel_run_15seeds.py --workers 5 2>&1 | tee logs_15seeds/parallel_run2.log" ENTER

        echo "rl-2 restarted with 10 workers"
        echo "rl-3 restarted with 5 workers"
        echo "Done. Exiting watcher."
        exit 0
    fi
    sleep 30
done
