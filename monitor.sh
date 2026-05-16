#!/bin/bash
# Monitor all 3 RL sessions and alert on failure or completion
# Run in a separate terminal: bash monitor.sh

RL1_LOG="/research/assignments/RL/RL-PA3/2.1/new_trial/bad_angles_torque6.log"
RL2_LOG="/research/assignments/RL/RL-PA3/2.2/logs_15seeds/parallel_run.log"
RL3_LOG="/research/assignments/RL/RL-PA3/2.3/logs_15seeds/parallel_run.log"
RL1_OUT="/research/assignments/RL/RL-PA3/2.1/new_trial/outputs/pendulum_bad_angles_torque6"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

alert() {
    echo -e "${RED}🚨 ALERT: $1${NC}"
    # Terminal bell
    echo -e "\a"
}

ok() {
    echo -e "${GREEN}✅ $1${NC}"
}

info() {
    echo -e "${CYAN}ℹ  $1${NC}"
}

prev_rl2=0
prev_rl3=0
prev_rl1_done=0
rl1_done=false
rl2_done=false
rl3_done=false

echo -e "${CYAN}=== RL Monitor started at $(date '+%H:%M:%S') ===${NC}"
echo "Checking every 60 seconds. Press Ctrl+C to stop."
echo ""

while true; do
    now=$(date '+%H:%M:%S')

    # ---- rl-1: 2.1 bad angles ----
    rl1_done_count=$(find "$RL1_OUT" -name "DONE.txt" 2>/dev/null | wc -l)
    rl1_failed=$(grep -c "FAILED\|Error\|failed=[^0]" "$RL1_LOG" 2>/dev/null | tail -1 || echo 0)
    rl1_last=$(tail -1 "$RL1_LOG" 2>/dev/null)

    # Check if rl-1 process is alive
    rl1_pid=$(pgrep -f "allinone_bad6.py" | head -1)

    if [ -z "$rl1_pid" ] && [ "$rl1_done" = "false" ]; then
        if echo "$rl1_last" | grep -q "DONE\|Output saved"; then
            ok "[$now] rl-1 (2.1 bad angles) COMPLETED! ($rl1_done_count DONE files)"
            rl1_done=true
        else
            alert "[$now] rl-1 (2.1 bad angles) PROCESS DIED unexpectedly!"
            alert "Last log line: $rl1_last"
        fi
    fi

    if echo "$rl1_last" | grep -q "failed=[1-9][0-9]*" && [ "$rl1_done" = "false" ]; then
        failed_n=$(echo "$rl1_last" | grep -o "failed=[0-9]*" | tail -1 | cut -d= -f2)
        if [ "$failed_n" -gt 5 ] 2>/dev/null; then
            alert "[$now] rl-1 has $failed_n failed jobs! Check bad_angles_torque6.log"
        fi
    fi

    # ---- rl-2: 2.2 LunarLander ----
    rl2_total=0
    for exp in cont_auto hover_fixed hover_auto disc_sac dqn; do
        n=$(ls /research/assignments/RL/RL-PA3/2.2/logs_15seeds/${exp}_seed*.json 2>/dev/null | grep -v config | wc -l)
        rl2_total=$((rl2_total + n))
    done

    rl2_pid=$(pgrep -f "parallel_run_15seeds.py" | grep "2.2\|RL-PA3/2.2" | head -1)
    # fallback: check by matching cwd
    if [ -z "$rl2_pid" ]; then
        rl2_pid=$(ls /proc/*/cwd 2>/dev/null | xargs -I{} sh -c 'readlink {} 2>/dev/null' | grep -l "RL-PA3/2.2" 2>/dev/null | head -1 | grep -o '[0-9]*')
    fi

    if [ "$rl2_total" -gt "$prev_rl2" ]; then
        new=$((rl2_total - prev_rl2))
        ok "[$now] rl-2 (2.2 LunarLander): +$new seeds done → $rl2_total/75 total"
        prev_rl2=$rl2_total
    fi

    if [ "$rl2_total" -ge 75 ] && [ "$rl2_done" = "false" ]; then
        ok "[$now] rl-2 (2.2 LunarLander) ALL DONE! 75/75 seeds complete 🎉"
        rl2_done=true
    fi

    # ---- rl-3: 2.3 Reacher ----
    rl3_total=0
    for r in Ra Rb Rc; do
        n=$(ls /research/assignments/RL/RL-PA3/2.3/logs_15seeds/reacher_${r}_seed*.json 2>/dev/null | grep -v config | wc -l)
        rl3_total=$((rl3_total + n))
    done

    if [ "$rl3_total" -gt "$prev_rl3" ]; then
        new=$((rl3_total - prev_rl3))
        ok "[$now] rl-3 (2.3 Reacher): +$new seeds done → $rl3_total/45 total"
        prev_rl3=$rl3_total
    fi

    if [ "$rl3_total" -ge 45 ] && [ "$rl3_done" = "false" ]; then
        ok "[$now] rl-3 (2.3 Reacher) ALL DONE! 45/45 seeds complete 🎉"
        rl3_done=true
    fi

    # ---- Check rl-2 and rl-3 tmux sessions still alive ----
    rl2_tmux=$(tmux list-panes -t rl-2 2>/dev/null | wc -l)
    rl3_tmux=$(tmux list-panes -t rl-3 2>/dev/null | wc -l)

    if [ "$rl2_tmux" -eq 0 ] && [ "$rl2_done" = "false" ]; then
        alert "[$now] rl-2 tmux session GONE! ($rl2_total/75 done)"
    fi
    if [ "$rl3_tmux" -eq 0 ] && [ "$rl3_done" = "false" ]; then
        alert "[$now] rl-3 tmux session GONE! ($rl3_total/45 done)"
    fi

    # ---- Periodic status every 10 min ----
    minute=$(date '+%M')
    if [ "$((10#$minute % 10))" -eq 0 ]; then
        info "[$now] STATUS → rl-1: $rl1_done_count/228 | rl-2: $rl2_total/75 | rl-3: $rl3_total/45"
    fi

    sleep 60
done
