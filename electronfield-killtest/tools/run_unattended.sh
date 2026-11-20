#!/usr/bin/env bash
# Runs every remaining pass with no Claude session attached. Safe to start more than once: every pass
# skips structures that already have a result. See research/UNATTENDED_RUN.md.
#   core 0: benchmark PBE run (paced) and semilocal pass
#   core 1: benchmark response pass, all sizes
#   core 2: outside reaction set in size tiers, then status reports every 2 hours
# All numerics on the GPU; three host cores, one thread each; all data on the D drive.
L=/mnt/c/Users/breth/myrepo/electronfield-killtest/tools/linux_gpu_core.sh
LOG=/mnt/d/xcresearch/logs
ST=/mnt/d/xcresearch/status
mkdir -p "$LOG" "$ST"
exec 9>"$ST/runner.lock"
flock -n 9 || { echo "$(date) runner already active; not starting a second one" >> "$LOG/unattended.log"; exit 0; }
rm -f "$ST"/*.done
echo "$(date) unattended runner started (pid $$)" >> "$LOG/unattended.log"

retry() {   # retry <max attempts> <command...>: restart a pass if its process dies
    local max=$1; shift
    local i=0
    until "$@"; do
        i=$((i + 1))
        echo "$(date) pass exited abnormally (attempt $i of $max): $*" >> "$LOG/unattended.log"
        [ "$i" -ge "$max" ] && return 1
        sleep 30
    done
}

status() {   # plain-text snapshot of coverage and model errors
    local out="$ST/status_$(date +%m%d_%H%M).txt"
    {
        echo "Snapshot $(date)"
        for d in gmtkn55 tae25 accdb; do
            r=/mnt/d/xcresearch/$d/pbe_qzvp_g3
            echo "$d: scf $(ls $r/results 2>/dev/null | grep -c json$)  semilocal $(ls $r/semi_v1/results 2>/dev/null | grep -c json$)  response $(ls $r/corr_v1/results 2>/dev/null | grep -c json$)"
        done
        echo "result files recording an error: $(grep -l '"error"' /mnt/d/xcresearch/*/pbe_qzvp_g3/results/*.json /mnt/d/xcresearch/*/pbe_qzvp_g3/*/results/*.json 2>/dev/null | wc -l)"
        echo; echo "== benchmark-only fits =="
        bash $L 2 -m xcgpu.fit --only "N:,P:" --lams 1e-7,1e-6,1e-5 2>&1 | grep -v "Extension modules"
        echo; echo "== trained outside the benchmark, tested on it =="
        bash $L 2 -m xcgpu.fit_general --models "N:,P:" --lams 1e-6,1e-5,1e-4,1e-3 2>&1 | grep -v "Extension modules"
        if [ "$1" = final ]; then
            echo; echo "== official pinned scorer, in-sample model N =="
            bash $L 2 -m xcgpu.score_model --model "N:" --name final_model_N 2>&1 | grep -v "Extension modules"
            echo; echo "== official pinned scorer, plain PBE =="
            bash $L 2 -m xcgpu.score_model --model "PBE" --name final_pbe 2>&1 | grep -v "Extension modules"
        fi
    } > "$out" 2>&1
    cp "$out" "$ST/latest.txt"
}

( retry 8 bash $L 0 -m xcgpu.run_scf --tag pbe_qzvp_g3 --pace-with corr_v1 --max-lead 40 >> "$LOG/pbe_qzvp_g3.log" 2>&1; touch "$ST/scf.done" ) &
( retry 8 bash $L 0 -m xcgpu.run_semi --scf-tag pbe_qzvp_g3 --tag semi_v1 --follow >> "$LOG/semi_v1.log" 2>&1; touch "$ST/semi.done" ) &
( retry 8 bash $L 1 -m xcgpu.run_corr --scf-tag pbe_qzvp_g3 --tag corr_v1 --follow >> "$LOG/corr_v1.log" 2>&1; touch "$ST/corr.done" ) &
(
    for M in 200 300 400 500; do
        retry 4 bash $L 2 -m xcgpu.run_scf --dataset accdb --tag pbe_qzvp_g3 --max-nao $M >> "$LOG/outside_chain.log" 2>&1
        retry 4 bash $L 2 -m xcgpu.run_semi --dataset accdb --scf-tag pbe_qzvp_g3 --max-nao $M >> "$LOG/outside_chain.log" 2>&1
        retry 4 bash $L 2 -m xcgpu.run_corr --dataset accdb --scf-tag pbe_qzvp_g3 --max-nao $M >> "$LOG/outside_chain.log" 2>&1
        echo "outside reaction tier up to $M basis functions finished at $(date)" >> "$LOG/outside_chain.log"
    done
    touch "$ST/outside.done"
) &

while true; do
    sleep 7200
    status
    if [ -f "$ST/scf.done" ] && [ -f "$ST/semi.done" ] && [ -f "$ST/corr.done" ] && [ -f "$ST/outside.done" ]; then
        status final
        echo "$(date) all passes finished; final report in $ST/latest.txt" >> "$LOG/unattended.log"
        break
    fi
done
wait
