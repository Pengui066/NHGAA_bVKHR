#!/usr/bin/env bash
# Supervised training runner: relaunches nn/train.py with --resume whenever
# it dies from an environment event (GPU contention, driver events, RDP
# reconnects on this shared desktop machine). Gives up only if three
# consecutive crashes happen within a minute of startup each (real bug).
cd "$(dirname "$0")/.."

CRASHES=0
while true; do
    START=$(date +%s)
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python nn/train.py \
        --data dumps/prepared_full --out runs/psi_v1 --steps 50000 \
        --micro-batch 2 --accum 4 --patch-size 512 --lr 2e-4 --resume
    CODE=$?
    if [ $CODE -eq 0 ]; then
        echo "supervisor: training finished cleanly"
        break
    fi
    NOW=$(date +%s)
    if [ $((NOW - START)) -lt 60 ]; then
        CRASHES=$((CRASHES + 1))
    else
        CRASHES=0
    fi
    if [ $CRASHES -ge 3 ]; then
        echo "supervisor: 3 fast consecutive crashes - giving up (deterministic bug?)"
        exit 1
    fi
    echo "supervisor: training exited with $CODE after $((NOW - START))s, resuming in 30s..."
    sleep 30
done
