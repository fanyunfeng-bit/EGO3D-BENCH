#!/usr/bin/env bash
# QA-SCMPruner ablation — Qwen2.5-VL-7B, VSI-Bench 16-frame (uniform-spaced), 5 relational tasks.
#   grid: keep{25,10,5} x stage2-signal{attn,cosine} x stage1-query-aware{off,on}  (r=7, K=14 defaults)
# no_think prompt (built into the runner) + robust \b[a-d]\b MC scoring.
# Also restores the two keep10 baselines (scmpruner, fastv) that earlier --limit 2 smokes clobbered.
# Resumable per (config, task) by JSONL line count. Output: logs/Qwen2.5-VL-7B-scmpruner_qa-keep<NN>[suffix]-vsibench/
set -uo pipefail
export PATH=/home/fyf/miniconda3/envs/ego3d/bin:$PATH
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TQDM_MININTERVAL=30   # throttle progress-bar log spam (was 311KB before)
cd /home/fyf/fyf/Research/Multi-View-Compression/Ego3D-Bench

ITEMS=data/vsibench/vsibench_items_16f.json
TASKS="object_rel_direction_easy,object_rel_direction_medium,object_rel_direction_hard,route_planning,object_rel_distance"
R=models/qwen2.5_vl_vsibench.py

run(){ echo "===== $(date +%F_%H:%M:%S) :: $* ====="; python "$R" --items "$ITEMS" --frames 16 --category "$TASKS" "$@" || echo "!!! FAILED: $*"; }

echo "########## QA-SCMPruner ablation START $(date) ##########"

# ---- 12 QA configs (the ablation) ----
for KEEP in 0.25 0.10 0.05; do
  for SIG in attn cosine; do
    for SW in 0 1; do
      run --compress_method scmpruner_qa --keep_ratio "$KEEP" --scm_sig "$SIG" --scm_softweight "$SW"
    done
  done
done

# ---- restore the two clobbered keep10 baselines (were 2-row smokes) ----
run --compress_method scmpruner --keep_ratio 0.10
run --compress_method fastv     --keep_ratio 0.10

echo "########## QA-SCMPruner ablation ALL DONE $(date) ##########"
