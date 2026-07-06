#!/usr/bin/env bash
# QA-SCMPruner vs baselines on the 5 cross-view relational VSI tasks (both models).
# Usage: bash scripts/run_scmpruner_qa.sh [MODEL] [KEEP]   MODEL=qwen|internvl  KEEP=0.10
set -euo pipefail
export PATH=/home/fyf/miniconda3/envs/ego3d/bin:$PATH
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL="${1:-qwen}"; KEEP="${2:-0.10}"
ITEMS=data/vsibench/vsibench_items_16f.json          # 16-frame manifest (frames16/ already prepped)
TASKS="object_rel_direction_easy,object_rel_direction_medium,object_rel_direction_hard,route_planning,object_rel_distance"
if [ "$MODEL" = "qwen" ]; then RUNNER=models/qwen2.5_vl_vsibench.py; else RUNNER=models/internvl3_vsibench.py; fi

# QA-SCMPruner ablation grid: signal x softweight x over-select r (K=14 fixed first)
for SIG in attn cosine; do for SW in 0 1; do for R in 7 3; do
  python "$RUNNER" --compress_method scmpruner_qa --items "$ITEMS" --frames 16 --keep_ratio "$KEEP" \
    --scm_sig "$SIG" --scm_softweight "$SW" --scm_r "$R" --category "$TASKS"
done; done; done

# baselines for the same budget (idempotent / resumable)
# NB: "random" is the registered Harness-A compress_method name (the "plain_random" name
# is a Harness-B/cvsp_curve.py alias and is NOT accepted by these VSI runners).
for M in none random vispruner scmpruner fastv; do
  python "$RUNNER" --compress_method "$M" --items "$ITEMS" --frames 16 --keep_ratio "$KEEP" --category "$TASKS" || true
done
echo "done: logs/*-scmpruner_qa-keep* + baselines"
