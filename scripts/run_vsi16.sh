#!/usr/bin/env bash
# VSI-Bench 16-frame compression comparison for ONE model on ONE device.
# Methods: baseline + {random, vispruner, fastv, scmpruner} x keep{25,10,5}.
# Tasks: the 5 cross-view relational/spatial tasks (all multiple-choice -> ACC), FULL data:
#   object_rel_direction_{easy,medium,hard}, object_rel_distance, route_planning  (1,872 items).
# Resumable per (task) by JSONL line count, so stopping/re-running is safe (deterministic order).
#
# Usage (one per device):
#   bash scripts/run_vsi16.sh internvl3_vsibench  InternVL3-8B  OpenGVLab/InternVL3-8B       [LIMIT]
#   bash scripts/run_vsi16.sh qwen2.5_vl_vsibench Qwen2.5-VL-7B Qwen/Qwen2.5-VL-7B-Instruct  [LIMIT]
# LIMIT (optional 4th arg) = first N items/task for a quick smoke; OMIT = full data (default).
set -euo pipefail
RUNNER=$1; MODEL_NAME=$2; MODEL_PATH=$3; LIMIT=${4:-}
ITEMS=data/vsibench/vsibench_items_16f.json
TASKS="object_rel_direction_easy object_rel_direction_medium object_rel_direction_hard route_planning object_rel_distance"
export PATH=/home/fyf/miniconda3/envs/ego3d/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
LIMIT_ARG=""; [ -n "$LIMIT" ] && LIMIT_ARG="--limit $LIMIT"
COMMON="--frames 16 --items $ITEMS $LIMIT_ARG --model_name $MODEL_NAME --model_path $MODEL_PATH"

# Execution order: one SETTING (task x keep-ratio) at a time, running ALL methods for it
# consecutively (baseline + random + vispruner + fastv + scmpruner), so a complete
# method-vs-method comparison for that setting lands together -> check in periodically as
# settings finish. Baseline is ratio-independent: computed once per task, resume-skipped after.
for rp in "0.25:25" "0.10:10" "0.05:5"; do
  r=${rp%%:*}; pct=${rp##*:}
  for task in $TASKS; do
    echo ">>> SETTING  $task  keep$pct  — running all methods (baseline/random/vispruner/fastv/scmpruner)"
    python "models/$RUNNER.py" --compress_method none $COMMON --category "$task"
    for m in random vispruner fastv scmpruner; do
      python "models/$RUNNER.py" --compress_method "$m" --keep_ratio "$r" $COMMON --category "$task"
    done
    echo ">>> SETTING  $task  keep$pct  DONE  (comparison ready)"
  done
done
echo ">>> [$MODEL_NAME] ALL DONE"
