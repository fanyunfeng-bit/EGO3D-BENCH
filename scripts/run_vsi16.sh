#!/usr/bin/env bash
# VSI-Bench compression comparison for ONE model on ONE device (frame count via FRAMES env).
# Methods: baseline + {random, vispruner, fastv, scmpruner} x keep{25,10,5}.
# Tasks: the 5 cross-view relational/spatial tasks (all multiple-choice -> ACC), FULL data:
#   object_rel_direction_{easy,medium,hard}, object_rel_distance, route_planning  (1,872 items).
# Resumable per (task) by JSONL line count, so stopping/re-running is safe (deterministic order).
#
# Usage (one per device):
#   bash scripts/run_vsi16.sh internvl3_vsibench  InternVL3-8B  OpenGVLab/InternVL3-8B       [LIMIT]
#   bash scripts/run_vsi16.sh qwen2.5_vl_vsibench Qwen2.5-VL-7B Qwen/Qwen2.5-VL-7B-Instruct  [LIMIT]
# LIMIT (optional 4th arg) = first N items/task for a quick smoke; OMIT = full data (default).
# FRAMES env var (default 16) selects the prepped items file (needs vsibench_items_<F>f.json).
# NOTE: no `set -e` -> a single failed run is logged and skipped (resumable); the batch continues.
set -uo pipefail
RUNNER=$1; MODEL_NAME=$2; MODEL_PATH=$3; LIMIT=${4:-}
FRAMES=${FRAMES:-16}
ITEMS=data/vsibench/vsibench_items_${FRAMES}f.json
TASKS="object_rel_direction_easy object_rel_direction_medium object_rel_direction_hard route_planning object_rel_distance"
export PATH=/home/fyf/miniconda3/envs/ego3d/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
LIMIT_ARG=""; [ -n "$LIMIT" ] && LIMIT_ARG="--limit $LIMIT"
COMMON="--frames $FRAMES --items $ITEMS $LIMIT_ARG --model_name $MODEL_NAME --model_path $MODEL_PATH"

# Execution order: one SETTING (task x keep-ratio) at a time, running ALL methods for it
# consecutively (baseline + random + vispruner + fastv + scmpruner), so a complete
# method-vs-method comparison for that setting lands together -> check in periodically as
# settings finish. Baseline is ratio-independent: computed once per task, resume-skipped after.
for rp in "0.25:25" "0.10:10" "0.05:5"; do
  r=${rp%%:*}; pct=${rp##*:}
  for task in $TASKS; do
    echo ">>> SETTING  $task  keep$pct  — running all methods (baseline/random/vispruner/fastv/scmpruner)"
    python "models/$RUNNER.py" --compress_method none $COMMON --category "$task" \
      || echo ">>> !! FAILED baseline $task (logged, continuing)"
    for m in random vispruner fastv scmpruner; do
      python "models/$RUNNER.py" --compress_method "$m" --keep_ratio "$r" $COMMON --category "$task" \
        || echo ">>> !! FAILED $m keep$pct $task (logged, continuing)"
    done
    echo ">>> SETTING  $task  keep$pct  DONE  (comparison ready)"
  done
done
echo ">>> [$MODEL_NAME] ALL DONE"
