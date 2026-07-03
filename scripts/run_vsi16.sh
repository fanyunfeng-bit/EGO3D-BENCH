#!/usr/bin/env bash
# VSI-Bench 16-frame compression comparison for ONE model on ONE device.
# Methods: baseline + {random, vispruner, fastv, scmpruner} x keep{25,10,5}, all 10 tasks
# (6 multiple-choice -> ACC, 4 numeric -> MRA). Resumable per (task) by JSONL line count,
# so re-running or bumping LIMIT extends a true prefix (deterministic sample order).
#
# Usage (one per device):
#   bash scripts/run_vsi16.sh internvl3_vsibench  InternVL3-8B  OpenGVLab/InternVL3-8B       [LIMIT]
#   bash scripts/run_vsi16.sh qwen2.5_vl_vsibench Qwen2.5-VL-7B Qwen/Qwen2.5-VL-7B-Instruct  [LIMIT]
# LIMIT = first N items PER category (default 20 -> ~half a day/device; bump to 50 later, resumes).
set -euo pipefail
RUNNER=$1; MODEL_NAME=$2; MODEL_PATH=$3; LIMIT=${4:-20}
ITEMS=data/vsibench/vsibench_items_16f.json
export PATH=/home/fyf/miniconda3/envs/ego3d/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
COMMON="--frames 16 --items $ITEMS --category all --limit $LIMIT --model_name $MODEL_NAME --model_path $MODEL_PATH"

echo ">>> [$MODEL_NAME] baseline (all tokens)"
python "models/$RUNNER.py" --compress_method none $COMMON
for r in 0.25 0.10 0.05; do
  for m in random vispruner fastv scmpruner; do
    echo ">>> [$MODEL_NAME] $m keep$r"
    python "models/$RUNNER.py" --compress_method "$m" --keep_ratio "$r" $COMMON
  done
done
echo ">>> [$MODEL_NAME] DONE"
