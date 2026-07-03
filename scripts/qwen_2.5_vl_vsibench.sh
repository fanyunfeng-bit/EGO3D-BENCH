#!/usr/bin/env bash
# Qwen2.5-VL-7B + visual-token compression on VSI-Bench (6 uniform frames, 448px each).
# Methods: none (baseline) | vispruner | random.  Run from repo root.
# Prep first:  python scripts/prep_vsibench.py --sources scannet,scannetpp,arkitscenes
#
#   bash scripts/qwen_2.5_vl_vsibench.sh --compress_method none      --category route_planning
#   bash scripts/qwen_2.5_vl_vsibench.sh --compress_method vispruner --category route_planning
set -e

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
MODEL_NAME="${MODEL_NAME:-Qwen2.5-VL-7B}"
ATTN="${ATTN:-flash_attention_2}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

python models/qwen2.5_vl_vsibench.py \
        --model_path "$MODEL_PATH" \
        --model_name "$MODEL_NAME" \
        --attn "$ATTN" \
        --keep_ratio 0.25 \
        --important_ratio 0.5 \
        --frames 6 \
        "$@"
