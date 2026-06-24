#!/usr/bin/env bash
# InternVL3-8B + visual-token compression on VSI-Bench (6 uniform frames/video).
# Methods: none (baseline) | vispruner | random.  Run from repo root.
# Prep first:  python scripts/prep_vsibench.py --sources scannet,scannetpp,arkitscenes
#
# Smoke (all 10 types, 5 each):  bash scripts/internvl3_vsibench.sh --compress_method none --category all --limit 5
# Full (2 smallest MC tasks):    bash scripts/internvl3_vsibench.sh --compress_method vispruner \
#                                     --category route_planning,object_rel_direction_easy
set -e

MODEL_PATH="${MODEL_PATH:-OpenGVLab/InternVL3-8B}"
MODEL_NAME="${MODEL_NAME:-InternVL3-8B}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

python models/internvl3_vsibench.py \
        --model_path "$MODEL_PATH" \
        --model_name "$MODEL_NAME" \
        --keep_ratio 0.25 \
        --important_ratio 0.5 \
        --frames 6 \
        "$@"
