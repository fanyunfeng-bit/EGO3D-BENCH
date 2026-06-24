#!/usr/bin/env bash
# InternVL3-8B + visual-token compression on Ego3D-Bench (extensible harness).
# Baseline vs VisPruner on one category, with ACC + efficiency (FLOPs / KV / mem / time).
#
# Run from repo root. Examples:
#   Smoke (20/category):  bash scripts/internvl3_compress.sh --compress_method none      --limit 20
#                         bash scripts/internvl3_compress.sh --compress_method vispruner --limit 20
#   Full category:        bash scripts/internvl3_compress.sh --compress_method none
#                         bash scripts/internvl3_compress.sh --compress_method vispruner
set -e

MODEL_PATH="${MODEL_PATH:-OpenGVLab/InternVL3-8B}"   # resolved from local HF cache
MODEL_NAME="${MODEL_NAME:-InternVL3-8B}"
CATEGORY="${CATEGORY:-Object_Centric_Absolute_Distance_MultiChoice}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"          # weights+dataset cached; stay offline

python models/internvl3_compress.py \
        --model_path "$MODEL_PATH" \
        --model_name "$MODEL_NAME" \
        --category "$CATEGORY" \
        --keep_ratio 0.5 \
        --important_ratio 0.5 \
        "$@"
