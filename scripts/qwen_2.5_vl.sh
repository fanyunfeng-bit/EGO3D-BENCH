#!/usr/bin/env bash
# Baseline benchmark: Qwen2.5-VL-7B-Instruct on Ego3D-Bench.
# Run from the repo root:  bash scripts/qwen_2.5_vl.sh
# Smoke test (2 samples/category):  bash scripts/qwen_2.5_vl.sh --limit 2 --model_name Qwen2.5-VL-7B-Instruct-smoke
# Override the attention backend:   ATTN=sdpa bash scripts/qwen_2.5_vl.sh
set -e

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"   # resolved from local HF cache
MODEL_NAME="${MODEL_NAME:-Qwen2.5-VL-7B-Instruct}"
ATTN="${ATTN:-flash_attention_2}"                          # flash_attention_2 | sdpa | eager

# Reduce CUDA fragmentation (reclaims reserved-but-unallocated memory near the cap)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# All weights + dataset are already cached locally; stay offline to avoid network stalls
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

python models/qwen2.5_vl.py \
        --model_name "$MODEL_NAME" \
        --model_path "$MODEL_PATH" \
        --attn "$ATTN" \
        "$@"
