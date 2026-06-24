#!/usr/bin/env bash
# Ego3D-VLM (post-training framework) with Qwen2.5-VL-7B-Instruct on Ego3D-Bench.
# Adds a textual cognitive map built from Grounding-DINO detections + Depth-Anything-V2 metric depth.
# Run from the repo root:  bash scripts/qwen_2.5_vl_ego3dvlm.sh
# Smoke test:              bash scripts/qwen_2.5_vl_ego3dvlm.sh --limit 2 --model_name Qwen2.5-VL-7B-Instruct-smoke
# Override attention:      ATTN=sdpa bash scripts/qwen_2.5_vl_ego3dvlm.sh
set -e

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
MODEL_NAME="${MODEL_NAME:-Qwen2.5-VL-7B-Instruct}"
REC_MODEL_PATH="${REC_MODEL_PATH:-IDEA-Research/grounding-dino-base}"
DEPTH_MODEL_PATH="${DEPTH_MODEL_PATH:-depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf}"
ATTN="${ATTN:-flash_attention_2}"

# Reduce CUDA fragmentation when Qwen + Grounding-DINO + Depth-Anything coexist on one GPU
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# All weights + dataset are already cached locally; stay offline to avoid network stalls
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

python models/qwen2.5_vl_ego3dvlm.py \
        --model_name "$MODEL_NAME" \
        --model_path "$MODEL_PATH" \
        --rec_model_path "$REC_MODEL_PATH" \
        --depth_model_path "$DEPTH_MODEL_PATH" \
        --attn "$ATTN" \
        "$@"
