#!/usr/bin/env bash
# MVPruner (arXiv:2606.27660, faithful port) on Qwen2.5-VL, VSI-Bench 6 groups (8 types), full 16f.
# keep25/10/5. baseline/random/vispruner/fastv already exist at this config -> reused for scoring.
set -uo pipefail
export PATH=/home/fyf/miniconda3/envs/ego3d/bin:$PATH
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TQDM_MININTERVAL=30
cd /home/fyf/fyf/Research/Multi-View-Compression/Ego3D-Bench

ITEMS=data/vsibench/vsibench_items_16f.json
TASKS="object_abs_distance,object_size_estimation,room_size_estimation,object_rel_direction_easy,object_rel_direction_medium,object_rel_direction_hard,object_rel_distance,route_planning"
R=models/qwen2.5_vl_vsibench.py
run(){ echo "===== $(date +%F_%H:%M:%S) :: $* ====="; python "$R" --items "$ITEMS" --frames 16 --category "$TASKS" "$@" || echo "!!! FAILED: $*"; }

echo "########## Qwen MVPruner 6-group START $(date) ##########"
for KEEP in 0.25 0.10 0.05; do
  run --compress_method mvpruner --keep_ratio "$KEEP"    # mv_l1=0 mv_l2=16 (paper defaults)
done
echo "########## ALL DONE $(date) ##########"
