#!/usr/bin/env bash
# Extreme-compression sweep: "when does random pruning break?"
# Curve = keep 100% (baseline, already on disk) -> 25% (on disk) -> 10% -> 5%,
# for BOTH random and vispruner, on two high-signal InternVL3-8B tasks:
#   * VSI object_rel_direction_easy (ACC, n=217, chance 0.25) -- fast, clean
#   * Ego3D Object_Centric_Absolute_Distance (RMSE, n=937)    -- continuous metric
# Sequential (single 3090, one model at a time). Resumable (skips written JSONL).
set -u
cd /home/fyf/fyf/Research/Multi-View-Compression/Ego3D-Bench
export PATH=/home/fyf/miniconda3/envs/ego3d/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
LOG=logs/extreme_sweep.driver.log
echo "================ sweep start $(date) ================" >> "$LOG"

run_vsi () {  # method ratio
  echo "---- VSI dir_easy method=$1 keep=$2 START $(date) ----" >> "$LOG"
  python models/internvl3_vsibench.py --compress_method "$1" --keep_ratio "$2" \
    --category object_rel_direction_easy >> "$LOG" 2>&1
  echo "---- VSI dir_easy method=$1 keep=$2 END   $(date) rc=$? ----" >> "$LOG"
}
run_ego () {  # method ratio
  echo "---- Ego3D ObjAbsDist method=$1 keep=$2 START $(date) ----" >> "$LOG"
  python models/internvl3_compress.py --compress_method "$1" --keep_ratio "$2" \
    --category Object_Centric_Absolute_Distance >> "$LOG" 2>&1
  echo "---- Ego3D ObjAbsDist method=$1 keep=$2 END   $(date) rc=$? ----" >> "$LOG"
}

# ---- Fast VSI sweep first (clean ACC collapse curve) ----
run_vsi vispruner 0.10
run_vsi random    0.10
run_vsi vispruner 0.05
run_vsi random    0.05
echo "######## VSI PART DONE $(date) ########" >> "$LOG"

# ---- Ego3D RMSE (continuous metric; fill missing random@25 + extreme points) ----
run_ego random    0.25
run_ego vispruner 0.10
run_ego random    0.10
run_ego vispruner 0.05
run_ego random    0.05
echo "######## EGO3D PART DONE $(date) ########" >> "$LOG"

echo "================ sweep ALL DONE $(date) ================" >> "$LOG"
