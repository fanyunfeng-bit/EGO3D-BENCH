#!/usr/bin/env bash
# Stage-1 cheap discrimination (Notes/CVSP-Method.md sweep, ★2 + rho).
# 4 cvsp budget variants, ALL with the ★2 L-threshold ON (delta_q=0.5), on the
# 3 currently-losing spatial tasks, keep3, n=200. baseline/plain/vispruner are
# REUSED from the completed main run (already in logs/cvsp/), not re-run here.
# Detached + resumable: cvsp_curve.py skips done JSONL lines; re-run to continue.
set -u
cd "$(dirname "$0")/.."
export PATH=/home/fyf/miniconda3/envs/ego3d/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1

CONFIGS="vsi:object_rel_direction_easy,vsi:object_rel_direction_medium,ego3d:Object_Centric_Absolute_Distance_MultiChoice"
LOG=logs/cvsp/sweep1.driver.log

run() {  # $1=rho_a  $2=rho_s  $3=tag
  echo "=== variant tag=$3 (rho_a=$1 rho_s=$2 delta_q=0.5) start $(date) ===" >>"$LOG"
  python scripts/cvsp_curve.py --methods cvsp --configs "$CONFIGS" \
    --ratios 0.03 --n 200 --rho_a "$1" --rho_s "$2" --delta_q 0.5 --tag "$3" >>"$LOG" 2>&1
  echo "=== variant tag=$3 done $(date) ===" >>"$LOG"
}

run 0.4 0.3 -a40s30-d50     # current ratio, threshold ON
run 0.3 0.2 -a30s20-d50     # coverage=50 (user A)
run 0.3 0.3 -a30s30-d50     # coverage=40 (user B)
run 0.2 0.5 -a20s50-d50     # saliency-heavy (control)
echo "=== ALL SWEEP1 DONE $(date) ===" >>"$LOG"
