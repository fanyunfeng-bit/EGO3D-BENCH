#!/usr/bin/env bash
# Stage-2 confirmation for ONE cvsp setting on the 6 SPATIAL tasks x keep10/5/3,
# n=200, kappa=2. Args: $1=rho_a  $2=rho_s  $3=tag. References baseline/plain/
# vispruner are reused from the main run. Detached + resumable (resume by JSONL
# line count; collect() is a deterministic prefix). Re-run to continue.
set -u
cd "$(dirname "$0")/.."
export PATH=/home/fyf/miniconda3/envs/ego3d/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
RA="${1:?rho_a}"; RS="${2:?rho_s}"; TAG="${3:?tag}"
LOG=logs/cvsp/stage2.log
S2="vsi:object_rel_direction_easy,vsi:object_rel_direction_medium,vsi:object_rel_direction_hard,vsi:object_rel_distance,ego3d:Object_Centric_Absolute_Distance_MultiChoice,ego3d:Ego_Centric_Absolute_Distance_MultiChoice"

echo "=== STAGE2 rho_a=$RA rho_s=$RS tag=$TAG start $(date) ===" >>"$LOG"
python scripts/cvsp_curve.py --methods cvsp --ratios 0.1,0.05,0.03 --n 200 \
  --rho_a "$RA" --rho_s "$RS" --kappa 2 --tag="$TAG" --configs "$S2" >>"$LOG" 2>&1
echo "=== STAGE2 done $(date) ===" >>"$LOG"
