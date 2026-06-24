#!/usr/bin/env bash
# Merge-effectiveness A/B (Notes/CVSP-Method.md §12.9 B). Detached + resumable.
# Only NEW condition: block_cvsp 8x8 + saliency-merge (--merge sal). The no-merge
# control (block_cvsp-bc8-a20) is reused from §M -- selection is identical, merge is
# the sole difference. 5 spatial tasks x keep10/5/3, n=200.
set -u
cd "$(dirname "$0")/.."
export PATH=/home/fyf/miniconda3/envs/ego3d/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
LOG=logs/cvsp/merge_ab.log
S5="vsi:object_rel_direction_hard,vsi:object_rel_direction_medium,vsi:object_rel_distance,ego3d:Localization,ego3d:Ego_Centric_Absolute_Distance_MultiChoice"

echo "=== block_cvsp 8x8 + saliency merge (rho_a=0.2, merge_dist=11) start $(date) ===" >>"$LOG"
python scripts/cvsp_curve.py --methods block_cvsp --ratios 0.1,0.05,0.03 --n 200 \
  --rho_a 0.2 --brows 8 --bcols 8 --merge sal --merge_dist 11.0 --tag=-bc8-mSal \
  --configs "$S5" >>"$LOG" 2>&1
echo "=== MERGE_AB DONE $(date) ===" >>"$LOG"
