#!/usr/bin/env bash
# Cosine-signal backup for §13 (attn came out flat). r in {2,3,7}; baseline cosine r=1 already done.
set -u
cd "$(dirname "$0")/.."
export PATH=/home/fyf/miniconda3/envs/ego3d/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
LOG=logs/cvsp/qstage.log
S5="vsi:object_rel_direction_hard,vsi:object_rel_direction_medium,vsi:object_rel_distance,ego3d:Localization,ego3d:Ego_Centric_Absolute_Distance_MultiChoice"
for R in 2 3 7; do
  echo "=== qstage cosine r=$R start $(date) ===" >>"$LOG"
  python scripts/qstage_curve.py --ratios 0.1,0.05 --signal cosine --r $R --n 200 --configs "$S5" >>"$LOG" 2>&1
done
echo "=== QSTAGE COSINE DONE $(date) ===" >>"$LOG"
