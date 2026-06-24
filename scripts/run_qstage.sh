#!/usr/bin/env bash
# Two-stage Query-Aware Block-CVSP (Notes/CVSP-Method.md §13). Detached + resumable.
# Baseline = qstage cosine r=1 (no-prune == single-stage block-cvsp 4x4 @ T, verified transparent).
# Two-stage = qstage attn r in {2,3,7} (attention signal = Nuwa code version, primary).
# keep10/5, 5 spatial tasks, n=200. Compare two-stage(attn) vs baseline(r1) at same layer-avg T.
set -u
cd "$(dirname "$0")/.."
export PATH=/home/fyf/miniconda3/envs/ego3d/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
LOG=logs/cvsp/qstage.log
S5="vsi:object_rel_direction_hard,vsi:object_rel_direction_medium,vsi:object_rel_distance,ego3d:Localization,ego3d:Ego_Centric_Absolute_Distance_MultiChoice"
PY=python

echo "=== qstage baseline cosine r=1 (single-stage 4x4) start $(date) ===" >>"$LOG"
$PY scripts/qstage_curve.py --ratios 0.1,0.05 --signal cosine --r 1 --n 200 --configs "$S5" >>"$LOG" 2>&1
for R in 2 3 7; do
  echo "=== qstage two-stage attn r=$R start $(date) ===" >>"$LOG"
  $PY scripts/qstage_curve.py --ratios 0.1,0.05 --signal attn --r $R --n 200 --configs "$S5" >>"$LOG" 2>&1
done
echo "=== QSTAGE ALL DONE $(date) ===" >>"$LOG"
