#!/usr/bin/env bash
# Ego3D-Bench spatial tasks, FULL data, keep10/5. Methods: baseline(full) / plain_random /
# vispruner (via cvsp_curve) + two-stage input_cos r7 (via qstage_curve). Detached + resumable
# (existing n=200 files are deterministic prefixes -> auto-extended to full, not re-run).
# 5 MC tasks scored by ACC, 2 number tasks (Absolute_Distance) by RMSE.
set -u
cd "$(dirname "$0")/.."
export PATH=/home/fyf/miniconda3/envs/ego3d/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
LOG=logs/cvsp/ego3d_full.log
T7="ego3d:Ego_Centric_Absolute_Distance,ego3d:Ego_Centric_Absolute_Distance_MultiChoice,ego3d:Ego_Centric_Relative_Distance,ego3d:Localization,ego3d:Object_Centric_Absolute_Distance,ego3d:Object_Centric_Absolute_Distance_MultiChoice,ego3d:Object_Centric_Relative_Distance"

echo "=== cvsp baseline/plain/vispruner FULL start $(date) ===" >>"$LOG"
python scripts/cvsp_curve.py --methods baseline,plain_random,vispruner --ratios 0.1,0.05 --n 99999 --configs "$T7" >>"$LOG" 2>&1
echo "=== qstage input_cos r7 FULL start $(date) ===" >>"$LOG"
python scripts/qstage_curve.py --signal input_cos --r 7 --ratios 0.1,0.05 --n 99999 --configs "$T7" >>"$LOG" 2>&1
echo "=== EGO3D FULL DONE $(date) ===" >>"$LOG"
