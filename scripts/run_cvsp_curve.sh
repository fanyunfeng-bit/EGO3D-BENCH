#!/usr/bin/env bash
# Detached, self-restarting (resumable) driver for the CVSP §8 main curve.
# 8 configs (2 Ego3D MC + 6 VSI MC) x {baseline + 4 ratios x 4 methods} x n=200.
# Resumable: cvsp_curve.py counts existing jsonl lines and skips them, so a crash
# + relaunch continues where it stopped. This driver re-runs python on any nonzero
# exit until it completes (exit 0).
#
# Launch detached (survives session exit):
#   setsid nohup bash scripts/run_cvsp_curve.sh > logs/cvsp/run.log 2>&1 & disown
# Watch:   tail -f logs/cvsp/run.log
# View:    PATH=/home/fyf/miniconda3/envs/ego3d/bin:$PATH python scripts/eval_cvsp.py
set -u
cd "$(dirname "$0")/.."

export PATH=/home/fyf/miniconda3/envs/ego3d/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1

CONFIGS="${CONFIGS:-ego3d:Object_Centric_Absolute_Distance_MultiChoice,ego3d:Ego_Centric_Absolute_Distance_MultiChoice,vsi:object_rel_direction_easy,vsi:object_rel_direction_medium,vsi:object_rel_direction_hard,vsi:object_rel_distance,vsi:route_planning,vsi:obj_appearance_order}"
RATIOS="${RATIOS:-0.25,0.1,0.05,0.03}"
N="${N:-200}"

mkdir -p logs/cvsp
for attempt in $(seq 1 30); do
  echo "=== attempt $attempt START $(date) ==="
  if python scripts/cvsp_curve.py --n "$N" --configs "$CONFIGS" --ratios "$RATIOS"; then
    echo "=== ALL DONE $(date) ==="
    exit 0
  fi
  echo "=== python exited nonzero; resume in 60s ($(date)) ==="
  sleep 60
done
echo "=== gave up after 30 attempts $(date) ==="
exit 1
