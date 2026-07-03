#!/usr/bin/env bash
# a20s40 (cvsp, rho_a=0.2/rho_s=0.4/kappa=2) FULL data on the 7 Ego3D spatial tasks, keep10/5.
# Resumes existing n=200 (5 MC tasks) -> full; 2 number tasks (AbsDist) run fresh w/ RMSE.
set -u
cd "$(dirname "$0")/.."
export PATH=/home/fyf/miniconda3/envs/ego3d/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
LOG=logs/cvsp/a20s40_full.log
T7="ego3d:Ego_Centric_Absolute_Distance,ego3d:Ego_Centric_Absolute_Distance_MultiChoice,ego3d:Ego_Centric_Relative_Distance,ego3d:Localization,ego3d:Object_Centric_Absolute_Distance,ego3d:Object_Centric_Absolute_Distance_MultiChoice,ego3d:Object_Centric_Relative_Distance"
echo "=== a20s40 FULL start $(date) ===" >>"$LOG"
python scripts/cvsp_curve.py --methods cvsp --rho_a 0.2 --rho_s 0.4 --kappa 2 --tau 0.85 --tag=-a20s40-k2 \
  --ratios 0.1 --n 99999 --configs "$T7" >>"$LOG" 2>&1
echo "=== A20S40 FULL DONE $(date) ===" >>"$LOG"
