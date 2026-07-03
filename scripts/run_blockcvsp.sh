#!/usr/bin/env bash
# Block-CVSP experiment (Notes/CVSP-Method.md §12.6). Detached + resumable.
# Core first: block_cvsp(anchor) -> block_cvsp(no-anchor ablation) -> nuwa-lite -> strat(Localization).
# baseline/plain/vispruner/cvsp(a20s40)/strat(4 tasks) reused from prior runs.
set -u
cd "$(dirname "$0")/.."
export PATH=/home/fyf/miniconda3/envs/ego3d/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
LOG=logs/cvsp/blockcvsp.log
PY=python
S5="vsi:object_rel_direction_hard,vsi:object_rel_direction_medium,vsi:object_rel_distance,ego3d:Localization,ego3d:Ego_Centric_Absolute_Distance_MultiChoice"

echo "=== block_cvsp anchor(rho_a=0.2, 8x8 fine blocks) start $(date) ===" >>"$LOG"
$PY scripts/cvsp_curve.py --methods block_cvsp --ratios 0.1,0.05,0.03 --n 200 --rho_a 0.2 --brows 8 --bcols 8 --tag=-bc8-a20 --configs "$S5" >>"$LOG" 2>&1
echo "=== block_cvsp no-anchor(rho_a=0, 8x8 fine blocks) start $(date) ===" >>"$LOG"
$PY scripts/cvsp_curve.py --methods block_cvsp --ratios 0.1,0.05,0.03 --n 200 --rho_a 0 --brows 8 --bcols 8 --tag=-bc8-noanc --configs "$S5" >>"$LOG" 2>&1
echo "=== nuwa-lite start $(date) ===" >>"$LOG"
$PY scripts/cvsp_curve.py --methods nuwa --ratios 0.1,0.05,0.03 --n 200 --configs "$S5" >>"$LOG" 2>&1
echo "=== strat_random for Localization start $(date) ===" >>"$LOG"
$PY scripts/cvsp_curve.py --methods strat_random --ratios 0.1,0.05,0.03 --n 200 --configs ego3d:Localization >>"$LOG" 2>&1
echo "=== BLOCKCVSP ALL DONE $(date) ===" >>"$LOG"
