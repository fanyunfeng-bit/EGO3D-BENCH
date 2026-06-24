#!/usr/bin/env bash
# Autonomous stage-3 (user granted authority 2026-06-20). Waits for the a20s40
# stage-2 (6 spatial tasks x keep10/5/3) to finish, evaluates, then BRANCHES:
#   GOOD -> a20s40 clearly beats plain+vispruner -> validate a20s40 on 3 MORE Ego3D
#           spatial MC tasks (Localization, Ego/Object_Centric_Relative_Distance),
#           full 4-method comparison, keep10/5/3.
#   BAD/INCOMPLETE -> run a20s50 + a30s20 on the same 6 spatial tasks for comparison,
#           then STOP for manual review (decide next step with the user).
# Detached + resumable. The wait loop only polls files (no GPU) so it coexists with
# the running a20s40 job; the branch starts its GPU work only after a20s40 exits.
set -u
cd "$(dirname "$0")/.."
export PATH=/home/fyf/miniconda3/envs/ego3d/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
LOG=logs/cvsp/stage3.log
PY=python
SP="vsi.object_rel_direction_easy vsi.object_rel_direction_medium vsi.object_rel_direction_hard vsi.object_rel_distance ego3d.Object_Centric_Absolute_Distance_MultiChoice ego3d.Ego_Centric_Absolute_Distance_MultiChoice"
S2="vsi:object_rel_direction_easy,vsi:object_rel_direction_medium,vsi:object_rel_direction_hard,vsi:object_rel_distance,ego3d:Object_Centric_Absolute_Distance_MultiChoice,ego3d:Ego_Centric_Absolute_Distance_MultiChoice"

echo "=== STAGE3 waiting for a20s40 stage2 to finish $(date) ===" >>"$LOG"
while :; do
  d=0
  for t in $SP; do
    for r in keep10 keep5 keep3; do
      f="logs/cvsp/$t.$r.cvsp-a20s40-k2.jsonl"
      [ -f "$f" ] && [ "$(wc -l <"$f" 2>/dev/null)" -ge 200 ] && d=$((d + 1))
    done
  done
  [ "$d" -ge 18 ] && break
  sleep 120
done
sleep 30   # let the a20s40 process fully exit and free the GPU
echo "=== a20s40 stage2 complete; evaluating $(date) ===" >>"$LOG"

$PY scripts/eval_stage2.py -a20s40-k2 >>"$LOG" 2>&1
DEC=$($PY -c "import json;print(json.load(open('logs/cvsp/stage2.decision.json'))['decision'])")
echo "=== a20s40 decision: $DEC $(date) ===" >>"$LOG"

if [ "$DEC" = "GOOD" ]; then
  echo "=== GOOD: validating a20s40 on 3 more Ego3D spatial tasks $(date) ===" >>"$LOG"
  $PY scripts/cvsp_curve.py --methods baseline,plain_random,vispruner,cvsp --ratios 0.1,0.05,0.03 --n 200 \
    --rho_a 0.2 --rho_s 0.4 --kappa 2 --tag=-a20s40-k2 \
    --configs ego3d:Localization,ego3d:Ego_Centric_Relative_Distance,ego3d:Object_Centric_Relative_Distance >>"$LOG" 2>&1
  echo "=== GOOD branch done $(date) ===" >>"$LOG"
else
  echo "=== $DEC: running a20s50 + a30s20 on 6 spatial tasks for comparison $(date) ===" >>"$LOG"
  $PY scripts/cvsp_curve.py --methods cvsp --ratios 0.1,0.05,0.03 --n 200 --rho_a 0.2 --rho_s 0.5 --kappa 2 --tag=-a20s50-k2 --configs "$S2" >>"$LOG" 2>&1
  $PY scripts/cvsp_curve.py --methods cvsp --ratios 0.1,0.05,0.03 --n 200 --rho_a 0.3 --rho_s 0.2 --kappa 2 --tag=-a30s20-k2 --configs "$S2" >>"$LOG" 2>&1
  echo "=== $DEC branch done; awaiting manual review $(date) ===" >>"$LOG"
fi
echo "=== STAGE3 DONE $(date) ===" >>"$LOG"
