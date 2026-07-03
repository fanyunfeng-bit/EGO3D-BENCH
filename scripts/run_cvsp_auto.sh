#!/usr/bin/env bash
# Auto pipeline (Notes/CVSP-Method.md sweep). Detached + resumable.
#   STAGE 1: discriminate 4 cvsp budget variants (★2 anchor reach on, kappa=2 ->
#            anchor pool = top 2*B_a tokens by raw L) on keep5, 3 tasks, n=200
#            -> pick the setting that beats vispruner+random.
#   GATE   : eval_sweep1.py writes GO/STOP + winning (rho_a,rho_s,tag).
#   STAGE 2: on GO, run the WINNING setting on all 6 spatial tasks at keep10/5/3,
#            n=200 (references baseline/plain/vispruner reused from the main run).
# baseline/plain/strat/vispruner are NOT re-run; collect() is a deterministic
# prefix so every cvsp_curve.py call resumes by JSONL line count. Re-run this
# script to continue after any interruption.
set -u
cd "$(dirname "$0")/.."
export PATH=/home/fyf/miniconda3/envs/ego3d/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
LOG=logs/cvsp/auto.log
PY=python

cvsp_run() {  # $1=ratios  $2=n  $3=rho_a  $4=rho_s  $5=tag  $6=configs
  $PY scripts/cvsp_curve.py --methods cvsp --ratios "$1" --n "$2" \
    --rho_a "$3" --rho_s "$4" --kappa 2 --tag="$5" --configs "$6" >>"$LOG" 2>&1
}

S1="vsi:object_rel_direction_easy,vsi:object_rel_direction_medium,ego3d:Object_Centric_Absolute_Distance_MultiChoice"
S2="vsi:object_rel_direction_easy,vsi:object_rel_direction_medium,vsi:object_rel_direction_hard,vsi:object_rel_distance,ego3d:Object_Centric_Absolute_Distance_MultiChoice,ego3d:Ego_Centric_Absolute_Distance_MultiChoice"

echo "=== STAGE1 (discriminate on keep5) start $(date) ===" >>"$LOG"
cvsp_run 0.05 200 0.4 0.3 -a40s30-k2 "$S1"
cvsp_run 0.05 200 0.3 0.2 -a30s20-k2 "$S1"
cvsp_run 0.05 200 0.3 0.3 -a30s30-k2 "$S1"
cvsp_run 0.05 200 0.2 0.5 -a20s50-k2 "$S1"
echo "=== STAGE1 done; running gate $(date) ===" >>"$LOG"

$PY scripts/eval_sweep1.py >>"$LOG" 2>&1
DEC=$($PY -c "import json;print(json.load(open('logs/cvsp/sweep1.decision.json'))['decision'])")
echo "=== gate decision: $DEC $(date) ===" >>"$LOG"

if [ "$DEC" = "GO" ]; then
  RA=$($PY -c "import json;d=json.load(open('logs/cvsp/sweep1.decision.json'));print(d['rho_a'])")
  RS=$($PY -c "import json;d=json.load(open('logs/cvsp/sweep1.decision.json'));print(d['rho_s'])")
  TAG=$($PY -c "import json;d=json.load(open('logs/cvsp/sweep1.decision.json'));print(d['tag'])")
  echo "=== STAGE2 winner rho_a=$RA rho_s=$RS tag=$TAG on 6 tasks x keep10/5/3 (n=200) $(date) ===" >>"$LOG"
  cvsp_run 0.1,0.05,0.03 200 "$RA" "$RS" "$TAG" "$S2"
  echo "=== STAGE2 done $(date) ===" >>"$LOG"
else
  echo "=== gate STOP: no budget cleared the keep5 bar; awaiting manual review ===" >>"$LOG"
fi
echo "=== AUTO PIPELINE DONE $(date) ===" >>"$LOG"
