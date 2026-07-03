#!/usr/bin/env bash
# Paper-exact pre-LLM cosine (§13): cos(proj(v_i), q_bar) in input-embed space, prune @ layer 14.
# r in {3,7} (r=2 skipped: consistently worst for attn & layer-cosine). baseline r1 reused.
set -u
cd "$(dirname "$0")/.."
export PATH=/home/fyf/miniconda3/envs/ego3d/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
LOG=logs/cvsp/qstage.log
S5="vsi:object_rel_direction_hard,vsi:object_rel_direction_medium,vsi:object_rel_distance,ego3d:Localization,ego3d:Ego_Centric_Absolute_Distance_MultiChoice"
for R in 3 7; do
  echo "=== qstage input_cos r=$R start $(date) ===" >>"$LOG"
  python scripts/qstage_curve.py --ratios 0.1,0.05 --signal input_cos --r $R --n 200 --configs "$S5" >>"$LOG" 2>&1
done
echo "=== QSTAGE INPUTCOS DONE $(date) ===" >>"$LOG"
