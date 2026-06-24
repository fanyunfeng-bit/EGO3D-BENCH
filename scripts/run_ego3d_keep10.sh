#!/usr/bin/env bash
# Ego3D keep 10% (90% pruned): random + vispruner over 6 tasks, InternVL3-8B.
# Baseline already on disk (logs/InternVL3-8B-baseline/) -> only run the 2 methods.
# Idempotent: skips any (task,method) whose result.json already exists.
# Continues past failures (no set -e). Resumable: runner skips written JSONL lines.
set -u
cd /home/fyf/fyf/Research/Multi-View-Compression/Ego3D-Bench
export PATH=/home/fyf/miniconda3/envs/ego3d/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
LOG=logs/ego3d_keep10.driver.log
KEEP=0.10
PCT=10
echo "================ ego3d keep10 sweep start $(date) ================" >> "$LOG"

# tasks ordered most-decisive first: 2 RMSE (continuous, fidelity-demanding) ->
# strong MC (Obj/Ego abs-dist, Travel_Time) -> Localization (weakest probe).
TASKS=(
  Object_Centric_Absolute_Distance
  Ego_Centric_Absolute_Distance
  Object_Centric_Absolute_Distance_MultiChoice
  Ego_Centric_Absolute_Distance_MultiChoice
  Travel_Time
  Localization
)

run () {  # method task
  local method="$1" task="$2"
  local res="logs/InternVL3-8B-${method}-keep${PCT}/${task}.result.json"
  if [ -f "$res" ]; then
    echo "---- SKIP (done) method=$method task=$task ----" >> "$LOG"; return 0
  fi
  echo "---- START method=$method task=$task $(date) ----" >> "$LOG"
  python models/internvl3_compress.py --compress_method "$method" --keep_ratio "$KEEP" \
    --category "$task" >> "$LOG" 2>&1
  echo "---- END   method=$method task=$task $(date) rc=$? ----" >> "$LOG"
}

for t in "${TASKS[@]}"; do
  run vispruner "$t"   # faster method first -> a number per task sooner
  run random    "$t"
  echo "######## TASK DONE $t $(date) ########" >> "$LOG"
done
echo "================ ego3d keep10 sweep ALL DONE $(date) ================" >> "$LOG"
