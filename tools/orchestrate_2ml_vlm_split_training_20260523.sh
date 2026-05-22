#!/usr/bin/env bash
set -euo pipefail

T9_BASE="/media/deepcybo/T9/lerobot_dataset_ZGC_2ml/Evo_RL_datasets"
SRC_SUCCESS="$T9_BASE/2mL_right_E268_success_zero_right_motion_removed_20260523"
SRC_ALL="$T9_BASE/2mL_right_E299_zero_right_motion_removed_all_20260523"
ANNOTATIONS="$T9_BASE/2mL_right_E268_success_zero_right_motion_removed_20260523_vlm_review/four_part_annotations"
SPLIT_ROOT="$T9_BASE/2mL_right_E268_success_zero_right_motion_removed_20260523_vlm_split_20260523"
REMOTE_DATASET_BASE="/mnt/project_eai/chp/datasets/converted_datasets/nero_task3_step1"
REMOTE_SPLIT_ROOT="$REMOTE_DATASET_BASE/2mL_right_vlm_split_zero_success_20260523"
REMOTE_QUEUE="/mnt/project_eai/chp/checkpoints/run_2ml_vlm_split_training_queue_20260523.sh"
LOCAL_QUEUE="$HOME/Workspace/Evo-RL/tools/run_2ml_vlm_split_training_queue_20260523.sh"
LOCAL_SPLITTER="$HOME/Workspace/Evo-RL/tools/vlm/split_nero_dataset_by_vlm_segments.py"
LOG="$T9_BASE/orchestrate_2ml_vlm_split_training_20260523.log"
EXPECTED=267

exec > >(tee -a "$LOG") 2>&1

echo "===== Orchestrator start: $(date) ====="
echo "Waiting for VLM annotations in $ANNOTATIONS"

while true; do
  status_json=$(python3 - <<PY
import json, pathlib, collections, sys
out=pathlib.Path("$ANNOTATIONS/per_episode")
c=collections.Counter()
for p in out.glob("episode_*_segments.json"):
    try:
        c[json.load(open(p)).get("status", "missing")] += 1
    except Exception:
        c["bad_json"] += 1
print(json.dumps(dict(c)))
PY
)
  echo "[$(date)] annotation status: $status_json"
  ok_count=$(python3 - <<PY
import json
c=json.loads('$status_json')
print(c.get('ok', 0))
PY
)
  total_count=$(python3 - <<PY
import json
c=json.loads('$status_json')
print(sum(c.values()))
PY
)
  if [ "$ok_count" -ge "$EXPECTED" ]; then
    break
  fi
  if [ "$total_count" -ge "$EXPECTED" ] && [ "$ok_count" -lt "$EXPECTED" ]; then
    echo "VLM annotation completed with non-ok entries: $status_json" >&2
    exit 10
  fi
  sleep 300
done

echo "===== VLM annotations complete: $(date) ====="
echo "===== Splitting datasets: $(date) ====="
source ~/miniconda3/etc/profile.d/conda.sh
conda activate workspace
cd "$HOME/Workspace/Evo-RL"
python "$LOCAL_SPLITTER" \
  --dataset-root "$SRC_SUCCESS" \
  --annotations-dir "$ANNOTATIONS" \
  --output-root "$SPLIT_ROOT" \
  --video-mode copy \
  --replace

echo "===== Copying datasets to wuwen: $(date) ====="
ssh wuwen "mkdir -p '$REMOTE_DATASET_BASE' '/mnt/project_eai/chp/checkpoints'"
ssh wuwen "rm -rf '$REMOTE_SPLIT_ROOT' '$REMOTE_DATASET_BASE/$(basename "$SRC_SUCCESS")' '$REMOTE_DATASET_BASE/$(basename "$SRC_ALL")'"
scp -r "$SPLIT_ROOT" wuwen:"$REMOTE_SPLIT_ROOT"
scp -r "$SRC_SUCCESS" wuwen:"$REMOTE_DATASET_BASE/"
scp -r "$SRC_ALL" wuwen:"$REMOTE_DATASET_BASE/"
scp "$LOCAL_QUEUE" wuwen:"$REMOTE_QUEUE"
ssh wuwen "chmod +x '$REMOTE_QUEUE'"

echo "===== Starting wuwen 8-GPU training queue: $(date) ====="
ssh wuwen "tmux kill-session -t train_2ml_vlm_split_20260523 2>/dev/null || true; tmux new-session -d -s train_2ml_vlm_split_20260523 'bash $REMOTE_QUEUE'"
ssh wuwen "tmux ls | grep train_2ml_vlm_split_20260523"

echo "===== Orchestrator finished setup: $(date) ====="
echo "Monitor wuwen tmux session: train_2ml_vlm_split_20260523"
