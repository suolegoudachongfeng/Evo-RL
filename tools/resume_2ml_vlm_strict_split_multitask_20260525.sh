#!/usr/bin/env bash
set -euo pipefail

T9_BASE="/media/deepcybo/T9/lerobot_dataset_ZGC_2ml/Evo_RL_datasets"
SRC_SUCCESS="$T9_BASE/2mL_right_E268_success_zero_right_motion_removed_20260523"
REVIEW_ROOT="$T9_BASE/2mL_right_E268_success_zero_right_motion_removed_20260523_vlm_review"
CLIPS_DIR="$REVIEW_ROOT/clips/right_wrist_image"
ANNOTATIONS="$REVIEW_ROOT/four_part_annotations_strict_release_20260525"
SPLIT_ROOT="$T9_BASE/2mL_right_E268_success_zero_right_motion_removed_20260523_vlm_split_strict_release_20260525"
COMBINED_ROOT="$SPLIT_ROOT/combined_4seg_multitask"

REPO="$HOME/Workspace/Evo-RL"
PROMPT_FILE="$REPO/tools/vlm/prompts/segment_2ml_four_parts_strict_release_prompt_zh.txt"
ANNOTATOR="$REPO/tools/vlm/annotate_2ml_four_parts.py"
SPLITTER="$REPO/tools/vlm/split_nero_dataset_by_vlm_segments.py"
CLIP_EXPORTER="$REPO/tools/vlm/export_split_review_clips_by_timestamps.py"
COMBINER="$REPO/tools/vlm/combine_2ml_segments_multitask.py"
TRAIN_QUEUE="$REPO/tools/run_2ml_vlm_multitask_training_queue_20260525.sh"

EXPECTED="${EXPECTED:-267}"
MODEL="${MODEL:-gemini-3.5-flash}"
LOG="$T9_BASE/resume_2ml_vlm_strict_split_multitask_20260525.log"

WUWEN2_HOST="${WUWEN2_HOST:-wuwen2}"
REMOTE_DATASET_NAME="${REMOTE_DATASET_NAME:-2mL_right_4seg_strict_release_multitask_20260525}"
REMOTE_DATASET_ROOT="${REMOTE_DATASET_ROOT:-/mnt/project_eai/chp/datasets/converted_datasets/nero_task3_step1/$REMOTE_DATASET_NAME}"
REMOTE_QUEUE="${REMOTE_QUEUE:-/mnt/project_eai/chp/checkpoints/run_2ml_vlm_multitask_training_queue_20260525.sh}"
REMOTE_TMUX="${REMOTE_TMUX:-train_2ml_vlm_multitask_20260525}"

echo "2mL strict VLM split resume + multi-task training launcher"
echo "This resumes failed annotations only, then regenerates split/review/combined dataset and starts wuwen2 training."
echo

while true; do
  if [ -d "$SRC_SUCCESS" ] && [ -d "$CLIPS_DIR" ]; then
    break
  fi
  date
  echo "Waiting for T9 paths:"
  echo "  $SRC_SUCCESS"
  echo "  $CLIPS_DIR"
  sleep 30
done

mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate workspace
cd "$REPO"

annotation_status() {
  python - <<PY
import collections, json, pathlib
root = pathlib.Path("$ANNOTATIONS/per_episode")
counter = collections.Counter()
for path in root.glob("episode_*_segments.json"):
    try:
        counter[json.load(open(path)).get("status", "missing")] += 1
    except Exception:
        counter["bad_json"] += 1
print(json.dumps(dict(counter), ensure_ascii=False, indent=2))
PY
}

ok_count() {
  python - <<PY
import json, pathlib
root = pathlib.Path("$ANNOTATIONS/per_episode")
ok = 0
for path in root.glob("episode_*_segments.json"):
    try:
        ok += int(json.load(open(path)).get("status") == "ok")
    except Exception:
        pass
print(ok)
PY
}

delete_non_ok_annotations() {
  python - <<PY
import json, pathlib
root = pathlib.Path("$ANNOTATIONS/per_episode")
deleted = []
for path in sorted(root.glob("episode_*_segments.json")):
    try:
        status = json.load(open(path)).get("status")
    except Exception:
        status = "bad_json"
    if status != "ok":
        deleted.append(str(path))
        path.unlink()
print(json.dumps({"deleted_non_ok": deleted}, ensure_ascii=False, indent=2))
PY
}

echo "===== Resume start: $(date) ====="
echo "Source dataset: $SRC_SUCCESS"
echo "Annotations: $ANNOTATIONS"
echo "Split root: $SPLIT_ROOT"
echo "Combined root: $COMBINED_ROOT"
echo "wuwen2 dataset: $REMOTE_DATASET_ROOT"
echo "wuwen2 tmux: $REMOTE_TMUX"

echo "===== Current annotation status ====="
annotation_status || true
current_ok="$(ok_count || echo 0)"
echo "OK annotations: $current_ok / $EXPECTED"

if [ "$current_ok" -lt "$EXPECTED" ]; then
  if [ -z "${NEOLINK_API_KEY:-}" ]; then
    read -rsp "Neolink API key (input hidden): " NEOLINK_API_KEY
    export NEOLINK_API_KEY
    echo
  else
    echo "Using NEOLINK_API_KEY from environment."
  fi

  echo "===== Deleting failed/bad annotation files before resume: $(date) ====="
  delete_non_ok_annotations

  echo "===== Resuming strict VLM annotation: $(date) ====="
  python "$ANNOTATOR" \
    --clips-dir "$CLIPS_DIR" \
    --output-dir "$ANNOTATIONS" \
    --prompt-file "$PROMPT_FILE" \
    --model "$MODEL" \
    --timeout-s 180 \
    --sleep-s 0.2 \
    --resume \
    --keep-raw
fi

echo "===== Checking final annotation status: $(date) ====="
annotation_status
current_ok="$(ok_count)"
if [ "$current_ok" -lt "$EXPECTED" ]; then
  echo "Expected $EXPECTED ok annotations, got $current_ok" >&2
  exit 2
fi

echo "===== Regenerating strict four-segment datasets: $(date) ====="
python "$SPLITTER" \
  --dataset-root "$SRC_SUCCESS" \
  --annotations-dir "$ANNOTATIONS" \
  --output-root "$SPLIT_ROOT" \
  --video-mode copy \
  --pre-margin-s 0.0 \
  --post-margin-s 0.0 \
  --min-confidence 0.6 \
  --min-frames 5 \
  --replace

echo "===== Exporting right-wrist review clips: $(date) ====="
python "$CLIP_EXPORTER" \
  --split-root "$SPLIT_ROOT" \
  --output-root "$SPLIT_ROOT/review_clips" \
  --camera observation.images.right_wrist_image \
  --replace \
  --workers 4

echo "===== Combining four segments into one multi-task dataset: $(date) ====="
python "$COMBINER" \
  --split-root "$SPLIT_ROOT" \
  --output-root "$COMBINED_ROOT" \
  --video-mode copy \
  --replace

echo "===== Combined dataset summary: $(date) ====="
cat "$COMBINED_ROOT/combine_summary.json"

sync_dir_to_wuwen2() {
  local src="$1"
  local dst="$2"
  echo "Syncing $src -> $WUWEN2_HOST:$dst"
  if command -v rsync >/dev/null 2>&1 && ssh "$WUWEN2_HOST" "command -v rsync >/dev/null 2>&1"; then
    ssh "$WUWEN2_HOST" "mkdir -p '$dst'"
    rsync -a --delete "$src"/ "$WUWEN2_HOST:$dst"/
  else
    tar -C "$src" -cf - . | ssh "$WUWEN2_HOST" "rm -rf '$dst' && mkdir -p '$dst' && tar -C '$dst' -xf -"
  fi
}

echo "===== Preparing wuwen2: $(date) ====="
ssh "$WUWEN2_HOST" "mkdir -p '/mnt/project_eai/chp/checkpoints' '$(dirname "$REMOTE_DATASET_ROOT")'"
sync_dir_to_wuwen2 "$COMBINED_ROOT" "$REMOTE_DATASET_ROOT"
scp "$TRAIN_QUEUE" "$WUWEN2_HOST:$REMOTE_QUEUE"
ssh "$WUWEN2_HOST" "chmod +x '$REMOTE_QUEUE'"

echo "===== wuwen2 GPU status before launch: $(date) ====="
ssh "$WUWEN2_HOST" "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader"

echo "===== Launching wuwen2 multi-task EvoRL training: $(date) ====="
if ssh "$WUWEN2_HOST" "tmux has-session -t '$REMOTE_TMUX' 2>/dev/null"; then
  echo "tmux session already exists on $WUWEN2_HOST: $REMOTE_TMUX"
else
  ssh "$WUWEN2_HOST" "tmux new-session -d -s '$REMOTE_TMUX' 'DATASET_NAME=\"$REMOTE_DATASET_NAME\" DATASET_ROOT=\"$REMOTE_DATASET_ROOT\" bash \"$REMOTE_QUEUE\"'"
  echo "Started tmux session on $WUWEN2_HOST: $REMOTE_TMUX"
fi

echo "===== Resume + launch complete: $(date) ====="
echo "Split root: $SPLIT_ROOT"
echo "Review clips: $SPLIT_ROOT/review_clips/right_wrist_image"
echo "Combined dataset on 4090/T9: $COMBINED_ROOT"
echo "Combined dataset on wuwen2: $REMOTE_DATASET_ROOT"
echo "Training tmux on wuwen2: $REMOTE_TMUX"
echo "Training log on wuwen2: /mnt/project_eai/chp/checkpoints/queue_2mL_vlm_multitask_20260525_8gpu.log"
echo "Local log: $LOG"
