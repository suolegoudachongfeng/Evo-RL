#!/usr/bin/env bash
set -euo pipefail

T9_BASE="/media/deepcybo/T9/lerobot_dataset_ZGC_2ml/Evo_RL_datasets"
SRC_SUCCESS="$T9_BASE/2mL_right_E268_success_zero_right_motion_removed_20260523"
REVIEW_ROOT="$T9_BASE/2mL_right_E268_success_zero_right_motion_removed_20260523_vlm_review"
CLIPS_DIR="$REVIEW_ROOT/clips/right_wrist_image"

OLD_ANNOTATIONS="$REVIEW_ROOT/four_part_annotations"
OLD_SPLIT_ROOT="$T9_BASE/2mL_right_E268_success_zero_right_motion_removed_20260523_vlm_split_20260523"

NEW_ANNOTATIONS="$REVIEW_ROOT/four_part_annotations_strict_release_20260525"
NEW_SPLIT_ROOT="$T9_BASE/2mL_right_E268_success_zero_right_motion_removed_20260523_vlm_split_strict_release_20260525"

REPO="$HOME/Workspace/Evo-RL"
PROMPT_FILE="$REPO/tools/vlm/prompts/segment_2ml_four_parts_strict_release_prompt_zh.txt"
ANNOTATOR="$REPO/tools/vlm/annotate_2ml_four_parts.py"
SPLITTER="$REPO/tools/vlm/split_nero_dataset_by_vlm_segments.py"
CLIP_EXPORTER="$REPO/tools/vlm/export_split_review_clips_by_timestamps.py"
LOG="$T9_BASE/restrict_2ml_vlm_split_20260525.log"
MODEL="${MODEL:-gemini-3.5-flash}"
EXPECTED="${EXPECTED:-267}"

echo "Strict 2mL VLM resplit runner"
echo "Model: $MODEL"
echo "This process will wait for T9, delete old split outputs only, then regenerate annotations/split/review clips."
echo

if [ -z "${NEOLINK_API_KEY:-}" ]; then
  read -rsp "Neolink API key (input hidden): " NEOLINK_API_KEY
  export NEOLINK_API_KEY
  echo
else
  echo "Using NEOLINK_API_KEY from environment."
fi

echo "Waiting for T9 source dataset and right-wrist clips..."
while true; do
  if [ -d "$SRC_SUCCESS" ] && [ -d "$CLIPS_DIR" ]; then
    break
  fi
  date
  echo "Missing one or more paths:"
  echo "  $SRC_SUCCESS"
  echo "  $CLIPS_DIR"
  sleep 30
done

mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "===== Strict VLM resplit start: $(date) ====="
echo "Source dataset: $SRC_SUCCESS"
echo "Review clips: $CLIPS_DIR"
echo "Old annotations to delete: $OLD_ANNOTATIONS"
echo "Old split root to delete: $OLD_SPLIT_ROOT"
echo "New annotations: $NEW_ANNOTATIONS"
echo "New split root: $NEW_SPLIT_ROOT"
echo "Prompt: $PROMPT_FILE"

clip_count=$(find "$CLIPS_DIR" -maxdepth 1 -type f -name 'episode_*_right_wrist_image.mp4' | wc -l | tr -d ' ')
echo "Right-wrist source clips: $clip_count"
if [ "$clip_count" -lt "$EXPECTED" ]; then
  echo "Expected at least $EXPECTED source clips, got $clip_count" >&2
  exit 2
fi

if [ ! -f "$PROMPT_FILE" ] || [ ! -f "$ANNOTATOR" ] || [ ! -f "$SPLITTER" ] || [ ! -f "$CLIP_EXPORTER" ]; then
  echo "Missing required repo script or prompt." >&2
  exit 3
fi

echo "===== Deleting old low-quality split outputs only: $(date) ====="
rm -rf "$OLD_ANNOTATIONS" "$OLD_SPLIT_ROOT" "$NEW_ANNOTATIONS" "$NEW_SPLIT_ROOT"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate workspace
cd "$REPO"

echo "===== Running strict VLM annotation: $(date) ====="
python "$ANNOTATOR" \
  --clips-dir "$CLIPS_DIR" \
  --output-dir "$NEW_ANNOTATIONS" \
  --prompt-file "$PROMPT_FILE" \
  --model "$MODEL" \
  --timeout-s 180 \
  --sleep-s 0.2 \
  --keep-raw

echo "===== Checking annotation status: $(date) ====="
python - <<PY
import collections, json, pathlib, sys
root = pathlib.Path("$NEW_ANNOTATIONS/per_episode")
counter = collections.Counter()
for path in root.glob("episode_*_segments.json"):
    try:
        counter[json.load(open(path)).get("status", "missing")] += 1
    except Exception:
        counter["bad_json"] += 1
print(json.dumps(dict(counter), ensure_ascii=False, indent=2))
if counter.get("ok", 0) < int("$EXPECTED"):
    raise SystemExit(f"Expected {int('$EXPECTED')} ok annotations, got {counter.get('ok', 0)}")
PY

echo "===== Splitting LeRobot dataset with strict timestamps: $(date) ====="
python "$SPLITTER" \
  --dataset-root "$SRC_SUCCESS" \
  --annotations-dir "$NEW_ANNOTATIONS" \
  --output-root "$NEW_SPLIT_ROOT" \
  --video-mode copy \
  --pre-margin-s 0.0 \
  --post-margin-s 0.0 \
  --min-confidence 0.6 \
  --min-frames 5 \
  --replace

echo "===== Exporting right-wrist review clips: $(date) ====="
python "$CLIP_EXPORTER" \
  --split-root "$NEW_SPLIT_ROOT" \
  --output-root "$NEW_SPLIT_ROOT/review_clips" \
  --camera observation.images.right_wrist_image \
  --replace \
  --workers 4

echo "===== Strict VLM resplit complete: $(date) ====="
echo "New split root: $NEW_SPLIT_ROOT"
echo "New annotations: $NEW_ANNOTATIONS"
echo "New review clips: $NEW_SPLIT_ROOT/review_clips/right_wrist_image"
echo "Log: $LOG"
