#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/deepcybo/Workspace/Evo-RL}"
PY="${PY:-/home/deepcybo/miniconda3/envs/workspace/bin/python}"
T9_BASE="${T9_BASE:-/media/deepcybo/T9/lerobot_dataset_ZGC_2ml/opc_segmentation_workspace}"
DATASET_ROOT="${DATASET_ROOT:-$T9_BASE/datasets/nero_task3_step2/opc_left_right_20260526_v01drop5_v02_merged_success_nozeroee}"
RUN_ID="${RUN_ID:-opc_threepart_nozeroee_20260527}"
WORK_ROOT="$T9_BASE/$RUN_ID"
CLIP_ROOT="$WORK_ROOT/review_clips"
ANNOTATION_ROOT="$WORK_ROOT/annotations"
SPLIT_ROOT="$WORK_ROOT/split_datasets"
COMBINED_ROOT="$WORK_ROOT/opc_threepart_nozeroee_multitask"
LOG="$WORK_ROOT/run.log"

mkdir -p "$WORK_ROOT"
exec > >(tee -a "$LOG") 2>&1

cd "$REPO"
export PYTHONPATH="$REPO/src"

echo "===== OPC three-part VLM split start: $(date) ====="
echo "DATASET_ROOT=$DATASET_ROOT"
echo "WORK_ROOT=$WORK_ROOT"

test -f "$DATASET_ROOT/meta/info.json"
test -f "$DATASET_ROOT/data/chunk-000/file-000.parquet"

echo "===== Export head camera review clips: $(date) ====="
"$PY" tools/export_dataset_episode_review_clips_ffmpeg.py \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$CLIP_ROOT" \
  --cameras observation.images.head_image \
  --height 240 \
  --crf 24 \
  --replace

echo "===== Review clips ready: $(date) ====="
echo "Clips dir: $CLIP_ROOT/clips/head_image"
echo
if [ -z "${NEOLINK_API_KEY:-}" ]; then
  read -rsp "Input NEOLINK_API_KEY, then press Enter: " NEOLINK_API_KEY
  echo
  export NEOLINK_API_KEY
fi

echo "===== VLM annotation: $(date) ====="
"$PY" tools/vlm/annotate_opc_three_parts.py \
  --clips-dir "$CLIP_ROOT/clips/head_image" \
  --output-dir "$ANNOTATION_ROOT" \
  --prompt-file "$REPO/tools/vlm/prompts/segment_opc_three_parts_prompt_zh.txt" \
  --video-glob 'episode_*_head_image.mp4' \
  --resume \
  --keep-raw \
  --sleep-s 0.2 \
  --timeout-s 240

echo "===== Split datasets by two internal boundaries: $(date) ====="
"$PY" tools/vlm/split_opc_dataset_by_vlm_segments.py \
  --dataset-root "$DATASET_ROOT" \
  --annotations-dir "$ANNOTATION_ROOT" \
  --output-root "$SPLIT_ROOT" \
  --video-mode copy \
  --pre-margin-s 0 \
  --post-margin-s 0 \
  --min-confidence 0.5 \
  --min-frames 5 \
  --replace

echo "===== Combine three segment datasets into one multitask dataset: $(date) ====="
"$PY" tools/vlm/combine_opc_segments_multitask.py \
  --split-root "$SPLIT_ROOT" \
  --output-root "$COMBINED_ROOT" \
  --video-mode copy \
  --replace

echo "===== Dataset reports: $(date) ====="
"$PY" -m lerobot.scripts.lerobot_dataset_report --dataset "$COMBINED_ROOT"

echo "===== OPC three-part VLM split complete: $(date) ====="
echo "ANNOTATION_ROOT=$ANNOTATION_ROOT"
echo "SPLIT_ROOT=$SPLIT_ROOT"
echo "COMBINED_ROOT=$COMBINED_ROOT"
