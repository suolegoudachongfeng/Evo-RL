#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/deepcybo/Workspace/Evo-RL}"
PY="${PY:-/home/deepcybo/miniconda3/envs/workspace/bin/python}"
ROOT="${ROOT:-/media/deepcybo/T9/lerobot_dataset_ZGC_2ml/opc_segmentation_workspace/opc_threepart_nozeroee_20260527}"
DATASET="${DATASET:-/media/deepcybo/T9/lerobot_dataset_ZGC_2ml/opc_segmentation_workspace/datasets/nero_task3_step2/opc_left_right_20260526_v01drop5_v02_merged_success_nozeroee}"

cd "$REPO"
export PYTHONPATH="$REPO/src"

{
  echo "===== Resume split with copied videos: $(date) ====="
  rm -rf "$ROOT/split_datasets" "$ROOT/opc_threepart_nozeroee_multitask"

  "$PY" tools/vlm/split_opc_dataset_by_vlm_segments.py \
    --dataset-root "$DATASET" \
    --annotations-dir "$ROOT/annotations" \
    --output-root "$ROOT/split_datasets" \
    --video-mode copy \
    --pre-margin-s 0 \
    --post-margin-s 0 \
    --min-confidence 0.5 \
    --min-frames 5 \
    --replace

  echo "===== Resume combine with copied videos: $(date) ====="
  "$PY" tools/vlm/combine_opc_segments_multitask.py \
    --split-root "$ROOT/split_datasets" \
    --output-root "$ROOT/opc_threepart_nozeroee_multitask" \
    --video-mode copy \
    --replace

  echo "===== Resume dataset report: $(date) ====="
  "$PY" -m lerobot.scripts.lerobot_dataset_report --dataset "$ROOT/opc_threepart_nozeroee_multitask"
  echo "===== Resume complete: $(date) ====="
} 2>&1 | tee -a "$ROOT/run.log"
