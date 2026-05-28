#!/usr/bin/env bash
set -euo pipefail

DATASET_BASE="/mnt/project_eai/chp/datasets/converted_datasets/nero_task3_step2"
CKPT_ROOT="/mnt/project_eai/chp/checkpoints"
PY="/mnt/project_eai/chp/envs/evorl/bin/python"

SRC_DATASET="$DATASET_BASE/opc_left_right_20260526_20260527_E154_merged_success_nozeroee"
DST_DATASET="$DATASET_BASE/opc_left_right_20260526_20260527_E154_merged_success_nozeroee_centercrop240_r224"
DST_REPO="nero_task3_step2/opc_left_right_20260526_20260527_E154_merged_success_nozeroee_centercrop240_r224"
CROP_SCRIPT="$CKPT_ROOT/crop_lerobot_videos_center_resize.py"

RUN_TAG="20260528_8gpu_E154_nozeroee_centercrop240_r224"
RUN_ID="opc_left_right_drawer_rack_E154_nozeroee_centercrop240_r224_evorl_20260528_8gpu"
LOG="$CKPT_ROOT/queue_opc_left_right_${RUN_TAG}_prepare_and_train.log"

exec > >(tee -a "$LOG") 2>&1

echo "===== OPC E154 center-crop 240 -> 224 prepare/train start: $(date) ====="
echo "SRC_DATASET=$SRC_DATASET"
echo "DST_DATASET=$DST_DATASET"
echo "DST_REPO=$DST_REPO"
echo "RUN_ID=$RUN_ID"

if [ ! -f "$DST_DATASET/meta/image_preprocessing_centercrop240_r224.json" ]; then
  echo "===== Creating center-cropped dataset: $(date) ====="
  "$PY" "$CROP_SCRIPT" \
    --src "$SRC_DATASET" \
    --dst "$DST_DATASET" \
    --repo-id "$DST_REPO" \
    --crop-size 240 \
    --output-size 224 \
    --codec libx264 \
    --fps 30 \
    --workers 8 \
    --overwrite
else
  echo "===== Cropped dataset already exists; reusing: $(date) ====="
fi

cd /mnt/project_eai/chp/workspace/Evo-RL
source /mnt/project_eai/chp/env.sh || true
export PYTHONPATH=/mnt/project_eai/chp/workspace/Evo-RL/src
export HF_HOME="/mnt/project_eai/chp/hf_cache"
export HF_HUB_CACHE="/mnt/project_eai/chp/hf_cache/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset TRANSFORMERS_CACHE

echo "===== Cropped dataset report: $(date) ====="
"$PY" -m lerobot.scripts.lerobot_dataset_report --dataset="$DST_DATASET"

echo "===== Launching EvoRL training: $(date) ====="
RUN_TAG="$RUN_TAG" \
RUN_ID="$RUN_ID" \
DATASET_NAME="opc_left_right_20260526_20260527_E154_merged_success_nozeroee_centercrop240_r224" \
DATASET_ROOT="$DST_DATASET" \
DATASET_REPO="$DST_REPO" \
PORT_BASE=32600 \
POLICY_STEPS=30000 \
VALUE_STEPS=8000 \
FREEZE_VISION_ENCODER=true \
FREEZE_LANGUAGE_MODEL=true \
ACP_N_STEP=50 \
ACP_INDICATOR_DROPOUT_PROB=0.3 \
WAIT_FOR_GPU_IDLE=true \
bash "$CKPT_ROOT/run_opc_left_right_evorl_8gpu_20260526.sh"

echo "===== OPC E154 center-crop 240 -> 224 prepare/train finished: $(date) ====="
