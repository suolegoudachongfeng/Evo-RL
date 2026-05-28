#!/usr/bin/env bash
set -euo pipefail

SRC="${SRC:-/media/deepcybo/T9/lerobot_dataset_ZGC_2ml/opc_segmentation_workspace/opc_threepart_nozeroee_20260527/opc_threepart_nozeroee_multitask/}"
DEST="${DEST:-/mnt/project_eai/chp/datasets/converted_datasets/nero_task3_step2/opc_threepart_nozeroee_multitask_20260528}"
LOG="${LOG:-/media/deepcybo/T9/lerobot_dataset_ZGC_2ml/opc_segmentation_workspace/opc_threepart_nozeroee_20260527/transfer_to_wuwen2.log}"
REMOTE_REPO="${REMOTE_REPO:-/mnt/project_eai/chp/workspace/Evo-RL}"
REMOTE_PY="${REMOTE_PY:-/mnt/project_eai/chp/envs/evorl/bin/python}"

{
  echo "===== transfer start $(date) ====="
  test -f "$SRC/meta/info.json"
  ssh wuwen2 "rm -rf '$DEST' && mkdir -p '$DEST'"
  tar -C "$SRC" -cf - . | ssh wuwen2 "tar -C '$DEST' -xf -"
  ssh wuwen2 "cd '$REMOTE_REPO' && PYTHONPATH='$REMOTE_REPO/src' '$REMOTE_PY' -m lerobot.scripts.lerobot_dataset_report --dataset='$DEST'"
  echo "===== transfer complete $(date) ====="
  echo "DEST=$DEST"
} 2>&1 | tee -a "$LOG"
