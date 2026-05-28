#!/usr/bin/env bash
set -euo pipefail

SRC="/home/deepcybo/.cache/huggingface/lerobot/nero_task3_step1/empty_merged_E113"
DEST_PARENT="/mnt/project_eai/chp/datasets/converted_datasets/nero_task3_step1"
DEST="${DEST_PARENT}/empty_merged_E113_raw_20260529"
TMP="${DEST}.tmp"
LOG="/home/deepcybo/transfer_empty_E113_to_wuwen2_20260529.log"

{
  echo "===== transfer empty_E113 raw to wuwen2 start $(date) ====="
  test -d "${SRC}"
  ssh wuwen2 "mkdir -p '${DEST_PARENT}' && rm -rf '${TMP}' && mkdir -p '${TMP}'"
  scp -p -r "${SRC}"/* "wuwen2:${TMP}/"
  ssh wuwen2 "rm -rf '${DEST}' && mv '${TMP}' '${DEST}' && du -sh '${DEST}' && find '${DEST}' -maxdepth 2 -type f | wc -l"
  echo "===== transfer empty_E113 raw to wuwen2 done $(date) ====="
} 2>&1 | tee -a "${LOG}"
