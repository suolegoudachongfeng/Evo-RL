#!/usr/bin/env bash
set -euo pipefail

EVO_RL_HOME="${EVO_RL_HOME:-/mnt/project_eai/chp/workspace/Evo-RL}"
DATASET_BASE="${DATASET_BASE:-/mnt/project_eai/chp/datasets/converted_datasets/nero_task3_step2}"
CKPT_ROOT="${CKPT_ROOT:-/mnt/project_eai/chp/checkpoints}"
PY="${PY:-/mnt/project_eai/chp/envs/evorl/bin/python}"
RUN_SCRIPT="${RUN_SCRIPT:-$CKPT_ROOT/run_opc_left_right_evorl_8gpu_20260526.sh}"
HOST="$(hostname -s)"

DATASET_NAME="${DATASET_NAME:-opc_threepart_nozeroee_multitask_20260528}"
DATASET_ROOT="$DATASET_BASE/$DATASET_NAME"
DATASET_REPO="nero_task3_step2/$DATASET_NAME"
RUN_TAG="${RUN_TAG:-20260528_8gpu_threepart_nozeroee}"
RUN_ID="${RUN_ID:-opc_threepart_nozeroee_multitask_evorl_20260528_8gpu}"
PORT_BASE="${PORT_BASE:-32900}"
GPU_IDLE_MEMORY_MB="${GPU_IDLE_MEMORY_MB:-2000}"
LOCK_ROOT="$CKPT_ROOT/locks"
TRAIN_LOCK="$LOCK_ROOT/train_${RUN_ID}.lock"
WAIT_LOG="$CKPT_ROOT/queue_${RUN_ID}_wait_${HOST}.log"
E154_TRAIN_LOCK="$LOCK_ROOT/train_opc_left_right_drawer_rack_E154_nozeroee_evorl_20260528_8gpu.lock"
E154_POLICY_OUT="$CKPT_ROOT/policy_train_pi05_opc_left_right_drawer_rack_E154_nozeroee_evorl_20260528_8gpu"
WAIT_FOR_E154_FIRST="${WAIT_FOR_E154_FIRST:-true}"

exec > >(tee -a "$WAIT_LOG") 2>&1

cd "$EVO_RL_HOME"
source /mnt/project_eai/chp/env.sh || true
export PYTHONPATH="$EVO_RL_HOME/src"
export HF_HOME="/mnt/project_eai/chp/hf_cache"
export HF_HUB_CACHE="/mnt/project_eai/chp/hf_cache/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCHDYNAMO_DISABLE=1
export TOKENIZERS_PARALLELISM=false
unset TRANSFORMERS_CACHE

mkdir -p "$LOCK_ROOT"

log() {
  printf "[%s] [%s] %s\n" "$(date)" "$HOST" "$*"
}

wait_for_dataset() {
  while [ ! -f "$DATASET_ROOT/meta/info.json" ] || [ ! -f "$DATASET_ROOT/data/chunk-000/file-000.parquet" ]; do
    log "Waiting for dataset: $DATASET_ROOT"
    sleep 60
  done
  "$PY" - <<PY
import json
from pathlib import Path
p = Path("$DATASET_ROOT/meta/info.json")
info = json.loads(p.read_text())
info["repo_id"] = "$DATASET_REPO"
p.write_text(json.dumps(info, indent=4, ensure_ascii=False))
PY
  "$PY" -m lerobot.scripts.lerobot_dataset_report --dataset="$DATASET_ROOT"
}

wait_for_gpu_idle() {
  log "Waiting for all 8 GPUs to become idle under ${GPU_IDLE_MEMORY_MB}MB."
  while true; do
    mapfile -t mems < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    local count="${#mems[@]}"
    local busy=0
    if [ "$count" -lt 8 ]; then
      busy=1
    else
      for mem in "${mems[@]}"; do
        mem="${mem//[[:space:]]/}"
        if [ "${mem:-0}" -gt "$GPU_IDLE_MEMORY_MB" ]; then
          busy=1
          break
        fi
      done
    fi
    if [ "$busy" -eq 0 ]; then
      log "GPUs idle enough."
      return
    fi
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
    sleep 300
  done
}

wait_for_e154_priority() {
  if [ "$WAIT_FOR_E154_FIRST" != "true" ]; then
    return
  fi
  while [ ! -d "$E154_TRAIN_LOCK" ] && [ ! -d "$E154_POLICY_OUT" ]; do
    log "Waiting for E154 nozeroee queue to claim a server before three-part training."
    sleep 300
  done
  log "E154 priority condition satisfied."
}

start_training_if_winner() {
  wait_for_e154_priority
  wait_for_gpu_idle
  if mkdir "$TRAIN_LOCK" 2>/dev/null; then
    log "Acquired train lock: $TRAIN_LOCK"
    DATASET_NAME="$DATASET_NAME" \
    DATASET_REPO="$DATASET_REPO" \
    DATASET_ROOT="$DATASET_ROOT" \
    RUN_TAG="$RUN_TAG" \
    RUN_ID="$RUN_ID" \
    PORT_BASE="$PORT_BASE" \
    POLICY_STEPS=30000 \
    VALUE_STEPS=8000 \
    FREEZE_VISION_ENCODER=true \
    FREEZE_LANGUAGE_MODEL=true \
    ACP_N_STEP=50 \
    ACP_INDICATOR_DROPOUT_PROB=0.3 \
    WAIT_FOR_GPU_IDLE=false \
      "$RUN_SCRIPT"
  else
    log "Training lock already exists; another server claimed this run: $TRAIN_LOCK"
  fi
}

log "Queue start for $RUN_ID"
wait_for_dataset
start_training_if_winner
log "Queue script exiting."
