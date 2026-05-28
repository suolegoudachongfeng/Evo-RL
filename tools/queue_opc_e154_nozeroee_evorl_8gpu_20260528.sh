#!/usr/bin/env bash
set -euo pipefail

EVO_RL_HOME="${EVO_RL_HOME:-/mnt/project_eai/chp/workspace/Evo-RL}"
DATASET_BASE="${DATASET_BASE:-/mnt/project_eai/chp/datasets/converted_datasets/nero_task3_step2}"
CKPT_ROOT="${CKPT_ROOT:-/mnt/project_eai/chp/checkpoints}"
PY="${PY:-/mnt/project_eai/chp/envs/evorl/bin/python}"
RUN_SCRIPT="${RUN_SCRIPT:-$CKPT_ROOT/run_opc_left_right_evorl_8gpu_20260526.sh}"
HOST="$(hostname -s)"

MERGED_NAME="opc_left_right_20260526_20260527_E154_merged_success"
NOZERO_NAME="${MERGED_NAME}_nozeroee"
MERGED_ROOT="$DATASET_BASE/$MERGED_NAME"
NOZERO_ROOT="$DATASET_BASE/$NOZERO_NAME"
MERGED_REPO="nero_task3_step2/$MERGED_NAME"
NOZERO_REPO="nero_task3_step2/$NOZERO_NAME"
RUN_TAG="${RUN_TAG:-20260528_8gpu_E154_nozeroee}"
RUN_ID="${RUN_ID:-opc_left_right_drawer_rack_E154_nozeroee_evorl_20260528_8gpu}"
PORT_BASE="${PORT_BASE:-32800}"
GPU_IDLE_MEMORY_MB="${GPU_IDLE_MEMORY_MB:-2000}"
LOCK_ROOT="$CKPT_ROOT/locks"
PREPARE_LOCK="$LOCK_ROOT/prepare_${NOZERO_NAME}.lock"
TRAIN_LOCK="$LOCK_ROOT/train_${RUN_ID}.lock"
WAIT_LOG="$CKPT_ROOT/queue_${RUN_ID}_wait_${HOST}.log"
TASK_PROMPT="open the instrument drawer, place the rack in the correct position, and then close the instrument drawer"

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

wait_for_file() {
  local path="$1"
  while [ ! -e "$path" ]; do
    log "Waiting for $path"
    sleep 60
  done
}

prepare_dataset() {
  if [ -f "$NOZERO_ROOT/meta/info.json" ] && [ -f "$NOZERO_ROOT/meta/zero_ee_action_filter_report.json" ]; then
    log "Prepared nozero dataset already exists: $NOZERO_ROOT"
    return
  fi

  if mkdir "$PREPARE_LOCK" 2>/dev/null; then
    log "Acquired prepare lock: $PREPARE_LOCK"
    "$PY" "$EVO_RL_HOME/tools/merge_lerobot_single_chunk_datasets.py" \
      --overwrite \
      --repo-id "$MERGED_REPO" \
      --dst-root "$MERGED_ROOT" \
      --task "$TASK_PROMPT" \
      --src-root "$DATASET_BASE/opc_left_right_20260526_v01drop5_v02_merged_success" \
      --src-root "$DATASET_BASE/opc_left_right_20260527_v01_success_promptfix" \
      --src-root "$DATASET_BASE/opc_left_right_20260527_v02_success_promptfix" \
      --src-root "$DATASET_BASE/opc_left_right_20260527_v06_success_promptfix" \
      --src-root "$DATASET_BASE/opc_left_right_20260527_v07_success_promptfix" \
      --src-root "$DATASET_BASE/opc_left_right_20260527_v08_success_promptfix"

    "$PY" "$EVO_RL_HOME/tools/filter_lerobot_zero_ee_actions.py" \
      --overwrite \
      --src-root "$MERGED_ROOT" \
      --dst-root "$NOZERO_ROOT" \
      --repo-id "$NOZERO_REPO" \
      --threshold 1e-6

    "$PY" -m lerobot.scripts.lerobot_dataset_report --dataset="$NOZERO_ROOT"
    log "Prepared nozero dataset: $NOZERO_ROOT"
  else
    log "Another server/process is preparing the dataset; waiting."
    wait_for_file "$NOZERO_ROOT/meta/zero_ee_action_filter_report.json"
    log "Detected prepared nozero dataset."
  fi
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

start_training_if_winner() {
  wait_for_gpu_idle
  if mkdir "$TRAIN_LOCK" 2>/dev/null; then
    log "Acquired train lock: $TRAIN_LOCK"
    DATASET_NAME="$NOZERO_NAME" \
    DATASET_REPO="$NOZERO_REPO" \
    DATASET_ROOT="$NOZERO_ROOT" \
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
prepare_dataset
start_training_if_winner
log "Queue script exiting."
