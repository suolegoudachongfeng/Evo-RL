#!/usr/bin/env bash
set -euo pipefail

EVO_RL_HOME="${EVO_RL_HOME:-/mnt/project_eai/chp/workspace/Evo-RL}"
DATASET_BASE="${DATASET_BASE:-/mnt/project_eai/chp/datasets/converted_datasets/nero_task3_step1}"
CKPT_ROOT="${CKPT_ROOT:-/mnt/project_eai/chp/checkpoints}"
PY="${PY:-/mnt/project_eai/chp/envs/evorl/bin/python}"
RUN_SCRIPT="${RUN_SCRIPT:-$CKPT_ROOT/run_opc_left_right_evorl_8gpu_20260526.sh}"
HOST="$(hostname -s)"

DATASET_NAME="${DATASET_NAME:-empty_merged_E113_drop58_success_baseline_20260529}"
DATASET_ROOT="$DATASET_BASE/$DATASET_NAME"
DATASET_REPO="nero_task3_step1/$DATASET_NAME"
RUN_TAG="${RUN_TAG:-20260529_8gpu_empty_E113_drop58_success_baseline}"
RUN_ID="${RUN_ID:-empty_E113_drop58_success_baseline_evorl_20260529_8gpu}"
PORT_BASE="${PORT_BASE:-33100}"
GPU_IDLE_MEMORY_MB="${GPU_IDLE_MEMORY_MB:-2000}"
LOCK_ROOT="$CKPT_ROOT/locks"
TRAIN_LOCK="$LOCK_ROOT/train_${RUN_ID}.lock"
WAIT_LOG="$CKPT_ROOT/queue_${RUN_ID}_wait_${HOST}.log"
TASK_PROMPT="Place small vials right-bottom then right-top with right gripper; place large vials left-bottom then left-top with left gripper."

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
    log "Waiting for prepared dataset: $DATASET_ROOT"
    sleep 60
  done
  "$PY" - <<PY
import json
from pathlib import Path
import pandas as pd

root = Path("$DATASET_ROOT")
repo_id = "$DATASET_REPO"
prompt = "$TASK_PROMPT"
info_path = root / "meta" / "info.json"
info = json.loads(info_path.read_text())
info["repo_id"] = repo_id
info["total_tasks"] = 1
info["splits"] = {"train": f"0:{info['total_episodes']}"}
info_path.write_text(json.dumps(info, indent=4, ensure_ascii=False) + "\\n")
pd.DataFrame({"task_index": [0]}, index=[prompt]).to_parquet(root / "meta" / "tasks.parquet")
for ep_path in sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
    eps = pd.read_parquet(ep_path)
    eps["tasks"] = [[prompt] for _ in range(len(eps))]
    eps["episode_success"] = "success"
    eps.to_parquet(ep_path, index=False)
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

start_training_if_winner() {
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
