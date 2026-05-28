#!/usr/bin/env bash
set -euo pipefail

EVO_RL_HOME="/mnt/project_eai/chp/workspace/Evo-RL"
DATASET_BASE="/mnt/project_eai/chp/datasets/converted_datasets/nero_task3_step2"
CKPT_ROOT="/mnt/project_eai/chp/checkpoints"
PY="/mnt/project_eai/chp/envs/evorl/bin/python"
PORT_BASE="${PORT_BASE:-32300}"
POLICY_STEPS="${POLICY_STEPS:-30000}"
VALUE_STEPS="${VALUE_STEPS:-8000}"
RUN_TAG="${RUN_TAG:-20260526_8gpu}"
DATASET_NAME="${DATASET_NAME:-opc_left_right_20260526_v01drop5_v02_merged_success}"
DATASET_ROOT="${DATASET_ROOT:-$DATASET_BASE/$DATASET_NAME}"
DATASET_REPO="${DATASET_REPO:-nero_task3_step2/$DATASET_NAME}"
RUN_ID="${RUN_ID:-opc_left_right_drawer_rack_evorl_${RUN_TAG}}"
FREEZE_VISION_ENCODER="${FREEZE_VISION_ENCODER:-true}"
FREEZE_LANGUAGE_MODEL="${FREEZE_LANGUAGE_MODEL:-true}"
ACP_N_STEP="${ACP_N_STEP:-50}"
ACP_INDICATOR_DROPOUT_PROB="${ACP_INDICATOR_DROPOUT_PROB:-0.3}"
WAIT_FOR_GPU_IDLE="${WAIT_FOR_GPU_IDLE:-true}"
GPU_IDLE_MEMORY_MB="${GPU_IDLE_MEMORY_MB:-2000}"
QUEUE_LOG="$CKPT_ROOT/queue_opc_left_right_${RUN_TAG}.log"

exec > >(tee -a "$QUEUE_LOG") 2>&1

wait_for_gpu_idle() {
  if [ "$WAIT_FOR_GPU_IDLE" != "true" ]; then
    return
  fi
  echo "===== Waiting for wuwen2 GPUs to become idle: $(date) ====="
  while true; do
    mapfile -t mems < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    local busy=0
    for mem in "${mems[@]}"; do
      mem="${mem//[[:space:]]/}"
      if [ "${mem:-0}" -gt "$GPU_IDLE_MEMORY_MB" ]; then
        busy=1
        break
      fi
    done
    if [ "$busy" -eq 0 ]; then
      echo "GPUs are idle enough to start: $(date)"
      break
    fi
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
    sleep 300
  done
}

cd "$EVO_RL_HOME"
source /mnt/project_eai/chp/env.sh || true
export PYTHONPATH="$EVO_RL_HOME/src"
export HF_HOME="/mnt/project_eai/chp/hf_cache"
export HF_HUB_CACHE="/mnt/project_eai/chp/hf_cache/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCHDYNAMO_DISABLE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LEROBOT_VALUE_TRAIN_FIND_UNUSED_PARAMETERS=false
unset TRANSFORMERS_CACHE

LANG_SN=/mnt/project_eai/chp/hf_cache/hub/models--google--gemma-3-270m/snapshots/9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1
VISION_SN=$(find /mnt/project_eai/chp/hf_cache/hub/models--google--siglip-so400m-patch14-384/snapshots -mindepth 1 -maxdepth 1 -type d | head -1)
PI05_BASE=/mnt/project_eai/chp/hf_cache/hub/models--lerobot--pi05_base

echo "===== OPC left/right drawer-rack EvoRL queue start: $(date) ====="
echo "RUN_ID=$RUN_ID"
echo "DATASET_REPO=$DATASET_REPO"
echo "DATASET_ROOT=$DATASET_ROOT"
echo "POLICY_STEPS=$POLICY_STEPS VALUE_STEPS=$VALUE_STEPS"
echo "FREEZE_VISION_ENCODER=$FREEZE_VISION_ENCODER FREEZE_LANGUAGE_MODEL=$FREEZE_LANGUAGE_MODEL"
echo "ACP_N_STEP=$ACP_N_STEP ACP_INDICATOR_DROPOUT_PROB=$ACP_INDICATOR_DROPOUT_PROB"

ensure_dataset() {
  if [ ! -f "$DATASET_ROOT/data/chunk-000/file-000.parquet" ]; then
    echo "Dataset parquet missing: $DATASET_ROOT" >&2
    exit 2
  fi
  if [ ! -f "$DATASET_ROOT/meta/tasks.parquet" ]; then
    echo "Dataset tasks parquet missing: $DATASET_ROOT/meta/tasks.parquet" >&2
    exit 2
  fi
}

ensure_pi05_base() {
  if [ ! -d "$PI05_BASE" ]; then
    echo "PI05 base missing: $PI05_BASE" >&2
    exit 3
  fi
  for name in config.json model.safetensors policy_preprocessor.json policy_postprocessor.json; do
    if [ ! -f "$PI05_BASE/$name" ]; then
      echo "PI05 base file missing: $PI05_BASE/$name" >&2
      exit 4
    fi
  done
}

backup_dir() {
  local d="$1"
  if [ -e "$d" ]; then
    local bak="${d}.bak_$(date +%Y%m%d_%H%M%S)"
    echo "Existing output dir found, moving: $d -> $bak"
    mv "$d" "$bak"
  fi
}

value_dir="$CKPT_ROOT/value_train_${RUN_ID}"
infer_dir="$CKPT_ROOT/value_infer_${RUN_ID}"
policy_dir="$CKPT_ROOT/policy_train_pi05_${RUN_ID}"
run_log="$CKPT_ROOT/evorl_${RUN_ID}.log"

ensure_dataset
ensure_pi05_base
backup_dir "$value_dir"
backup_dir "$infer_dir"
backup_dir "$policy_dir"
wait_for_gpu_idle

{
  echo "===== Dataset report before value train: $(date) ====="
  "$PY" -m lerobot.scripts.lerobot_dataset_report --dataset="$DATASET_ROOT"

  echo "===== Value train: $(date) ====="
  "$PY" -m accelerate.commands.launch \
    --num_processes 8 \
    --num_machines 1 \
    --mixed_precision bf16 \
    --main_process_port "$PORT_BASE" \
    -m lerobot.scripts.lerobot_value_train \
    --dataset.repo_id="$DATASET_REPO" \
    --dataset.root="$DATASET_ROOT" \
    --value.type=pistar06 \
    --value.vision_repo_id="$VISION_SN" \
    --value.language_repo_id="$LANG_SN" \
    --value.dtype=bfloat16 \
    --value.freeze_vision_encoder="$FREEZE_VISION_ENCODER" \
    --value.freeze_language_model="$FREEZE_LANGUAGE_MODEL" \
    --batch_size=8 \
    --num_workers=4 \
    --steps="$VALUE_STEPS" \
    --save_checkpoint=true \
    --save_freq=4000 \
    --wandb.enable=false \
    --output_dir="$value_dir" \
    --job_name="value_train_${RUN_ID}"

  echo "===== Value inference / ACP field injection: $(date) ====="
  "$PY" -m accelerate.commands.launch \
    --num_processes 8 \
    --num_machines 1 \
    --mixed_precision bf16 \
    --main_process_port "$((PORT_BASE + 1))" \
    -m lerobot.scripts.lerobot_value_infer \
    --dataset.repo_id="$DATASET_REPO" \
    --dataset.root="$DATASET_ROOT" \
    --inference.checkpoint_path="$value_dir" \
    --inference.checkpoint_ref=last \
    --runtime.device=cuda \
    --runtime.batch_size=16 \
    --runtime.num_workers=4 \
    --acp.enable=true \
    --acp.n_step="$ACP_N_STEP" \
    --acp.positive_ratio=0.3 \
    --acp.value_field=complementary_info.value_v1 \
    --acp.advantage_field=complementary_info.advantage_v1 \
    --acp.indicator_field=complementary_info.acp_indicator_v1 \
    --output_dir="$infer_dir" \
    --job_name="value_infer_${RUN_ID}"

  echo "===== Dataset report after value inference: $(date) ====="
  "$PY" -m lerobot.scripts.lerobot_dataset_report --dataset="$DATASET_ROOT"

  echo "===== PI05 EvoRL smoke: $(date) ====="
  smoke_dir="$CKPT_ROOT/smoke_policy_${RUN_ID}"
  rm -rf "$smoke_dir"
  CUDA_VISIBLE_DEVICES=0 "$PY" -m lerobot.scripts.lerobot_train \
    --dataset.repo_id="$DATASET_REPO" \
    --dataset.root="$DATASET_ROOT" \
    --policy.type=pi05 \
    --policy.pretrained_path="$PI05_BASE" \
    --policy.device=cuda \
    --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=true \
    --policy.push_to_hub=false \
    --policy.private=false \
    --batch_size=1 \
    --steps=1 \
    --save_checkpoint=false \
    --acp.enable=true \
    --acp.indicator_field=complementary_info.acp_indicator_v1 \
    --acp.indicator_dropout_prob="$ACP_INDICATOR_DROPOUT_PROB" \
    --output_dir="$smoke_dir" \
    --job_name="smoke_policy_${RUN_ID}" \
    --wandb.enable=false
  rm -rf "$smoke_dir"

  echo "===== PI05 OPC left/right drawer-rack EvoRL full train: $(date) ====="
  "$PY" -m accelerate.commands.launch \
    --num_processes 8 \
    --num_machines 1 \
    --mixed_precision bf16 \
    --main_process_port "$((PORT_BASE + 2))" \
    -m lerobot.scripts.lerobot_train \
    --dataset.repo_id="$DATASET_REPO" \
    --dataset.root="$DATASET_ROOT" \
    --policy.type=pi05 \
    --policy.pretrained_path="$PI05_BASE" \
    --policy.device=cuda \
    --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=true \
    --policy.push_to_hub=false \
    --policy.private=false \
    --batch_size=8 \
    --steps="$POLICY_STEPS" \
    --save_freq=5000 \
    --log_freq=50 \
    --acp.enable=true \
    --acp.indicator_field=complementary_info.acp_indicator_v1 \
    --acp.indicator_dropout_prob="$ACP_INDICATOR_DROPOUT_PROB" \
    --output_dir="$policy_dir" \
    --job_name="policy_train_pi05_${RUN_ID}" \
    --wandb.enable=false

  echo "===== OPC left/right drawer-rack EvoRL queue finished: $(date) ====="
  echo "VALUE_DIR=$value_dir"
  echo "INFER_DIR=$infer_dir"
  echo "POLICY_DIR=$policy_dir"
} 2>&1 | tee -a "$run_log"
