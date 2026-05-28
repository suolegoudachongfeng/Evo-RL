#!/usr/bin/env bash
set -euo pipefail

EVO_RL_HOME="/mnt/project_eai/chp/workspace/Evo-RL"
DATASET_BASE="/mnt/project_eai/chp/datasets/converted_datasets/nero_task3_step1"
CKPT_ROOT="/mnt/project_eai/chp/checkpoints"
PY="/mnt/project_eai/chp/envs/evorl/bin/python"
PORT_BASE="${PORT_BASE:-32200}"
POLICY_STEPS="${POLICY_STEPS:-30000}"
RUN_TAG="${RUN_TAG:-20260526_8gpu_nact30_reuse_frozen_value}"
DATASET_NAME="${DATASET_NAME:-2mL_right_4seg_strict_release_multitask_20260525}"
DATASET_ROOT="${DATASET_ROOT:-$DATASET_BASE/$DATASET_NAME}"
DATASET_REPO="${DATASET_REPO:-nero_task3_step1/$DATASET_NAME}"
RUN_ID="${RUN_ID:-2mL_right_4seg_strict_release_multitask_evorl_nact30_reuse_frozen_value_${RUN_TAG}}"
VALUE_DIR_REUSE="${VALUE_DIR_REUSE:-$CKPT_ROOT/value_train_2mL_right_4seg_strict_release_multitask_evorl_20260525_8gpu}"
ACP_N_STEP="${ACP_N_STEP:-50}"
POLICY_CHUNK_SIZE="${POLICY_CHUNK_SIZE:-50}"
POLICY_N_ACTION_STEPS="${POLICY_N_ACTION_STEPS:-30}"
QUEUE_LOG="$CKPT_ROOT/queue_2mL_vlm_multitask_${RUN_TAG}.log"

exec > >(tee -a "$QUEUE_LOG") 2>&1

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
unset TRANSFORMERS_CACHE

PI05_BASE=/mnt/project_eai/chp/hf_cache/hub/models--lerobot--pi05_base

echo "===== 2mL VLM multi-task EvoRL n_action_steps=30 control queue start: $(date) ====="
echo "RUN_ID=$RUN_ID"
echo "DATASET_REPO=$DATASET_REPO"
echo "DATASET_ROOT=$DATASET_ROOT"
echo "VALUE_DIR_REUSE=$VALUE_DIR_REUSE"
echo "POLICY_STEPS=$POLICY_STEPS"
echo "ACP_N_STEP=$ACP_N_STEP"
echo "POLICY_CHUNK_SIZE=$POLICY_CHUNK_SIZE"
echo "POLICY_N_ACTION_STEPS=$POLICY_N_ACTION_STEPS"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

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

ensure_reused_value() {
  if [ ! -d "$VALUE_DIR_REUSE" ]; then
    echo "Reused value directory missing: $VALUE_DIR_REUSE" >&2
    exit 5
  fi
  if ! find "$VALUE_DIR_REUSE" -maxdepth 3 -type d -name pretrained_model | grep -q .; then
    echo "No value pretrained_model checkpoint found under: $VALUE_DIR_REUSE" >&2
    exit 6
  fi
}

backup_dir() {
  local d="$1"
  if [ -e "$d" ]; then
    local bak="${d}.bak_$(date +%Y%m%d_%H%M%S)"
    echo "Existing output dir found, moving: $d -> $bak"
    mv "$d" "$bak"
  fi
}

infer_dir="$CKPT_ROOT/value_infer_${RUN_ID}"
policy_dir="$CKPT_ROOT/policy_train_pi05_${RUN_ID}"
run_log="$CKPT_ROOT/evorl_${RUN_ID}.log"

ensure_dataset
ensure_pi05_base
ensure_reused_value
backup_dir "$infer_dir"
backup_dir "$policy_dir"

{
  echo "===== Dataset report before reused-value inference: $(date) ====="
  "$PY" -m lerobot.scripts.lerobot_dataset_report --dataset="$DATASET_ROOT"

  echo "===== Reused frozen-vision value inference / ACP field injection: $(date) ====="
  "$PY" -m accelerate.commands.launch \
    --num_processes 8 \
    --num_machines 1 \
    --mixed_precision bf16 \
    --main_process_port "$PORT_BASE" \
    -m lerobot.scripts.lerobot_value_infer \
    --dataset.repo_id="$DATASET_REPO" \
    --dataset.root="$DATASET_ROOT" \
    --inference.checkpoint_path="$VALUE_DIR_REUSE" \
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

  echo "===== Dataset report after reused-value inference: $(date) ====="
  "$PY" -m lerobot.scripts.lerobot_dataset_report --dataset="$DATASET_ROOT"

  echo "===== PI05 EvoRL n_action_steps=30 smoke: $(date) ====="
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
    --policy.chunk_size="$POLICY_CHUNK_SIZE" \
    --policy.n_action_steps="$POLICY_N_ACTION_STEPS" \
    --batch_size=1 \
    --steps=1 \
    --save_checkpoint=false \
    --acp.enable=true \
    --acp.indicator_field=complementary_info.acp_indicator_v1 \
    --acp.indicator_dropout_prob=0.3 \
    --output_dir="$smoke_dir" \
    --job_name="smoke_policy_${RUN_ID}" \
    --wandb.enable=false
  rm -rf "$smoke_dir"

  echo "===== PI05 multi-task EvoRL n_action_steps=30 full train: $(date) ====="
  "$PY" -m accelerate.commands.launch \
    --num_processes 8 \
    --num_machines 1 \
    --mixed_precision bf16 \
    --main_process_port "$((PORT_BASE + 1))" \
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
    --policy.chunk_size="$POLICY_CHUNK_SIZE" \
    --policy.n_action_steps="$POLICY_N_ACTION_STEPS" \
    --batch_size=8 \
    --steps="$POLICY_STEPS" \
    --save_freq=5000 \
    --log_freq=50 \
    --acp.enable=true \
    --acp.indicator_field=complementary_info.acp_indicator_v1 \
    --acp.indicator_dropout_prob=0.3 \
    --output_dir="$policy_dir" \
    --job_name="policy_train_pi05_${RUN_ID}" \
    --wandb.enable=false

  echo "===== 2mL VLM multi-task EvoRL n_action_steps=30 queue finished: $(date) ====="
  echo "REUSED_VALUE_DIR=$VALUE_DIR_REUSE"
  echo "INFER_DIR=$infer_dir"
  echo "POLICY_DIR=$policy_dir"
} 2>&1 | tee -a "$run_log"
