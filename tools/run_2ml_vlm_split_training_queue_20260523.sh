#!/usr/bin/env bash
set -euo pipefail

EVO_RL_HOME="/mnt/project_eai/chp/workspace/Evo-RL"
DATASET_BASE="/mnt/project_eai/chp/datasets/converted_datasets/nero_task3_step1"
CKPT_ROOT="/mnt/project_eai/chp/checkpoints"
PY="/mnt/project_eai/chp/envs/evorl/bin/python"
PORT_BASE=31600
POLICY_STEPS=30000
VALUE_STEPS=8000
RUN_TAG="20260523_8gpu"
QUEUE_LOG="$CKPT_ROOT/queue_2mL_vlm_split_zero_motion_${RUN_TAG}.log"

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

LANG_SN=/mnt/project_eai/chp/hf_cache/hub/models--google--gemma-3-270m/snapshots/9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1
VISION_SN=$(find /mnt/project_eai/chp/hf_cache/hub/models--google--siglip-so400m-patch14-384/snapshots -mindepth 1 -maxdepth 1 -type d | head -1)
PI05_BASE=/mnt/project_eai/chp/hf_cache/hub/models--lerobot--pi05_base

echo "===== 2mL VLM split training queue start: $(date) ====="
echo "POLICY_STEPS=$POLICY_STEPS VALUE_STEPS=$VALUE_STEPS"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

ensure_dataset() {
  local root="$1"
  if [ ! -f "$root/data/chunk-000/file-000.parquet" ]; then
    echo "Dataset parquet missing: $root" >&2
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

run_evorl() {
  local run_id="$1"
  local dataset_repo="$2"
  local dataset_root="$3"
  local port="$4"
  local value_dir="$CKPT_ROOT/value_train_${run_id}"
  local infer_dir="$CKPT_ROOT/value_infer_${run_id}"
  local policy_dir="$CKPT_ROOT/policy_train_pi05_${run_id}"
  local log="$CKPT_ROOT/evorl_${run_id}.log"

  ensure_dataset "$dataset_root"
  ensure_pi05_base
  backup_dir "$value_dir"
  backup_dir "$infer_dir"
  backup_dir "$policy_dir"

  {
    echo "===== EvoRL start: $(date) ====="
    echo "RUN_ID=$run_id"
    echo "DATASET_REPO=$dataset_repo"
    echo "DATASET_ROOT=$dataset_root"

    "$PY" -m lerobot.scripts.lerobot_dataset_report --dataset="$dataset_root"

    echo "===== Value train: $(date) ====="
    "$PY" -m accelerate.commands.launch \
      --num_processes 8 \
      --num_machines 1 \
      --mixed_precision bf16 \
      --main_process_port "$port" \
      -m lerobot.scripts.lerobot_value_train \
      --dataset.repo_id="$dataset_repo" \
      --dataset.root="$dataset_root" \
      --value.type=pistar06 \
      --value.vision_repo_id="$VISION_SN" \
      --value.language_repo_id="$LANG_SN" \
      --value.dtype=bfloat16 \
      --value.freeze_vision_encoder=true \
      --value.freeze_language_model=true \
      --batch_size=8 \
      --num_workers=4 \
      --steps="$VALUE_STEPS" \
      --save_checkpoint=true \
      --save_freq=4000 \
      --wandb.enable=false \
      --output_dir="$value_dir" \
      --job_name="value_train_${run_id}"

    echo "===== Value inference: $(date) ====="
    "$PY" -m accelerate.commands.launch \
      --num_processes 8 \
      --num_machines 1 \
      --mixed_precision bf16 \
      --main_process_port "$((port + 1))" \
      -m lerobot.scripts.lerobot_value_infer \
      --dataset.repo_id="$dataset_repo" \
      --dataset.root="$dataset_root" \
      --inference.checkpoint_path="$value_dir" \
      --inference.checkpoint_ref=last \
      --runtime.device=cuda \
      --runtime.batch_size=16 \
      --runtime.num_workers=4 \
      --acp.enable=true \
      --acp.n_step=50 \
      --acp.positive_ratio=0.3 \
      --acp.value_field=complementary_info.value_v1 \
      --acp.advantage_field=complementary_info.advantage_v1 \
      --acp.indicator_field=complementary_info.acp_indicator_v1 \
      --output_dir="$infer_dir" \
      --job_name="value_infer_${run_id}"

    echo "===== Dataset report after value inference: $(date) ====="
    "$PY" -m lerobot.scripts.lerobot_dataset_report --dataset="$dataset_root"

    echo "===== PI05 EvoRL smoke: $(date) ====="
    local smoke_dir="$CKPT_ROOT/smoke_policy_${run_id}"
    rm -rf "$smoke_dir"
    CUDA_VISIBLE_DEVICES=0 "$PY" -m lerobot.scripts.lerobot_train \
      --dataset.repo_id="$dataset_repo" \
      --dataset.root="$dataset_root" \
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
      --acp.indicator_dropout_prob=0.3 \
      --output_dir="$smoke_dir" \
      --job_name="smoke_policy_${run_id}" \
      --wandb.enable=false
    rm -rf "$smoke_dir"

    echo "===== PI05 EvoRL full train: $(date) ====="
    "$PY" -m accelerate.commands.launch \
      --num_processes 8 \
      --num_machines 1 \
      --mixed_precision bf16 \
      --main_process_port "$((port + 2))" \
      -m lerobot.scripts.lerobot_train \
      --dataset.repo_id="$dataset_repo" \
      --dataset.root="$dataset_root" \
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
      --acp.indicator_dropout_prob=0.3 \
      --output_dir="$policy_dir" \
      --job_name="policy_train_pi05_${run_id}" \
      --wandb.enable=false

    echo "===== EvoRL finished: $(date) ====="
    echo "POLICY_DIR=$policy_dir"
  } 2>&1 | tee -a "$log"
}

run_bc_pi05() {
  local run_id="$1"
  local dataset_repo="$2"
  local dataset_root="$3"
  local port="$4"
  local policy_dir="$CKPT_ROOT/policy_train_pi05_${run_id}"
  local log="$CKPT_ROOT/bc_pi05_${run_id}.log"

  ensure_dataset "$dataset_root"
  ensure_pi05_base
  backup_dir "$policy_dir"

  {
    echo "===== BC-only PI05 start: $(date) ====="
    echo "RUN_ID=$run_id"
    echo "DATASET_REPO=$dataset_repo"
    echo "DATASET_ROOT=$dataset_root"
    "$PY" -m lerobot.scripts.lerobot_dataset_report --dataset="$dataset_root"

    echo "===== PI05 BC smoke: $(date) ====="
    local smoke_dir="$CKPT_ROOT/smoke_policy_${run_id}"
    rm -rf "$smoke_dir"
    CUDA_VISIBLE_DEVICES=0 "$PY" -m lerobot.scripts.lerobot_train \
      --dataset.repo_id="$dataset_repo" \
      --dataset.root="$dataset_root" \
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
      --acp.enable=false \
      --output_dir="$smoke_dir" \
      --job_name="smoke_policy_${run_id}" \
      --wandb.enable=false
    rm -rf "$smoke_dir"

    echo "===== PI05 BC full train: $(date) ====="
    "$PY" -m accelerate.commands.launch \
      --num_processes 8 \
      --num_machines 1 \
      --mixed_precision bf16 \
      --main_process_port "$port" \
      -m lerobot.scripts.lerobot_train \
      --dataset.repo_id="$dataset_repo" \
      --dataset.root="$dataset_root" \
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
      --acp.enable=false \
      --output_dir="$policy_dir" \
      --job_name="policy_train_pi05_${run_id}" \
      --wandb.enable=false

    echo "===== BC-only PI05 finished: $(date) ====="
    echo "POLICY_DIR=$policy_dir"
  } 2>&1 | tee -a "$log"
}

task_names=(
  "seg1_grasp_first_vial"
  "seg2_insert_first_vial_bottom_right"
  "seg3_grasp_remaining_vial"
  "seg4_insert_remaining_vial_top_right"
)

for idx in "${!task_names[@]}"; do
  name="${task_names[$idx]}"
  root="$DATASET_BASE/2mL_right_vlm_split_zero_success_20260523/$name"
  repo="nero_task3_step1/2mL_right_${name}_vlm_zero_success_20260523"
  run_evorl "2mL_right_${name}_vlm_zero_success_evorl_${RUN_TAG}" "$repo" "$root" "$((PORT_BASE + idx * 20))"
  run_bc_pi05 "2mL_right_${name}_vlm_zero_success_bc_${RUN_TAG}" "$repo" "$root" "$((PORT_BASE + idx * 20 + 10))"
done

run_evorl \
  "2mL_right_full_success_zero_motion_evorl_${RUN_TAG}" \
  "nero_task3_step1/2mL_right_E268_success_zero_right_motion_removed_20260523" \
  "$DATASET_BASE/2mL_right_E268_success_zero_right_motion_removed_20260523" \
  "$((PORT_BASE + 120))"

run_evorl \
  "2mL_right_full_all_zero_motion_evorl_${RUN_TAG}" \
  "nero_task3_step1/2mL_right_E299_zero_right_motion_removed_all_20260523" \
  "$DATASET_BASE/2mL_right_E299_zero_right_motion_removed_all_20260523" \
  "$((PORT_BASE + 140))"

echo "===== 2mL VLM split training queue finished: $(date) ====="
