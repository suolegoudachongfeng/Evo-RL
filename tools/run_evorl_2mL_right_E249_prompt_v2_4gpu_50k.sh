#!/usr/bin/env bash
set -euo pipefail

RUN_ID="2mL_right_retrain_20260520_E249_prompt_v2_4gpu_50k"
EVO_RL_HOME="/mnt/project_eai/chp/workspace/Evo-RL"
DATASET_REPO="nero_task3_step1/2mL_right_retrain_20260520_E249_prompt_v2_evorl"
DATASET_ROOT="/mnt/project_eai/chp/datasets/converted_datasets/nero_task3_step1/2mL_right_retrain_20260520_E249_prompt_v2_evorl"
CKPT_ROOT="/mnt/project_eai/chp/checkpoints"
VALUE_DIR="$CKPT_ROOT/value_train_${RUN_ID}"
VALUE_INFER_DIR="$CKPT_ROOT/value_infer_${RUN_ID}"
POLICY_DIR="$CKPT_ROOT/policy_train_pi05_${RUN_ID}"
LOG="$CKPT_ROOT/evorl_${RUN_ID}.log"
PY="/mnt/project_eai/chp/envs/evorl/bin/python"
PORT=29861

exec > >(tee -a "$LOG") 2>&1

echo "===== EvoRL pipeline start: $(date) ====="
echo "RUN_ID=$RUN_ID"
echo "DATASET_REPO=$DATASET_REPO"
echo "DATASET_ROOT=$DATASET_ROOT"
echo "VALUE_DIR=$VALUE_DIR"
echo "VALUE_INFER_DIR=$VALUE_INFER_DIR"
echo "POLICY_DIR=$POLICY_DIR"

cd "$EVO_RL_HOME"
source /mnt/project_eai/chp/env.sh || true
export PYTHONPATH="$EVO_RL_HOME/src"
export HF_HOME="/mnt/project_eai/chp/hf_cache"
export HF_HUB_CACHE="/mnt/project_eai/chp/hf_cache/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCHDYNAMO_DISABLE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3
unset TRANSFORMERS_CACHE

LANG_SN=/mnt/project_eai/chp/hf_cache/hub/models--google--gemma-3-270m/snapshots/9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1
VISION_SN=$(find /mnt/project_eai/chp/hf_cache/hub/models--google--siglip-so400m-patch14-384/snapshots -mindepth 1 -maxdepth 1 -type d | head -1)
PI05_BASE=/mnt/project_eai/chp/hf_cache/hub/models--lerobot--pi05_base

nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

if [ ! -f "$DATASET_ROOT/data/chunk-000/file-000.parquet" ]; then
  echo "Dataset parquet missing: $DATASET_ROOT" >&2
  exit 2
fi
if [ ! -d "$PI05_BASE" ]; then
  echo "PI05 base missing: $PI05_BASE" >&2
  exit 3
fi
if [ ! -d "$LANG_SN" ] || [ ! -d "$VISION_SN" ]; then
  echo "Value model cache missing. LANG_SN=$LANG_SN VISION_SN=$VISION_SN" >&2
  exit 4
fi

for d in "$VALUE_DIR" "$VALUE_INFER_DIR" "$POLICY_DIR"; do
  if [ -e "$d" ]; then
    bak="${d}.bak_$(date +%Y%m%d_%H%M%S)"
    echo "Existing output dir found, moving: $d -> $bak"
    mv "$d" "$bak"
  fi
done

echo "===== Dataset report before training ====="
"$PY" -m lerobot.scripts.lerobot_dataset_report --dataset="$DATASET_ROOT"

echo "===== Value train: $(date) ====="
"$PY" -m accelerate.commands.launch \
  --num_processes 4 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --main_process_port "$PORT" \
  -m lerobot.scripts.lerobot_value_train \
  --dataset.repo_id="$DATASET_REPO" \
  --dataset.root="$DATASET_ROOT" \
  --value.type=pistar06 \
  --value.vision_repo_id="$VISION_SN" \
  --value.language_repo_id="$LANG_SN" \
  --value.dtype=bfloat16 \
  --value.freeze_vision_encoder=true \
  --value.freeze_language_model=true \
  --batch_size=8 \
  --num_workers=4 \
  --steps=8000 \
  --save_checkpoint=true \
  --save_freq=4000 \
  --wandb.enable=false \
  --output_dir="$VALUE_DIR" \
  --job_name="value_train_${RUN_ID}"

echo "===== Value inference: $(date) ====="
"$PY" -m accelerate.commands.launch \
  --num_processes 4 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --main_process_port "$((PORT + 1))" \
  -m lerobot.scripts.lerobot_value_infer \
  --dataset.repo_id="$DATASET_REPO" \
  --dataset.root="$DATASET_ROOT" \
  --inference.checkpoint_path="$VALUE_DIR" \
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
  --output_dir="$VALUE_INFER_DIR" \
  --job_name="value_infer_${RUN_ID}"

echo "===== Dataset report after value inference: $(date) ====="
"$PY" -m lerobot.scripts.lerobot_dataset_report --dataset="$DATASET_ROOT"

echo "===== PI05 policy train smoke with ACP: $(date) ====="
SMOKE_DIR="$CKPT_ROOT/smoke_policy_${RUN_ID}"
rm -rf "$SMOKE_DIR"
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
  --acp.enable=true \
  --acp.indicator_field=complementary_info.acp_indicator_v1 \
  --acp.indicator_dropout_prob=0.3 \
  --output_dir="$SMOKE_DIR" \
  --job_name="smoke_policy_${RUN_ID}" \
  --wandb.enable=false
rm -rf "$SMOKE_DIR"

echo "===== PI05 policy train full: $(date) ====="
"$PY" -m accelerate.commands.launch \
  --num_processes 4 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --main_process_port "$((PORT + 2))" \
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
  --steps=50000 \
  --acp.enable=true \
  --acp.indicator_field=complementary_info.acp_indicator_v1 \
  --acp.indicator_dropout_prob=0.3 \
  --output_dir="$POLICY_DIR" \
  --job_name="policy_train_pi05_${RUN_ID}" \
  --wandb.enable=false

echo "===== EvoRL pipeline finished: $(date) ====="
echo "POLICY_DIR=$POLICY_DIR"
