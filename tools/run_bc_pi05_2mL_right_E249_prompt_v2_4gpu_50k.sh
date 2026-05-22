#!/usr/bin/env bash
set -euo pipefail

RUN_ID="2mL_right_retrain_20260520_E249_prompt_v2_bc_pi05_4gpu_50k"
EVO_RL_HOME="/mnt/project_eai/chp/workspace/Evo-RL"
DATASET_REPO="nero_task3_step1/2mL_right_retrain_20260520_E249_prompt_v2_evorl"
DATASET_ROOT="/mnt/project_eai/chp/datasets/converted_datasets/nero_task3_step1/2mL_right_retrain_20260520_E249_prompt_v2_evorl"
CKPT_ROOT="/mnt/project_eai/chp/checkpoints"
POLICY_DIR="$CKPT_ROOT/policy_train_pi05_${RUN_ID}"
LOG="$CKPT_ROOT/bc_pi05_${RUN_ID}.log"
PY="/mnt/project_eai/chp/envs/evorl/bin/python"
PORT=29971

exec > >(tee -a "$LOG") 2>&1

echo "===== BC-only PI05 train start: $(date) ====="
echo "RUN_ID=$RUN_ID"
echo "DATASET_REPO=$DATASET_REPO"
echo "DATASET_ROOT=$DATASET_ROOT"
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
export CUDA_VISIBLE_DEVICES=4,5,6,7
unset TRANSFORMERS_CACHE

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
for name in config.json model.safetensors policy_preprocessor.json policy_postprocessor.json; do
  if [ ! -f "$PI05_BASE/$name" ]; then
    echo "PI05 base file missing: $PI05_BASE/$name" >&2
    exit 4
  fi
done

if [ -e "$POLICY_DIR" ]; then
  bak="${POLICY_DIR}.bak_$(date +%Y%m%d_%H%M%S)"
  echo "Existing output dir found, moving: $POLICY_DIR -> $bak"
  mv "$POLICY_DIR" "$bak"
fi

echo "===== Dataset report before BC train ====="
"$PY" -m lerobot.scripts.lerobot_dataset_report --dataset="$DATASET_ROOT"

echo "===== PI05 BC smoke: $(date) ====="
SMOKE_DIR="$CKPT_ROOT/smoke_policy_${RUN_ID}"
rm -rf "$SMOKE_DIR"
CUDA_VISIBLE_DEVICES=4 "$PY" -m lerobot.scripts.lerobot_train \
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
  --acp.enable=false \
  --output_dir="$SMOKE_DIR" \
  --job_name="smoke_policy_${RUN_ID}" \
  --wandb.enable=false
rm -rf "$SMOKE_DIR"

echo "===== PI05 BC full: $(date) ====="
"$PY" -m accelerate.commands.launch \
  --num_processes 4 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --main_process_port "$PORT" \
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
  --acp.enable=false \
  --output_dir="$POLICY_DIR" \
  --job_name="policy_train_pi05_${RUN_ID}" \
  --wandb.enable=false

echo "===== BC-only PI05 train finished: $(date) ====="
echo "POLICY_DIR=$POLICY_DIR"
