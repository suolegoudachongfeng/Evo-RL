#!/usr/bin/env bash
set -euo pipefail

RUN_ID="2mL_right_E299_prompt_v2_from_E249_evorl_full_ft_20260522_2gpu_5k"
EVO_RL_HOME="/mnt/project_eai/chp/workspace/Evo-RL"
CKPT_ROOT="/mnt/project_eai/chp/checkpoints"
CONFIG_PATH="$CKPT_ROOT/policy_train_pi05_${RUN_ID}/checkpoints/005000/pretrained_model/train_config.json"
LOG="$CKPT_ROOT/resume_${RUN_ID}_to10k.log"
PY="/mnt/project_eai/chp/envs/evorl/bin/python"
PORT=30511

exec > >(tee -a "$LOG") 2>&1

echo "===== Resume full-ft E299 policy 5k -> 10k: $(date) ====="
echo "CONFIG_PATH=$CONFIG_PATH"
echo "TARGET_STEPS=10000"

cd "$EVO_RL_HOME"
source /mnt/project_eai/chp/env.sh || true
export PYTHONPATH="$EVO_RL_HOME/src"
export HF_HOME="/mnt/project_eai/chp/hf_cache"
export HF_HUB_CACHE="/mnt/project_eai/chp/hf_cache/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCHDYNAMO_DISABLE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=4,5
unset TRANSFORMERS_CACHE

"$PY" -m accelerate.commands.launch \
  --num_processes 2 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --main_process_port "$PORT" \
  -m lerobot.scripts.lerobot_train \
  --config_path="$CONFIG_PATH" \
  --resume=true \
  --steps=10000 \
  --save_freq=1000 \
  --log_freq=50 \
  --wandb.enable=false

echo "===== Resume full-ft E299 policy finished: $(date) ====="

