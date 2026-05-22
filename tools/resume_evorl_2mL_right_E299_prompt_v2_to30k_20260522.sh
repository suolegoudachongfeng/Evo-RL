#!/usr/bin/env bash
set -euo pipefail

RUN_ID="2mL_right_E299_prompt_v2_evorl_from_scratch_20260522_4gpu_15k"
EVO_RL_HOME="/mnt/project_eai/chp/workspace/Evo-RL"
CKPT_ROOT="/mnt/project_eai/chp/checkpoints"
CONFIG_PATH="$CKPT_ROOT/policy_train_pi05_${RUN_ID}/checkpoints/015000/pretrained_model/train_config.json"
LOG="$CKPT_ROOT/resume_${RUN_ID}_to30k.log"
PY="/mnt/project_eai/chp/envs/evorl/bin/python"
PORT=30501

exec > >(tee -a "$LOG") 2>&1

echo "===== Resume EvoRL E299 policy 15k -> 30k: $(date) ====="
echo "CONFIG_PATH=$CONFIG_PATH"
echo "TARGET_STEPS=30000"

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

"$PY" -m accelerate.commands.launch \
  --num_processes 4 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --main_process_port "$PORT" \
  -m lerobot.scripts.lerobot_train \
  --config_path="$CONFIG_PATH" \
  --resume=true \
  --steps=30000 \
  --save_freq=1000 \
  --log_freq=50 \
  --wandb.enable=false

echo "===== Resume EvoRL E299 policy finished: $(date) ====="

