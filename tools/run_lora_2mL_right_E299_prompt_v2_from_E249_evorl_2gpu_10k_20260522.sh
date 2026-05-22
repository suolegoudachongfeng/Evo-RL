#!/usr/bin/env bash
set -euo pipefail

RUN_ID="2mL_right_E299_prompt_v2_from_E249_evorl_lora_r16_20260522_2gpu_10k"
EVO_RL_HOME="/mnt/project_eai/chp/workspace/Evo-RL"
DATASET_REPO="nero_task3_step1/2mL_right_E249_plus_dagger50_prompt_v2_policy_acp_20260522"
DATASET_ROOT="/mnt/project_eai/chp/datasets/converted_datasets/nero_task3_step1/2mL_right_E249_plus_dagger50_prompt_v2_policy_acp_20260522"
BASE_POLICY="/mnt/project_eai/chp/checkpoints/policy_train_pi05_2mL_right_retrain_20260520_E249_prompt_v2_4gpu_50k/checkpoints/050000/pretrained_model"
OUT="/mnt/project_eai/chp/checkpoints/policy_train_pi05_${RUN_ID}"
SMOKE_OUT="/mnt/project_eai/chp/checkpoints/smoke_pi05_${RUN_ID}"
LOG="/mnt/project_eai/chp/checkpoints/${RUN_ID}.log"
PY="/mnt/project_eai/chp/envs/evorl/bin/python"
PORT=30531
PEFT_TARGET_MODULES='(.*\.paligemma\..*\.self_attn\.(q|v)_proj|.*\.gemma_expert\..*\.self_attn\.(q|v)_proj|model\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out))'

exec > >(tee -a "$LOG") 2>&1

cd "$EVO_RL_HOME"
source /mnt/project_eai/chp/env.sh || true
export PYTHONPATH="$EVO_RL_HOME/src"
export HF_HOME="/mnt/project_eai/chp/hf_cache"
export HF_HUB_CACHE="/mnt/project_eai/chp/hf_cache/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCHDYNAMO_DISABLE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=6,7
unset TRANSFORMERS_CACHE

echo "===== LoRA E299 10k smoke: $(date) ====="
rm -rf "$SMOKE_OUT"
CUDA_VISIBLE_DEVICES=6 "$PY" -m lerobot.scripts.lerobot_train \
  --dataset.repo_id="$DATASET_REPO" \
  --dataset.root="$DATASET_ROOT" \
  --policy.type=pi05 \
  --policy.pretrained_path="$BASE_POLICY" \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.train_expert_only=false \
  --policy.gradient_checkpointing=false \
  --policy.push_to_hub=false \
  --policy.private=false \
  --peft.method_type=LORA \
  --peft.r=16 \
  --peft.target_modules="$PEFT_TARGET_MODULES" \
  --batch_size=1 \
  --steps=1 \
  --save_checkpoint=false \
  --acp.enable=true \
  --acp.indicator_field=complementary_info.acp_indicator_v1 \
  --acp.indicator_dropout_prob=0.3 \
  --output_dir="$SMOKE_OUT" \
  --job_name="smoke_${RUN_ID}" \
  --wandb.enable=false
rm -rf "$SMOKE_OUT"

echo "===== LoRA E299 10k full train: $(date) ====="
if [ -e "$OUT" ]; then
  mv "$OUT" "${OUT}.bak_$(date +%Y%m%d_%H%M%S)"
fi
"$PY" -m accelerate.commands.launch \
  --num_processes 2 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --main_process_port "$PORT" \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id="$DATASET_REPO" \
  --dataset.root="$DATASET_ROOT" \
  --policy.type=pi05 \
  --policy.pretrained_path="$BASE_POLICY" \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.train_expert_only=false \
  --policy.gradient_checkpointing=false \
  --policy.push_to_hub=false \
  --policy.private=false \
  --peft.method_type=LORA \
  --peft.r=16 \
  --peft.target_modules="$PEFT_TARGET_MODULES" \
  --batch_size=16 \
  --steps=10000 \
  --save_freq=1000 \
  --log_freq=50 \
  --acp.enable=true \
  --acp.indicator_field=complementary_info.acp_indicator_v1 \
  --acp.indicator_dropout_prob=0.3 \
  --output_dir="$OUT" \
  --job_name="$RUN_ID" \
  --wandb.enable=false

echo "===== LoRA E299 10k finished: $(date) ====="
echo "OUT=$OUT"
