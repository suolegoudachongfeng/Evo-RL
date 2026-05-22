#!/usr/bin/env bash
set -euo pipefail

export EVO_RL_HOME=/mnt/project_eai/chp/workspace/Evo-RL
export HF_HOME=/mnt/project_eai/chp/hf_cache
export HF_HUB_CACHE=/mnt/project_eai/chp/hf_cache/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCHDYNAMO_DISABLE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${EVO_RL_HOME}/src"
unset TRANSFORMERS_CACHE

export CUDA_VISIBLE_DEVICES=4,5,6,7

PYTHON=/mnt/project_eai/chp/envs/evorl/bin/python
DATA_ROOT=/mnt/project_eai/chp/datasets/converted_datasets/nero_task3_step1/eval_dagger_2mL_right_E249_prompt_v2_evorl_050000_20260521_v01
DATA_REPO=nero_task3_step1/eval_dagger_2mL_right_E249_prompt_v2_evorl_050000_20260521_v01
BASE_POLICY=/mnt/project_eai/chp/checkpoints/policy_train_pi05_2mL_right_retrain_20260520_E249_prompt_v2_4gpu_50k/checkpoints/050000/pretrained_model
OUT=/mnt/project_eai/chp/checkpoints/policy_train_pi05_2mL_right_E249_prompt_v2_evorl_050000_dagger25_full_ft_20260521_003000_4gpu
SMOKE_OUT=/mnt/project_eai/chp/checkpoints/smoke_pi05_2mL_right_E249_prompt_v2_evorl_050000_dagger25_full_ft_4gpu

cd "${EVO_RL_HOME}"

echo "===== DAgger full PI05 smoke: $(date) ====="
rm -rf "${SMOKE_OUT}"
CUDA_VISIBLE_DEVICES=4 "${PYTHON}" -m lerobot.scripts.lerobot_train \
  --dataset.repo_id="${DATA_REPO}" \
  --dataset.root="${DATA_ROOT}" \
  --policy.type=pi05 \
  --policy.pretrained_path="${BASE_POLICY}" \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.train_expert_only=false \
  --policy.gradient_checkpointing=true \
  --policy.push_to_hub=false \
  --policy.private=false \
  --batch_size=8 \
  --steps=1 \
  --save_checkpoint=false \
  --acp.enable=true \
  --acp.indicator_field=complementary_info.acp_indicator_v1 \
  --acp.indicator_dropout_prob=0.3 \
  --output_dir="${SMOKE_OUT}" \
  --job_name=smoke_pi05_2mL_right_E249_prompt_v2_evorl_050000_dagger25_full_ft_4gpu \
  --wandb.enable=false
rm -rf "${SMOKE_OUT}"

echo "===== DAgger full PI05 4-GPU fine-tune: $(date) ====="
if [ -e "${OUT}" ]; then
  mv "${OUT}" "${OUT}.backup_$(date +%Y%m%d_%H%M%S)"
fi

"${PYTHON}" -m accelerate.commands.launch \
  --num_processes 4 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --main_process_port 29921 \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id="${DATA_REPO}" \
  --dataset.root="${DATA_ROOT}" \
  --policy.type=pi05 \
  --policy.pretrained_path="${BASE_POLICY}" \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.train_expert_only=false \
  --policy.gradient_checkpointing=true \
  --policy.push_to_hub=false \
  --policy.private=false \
  --batch_size=8 \
  --steps=3000 \
  --save_freq=3000 \
  --log_freq=50 \
  --acp.enable=true \
  --acp.indicator_field=complementary_info.acp_indicator_v1 \
  --acp.indicator_dropout_prob=0.3 \
  --output_dir="${OUT}" \
  --job_name=policy_train_pi05_2mL_right_E249_prompt_v2_evorl_050000_dagger25_full_ft_003000_4gpu \
  --wandb.enable=false

echo "===== DAgger full PI05 fine-tune finished: $(date) ====="
