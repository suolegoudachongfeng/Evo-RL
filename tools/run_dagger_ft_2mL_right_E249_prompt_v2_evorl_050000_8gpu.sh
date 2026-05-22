#!/usr/bin/env bash
set -euo pipefail

export EVO_RL_HOME=/mnt/project_eai/chp/workspace/Evo-RL
export HF_HOME=/mnt/project_eai/chp/hf_cache
export HF_HUB_CACHE=/mnt/project_eai/chp/hf_cache/hub
export HF_HUB_OFFLINE=1
export TORCHDYNAMO_DISABLE=1
export PYTHONPATH="${EVO_RL_HOME}/src"
unset TRANSFORMERS_CACHE

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

PYTHON=/mnt/project_eai/chp/envs/evorl/bin/python
DATA_ROOT=/mnt/project_eai/chp/datasets/converted_datasets/nero_task3_step1/eval_dagger_2mL_right_E249_prompt_v2_evorl_050000_20260521_v01
DATA_REPO=nero_task3_step1/eval_dagger_2mL_right_E249_prompt_v2_evorl_050000_20260521_v01
BASE_POLICY=/mnt/project_eai/chp/checkpoints/policy_train_pi05_2mL_right_retrain_20260520_E249_prompt_v2_4gpu_50k/checkpoints/050000/pretrained_model
OUT=/mnt/project_eai/chp/checkpoints/policy_train_pi05_2mL_right_E249_prompt_v2_evorl_050000_dagger25_20260521_ft_005000_8gpu
SMOKE_OUT=/mnt/project_eai/chp/checkpoints/smoke_pi05_2mL_right_E249_prompt_v2_evorl_050000_dagger25_8gpu

cd "${EVO_RL_HOME}"

echo "===== DAgger PI05 smoke: $(date) ====="
rm -rf "${SMOKE_OUT}"
"${PYTHON}" -m lerobot.scripts.lerobot_train \
  --dataset.repo_id="${DATA_REPO}" \
  --dataset.root="${DATA_ROOT}" \
  --policy.type=pi05 \
  --policy.pretrained_path="${BASE_POLICY}" \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.train_expert_only=true \
  --policy.gradient_checkpointing=false \
  --policy.push_to_hub=false \
  --policy.private=false \
  --batch_size=1 \
  --steps=1 \
  --save_checkpoint=false \
  --acp.enable=true \
  --acp.indicator_field=complementary_info.acp_indicator_v1 \
  --acp.indicator_dropout_prob=0.3 \
  --output_dir="${SMOKE_OUT}" \
  --job_name=smoke_pi05_2mL_right_E249_prompt_v2_evorl_050000_dagger25 \
  --wandb.enable=false

echo "===== DAgger PI05 full 8-GPU fine-tune: $(date) ====="
if [ -e "${OUT}" ]; then
  mv "${OUT}" "${OUT}.backup_$(date +%Y%m%d_%H%M%S)"
fi

"${PYTHON}" -m accelerate.commands.launch \
  --num_processes 8 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --main_process_port 29883 \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id="${DATA_REPO}" \
  --dataset.root="${DATA_ROOT}" \
  --policy.type=pi05 \
  --policy.pretrained_path="${BASE_POLICY}" \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.train_expert_only=true \
  --policy.gradient_checkpointing=false \
  --policy.push_to_hub=false \
  --policy.private=false \
  --batch_size=8 \
  --steps=5000 \
  --save_freq=2500 \
  --log_freq=50 \
  --acp.enable=true \
  --acp.indicator_field=complementary_info.acp_indicator_v1 \
  --acp.indicator_dropout_prob=0.3 \
  --output_dir="${OUT}" \
  --job_name=policy_train_pi05_2mL_right_E249_prompt_v2_evorl_050000_dagger25_ft_005000_8gpu \
  --wandb.enable=false

echo "===== DAgger PI05 fine-tune finished: $(date) ====="
