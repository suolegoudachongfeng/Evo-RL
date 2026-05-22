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

PYTHON=/mnt/project_eai/chp/envs/evorl/bin/python
DATA_ROOT=/mnt/project_eai/chp/datasets/converted_datasets/nero_task3_step1/eval_dagger_2mL_right_E249_prompt_v2_evorl_050000_20260521_v01
DATA_REPO=nero_task3_step1/eval_dagger_2mL_right_E249_prompt_v2_evorl_050000_20260521_v01
BASE_POLICY=/mnt/project_eai/chp/checkpoints/policy_train_pi05_2mL_right_retrain_20260520_E249_prompt_v2_4gpu_50k/checkpoints/050000/pretrained_model
BENCH_ROOT=/mnt/project_eai/chp/checkpoints/batch_bench_2mL_right_E249_20260521
PEFT_TARGET_MODULES='(.*\.paligemma\..*\.self_attn\.(q|v)_proj|.*\.gemma_expert\..*\.self_attn\.(q|v)_proj|model\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out))'

mkdir -p "${BENCH_ROOT}"
cd "${EVO_RL_HOME}"

run_lora() {
  local bs="$1"
  local log="${BENCH_ROOT}/lora_bs${bs}.log"
  echo "===== LORA bs=${bs} start $(date) =====" | tee "${log}"
  rm -rf "${BENCH_ROOT}/lora_bs${bs}_out"
  set +e
  CUDA_VISIBLE_DEVICES=0,1,2,3 "${PYTHON}" -m accelerate.commands.launch \
    --num_processes 4 \
    --num_machines 1 \
    --mixed_precision bf16 \
    --main_process_port "$((30000 + bs))" \
    -m lerobot.scripts.lerobot_train \
    --dataset.repo_id="${DATA_REPO}" \
    --dataset.root="${DATA_ROOT}" \
    --policy.type=pi05 \
    --policy.pretrained_path="${BASE_POLICY}" \
    --policy.device=cuda \
    --policy.dtype=bfloat16 \
    --policy.train_expert_only=false \
    --policy.gradient_checkpointing=false \
    --policy.push_to_hub=false \
    --policy.private=false \
    --peft.method_type=LORA \
    --peft.r=16 \
    --peft.target_modules="${PEFT_TARGET_MODULES}" \
    --batch_size="${bs}" \
    --steps=20 \
    --save_checkpoint=false \
    --log_freq=5 \
    --acp.enable=true \
    --acp.indicator_field=complementary_info.acp_indicator_v1 \
    --acp.indicator_dropout_prob=0.3 \
    --output_dir="${BENCH_ROOT}/lora_bs${bs}_out" \
    --job_name="bench_lora_bs${bs}" \
    --wandb.enable=false >> "${log}" 2>&1
  local status=$?
  set -e
  echo "===== LORA bs=${bs} status=${status} end $(date) =====" | tee -a "${log}"
  return 0
}

run_full() {
  local bs="$1"
  local log="${BENCH_ROOT}/full_bs${bs}.log"
  echo "===== FULL bs=${bs} start $(date) =====" | tee "${log}"
  rm -rf "${BENCH_ROOT}/full_bs${bs}_out"
  set +e
  CUDA_VISIBLE_DEVICES=4,5,6,7 "${PYTHON}" -m accelerate.commands.launch \
    --num_processes 4 \
    --num_machines 1 \
    --mixed_precision bf16 \
    --main_process_port "$((30100 + bs))" \
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
    --batch_size="${bs}" \
    --steps=20 \
    --save_checkpoint=false \
    --log_freq=5 \
    --acp.enable=true \
    --acp.indicator_field=complementary_info.acp_indicator_v1 \
    --acp.indicator_dropout_prob=0.3 \
    --output_dir="${BENCH_ROOT}/full_bs${bs}_out" \
    --job_name="bench_full_bs${bs}" \
    --wandb.enable=false >> "${log}" 2>&1
  local status=$?
  set -e
  echo "===== FULL bs=${bs} status=${status} end $(date) =====" | tee -a "${log}"
  return 0
}

case "${1:-all}" in
  lora)
    for bs in 8 16 24; do run_lora "${bs}"; done
    ;;
  full)
    for bs in 1 2 4; do run_full "${bs}"; done
    ;;
  all)
    for bs in 8 16 24; do run_lora "${bs}"; done
    for bs in 1 2 4; do run_full "${bs}"; done
    ;;
  lora20)
    run_lora 20
    ;;
  full8)
    run_full 8
    ;;
  *)
    echo "Usage: $0 [lora|full|all|lora20|full8]" >&2
    exit 2
    ;;
esac

echo "===== summary ====="
grep -H -E "status=|step:[0-9K]+|out of memory|CUDA|RuntimeError|Traceback" "${BENCH_ROOT}"/*.log || true
