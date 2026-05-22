#!/usr/bin/env bash
set -euo pipefail

export EVO_RL_HOME=${EVO_RL_HOME:-/mnt/project_eai/chp/workspace/Evo-RL}
export PY=${PY:-/mnt/project_eai/chp/envs/evorl/bin/python}
export HF_HOME=${HF_HOME:-/mnt/project_eai/chp/hf_cache}
export HF_HUB_CACHE=${HF_HUB_CACHE:-/mnt/project_eai/chp/hf_cache/hub}
export HF_HUB_OFFLINE=1
export TORCHDYNAMO_DISABLE=1
export PYTHONPATH="$EVO_RL_HOME/src"
unset TRANSFORMERS_CACHE

GPU_SET=${GPU_SET:-4,5,6,7}
PORT=${PORT:-30621}
WAIT_MAX_USED_MB=${WAIT_MAX_USED_MB:-2000}
WAIT_INTERVAL_S=${WAIT_INTERVAL_S:-300}
NUM_GPUS=$(python3 - <<PY
print(len("${GPU_SET}".split(",")))
PY
)
FIRST_GPU=${GPU_SET%%,*}

DATASET_REPO_ID=nero_task3_step1/2mL_right_E299_success_only_prompt_v2_policy_acp_20260522
DATASET_ROOT=/mnt/project_eai/chp/datasets/converted_datasets/nero_task3_step1/2mL_right_E299_success_only_prompt_v2_policy_acp_20260522
PI05_BASE=/mnt/project_eai/chp/hf_cache/hub/models--lerobot--pi05_base
JOB_NAME=2mL_right_E299_success_only_prompt_v2_bc_pi05_20260522_4gpu_030000
OUTPUT_DIR=/mnt/project_eai/chp/checkpoints/policy_train_pi05_${JOB_NAME}
LOG=/mnt/project_eai/chp/checkpoints/${JOB_NAME}.log

wait_for_gpus() {
  while true; do
    local busy=0
    IFS=',' read -ra ids <<< "$GPU_SET"
    for id in "${ids[@]}"; do
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$id" | tr -d ' ')
      if [ "$used" -ge "$WAIT_MAX_USED_MB" ]; then
        busy=1
      fi
    done
    if [ "$busy" -eq 0 ]; then
      echo "$(date '+%F %T') GPUs $GPU_SET are free enough; starting training." | tee -a "$LOG"
      break
    fi
    echo "$(date '+%F %T') waiting for GPUs $GPU_SET to be free..." | tee -a "$LOG"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tee -a "$LOG"
    sleep "$WAIT_INTERVAL_S"
  done
}

cd "$EVO_RL_HOME"
mkdir -p "$(dirname "$LOG")"

{
  echo "=== Success-only BC PI05 training ==="
  date
  echo "GPU_SET=$GPU_SET NUM_GPUS=$NUM_GPUS PORT=$PORT"
  echo "DATASET_ROOT=$DATASET_ROOT"
  echo "PI05_BASE=$PI05_BASE"
  echo "OUTPUT_DIR=$OUTPUT_DIR"
} | tee -a "$LOG"

wait_for_gpus

echo "=== smoke test ===" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="$FIRST_GPU" "$PY" -m lerobot.scripts.lerobot_train \
  --dataset.repo_id="$DATASET_REPO_ID" \
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
  --save_checkpoint=false \
  --output_dir=/mnt/project_eai/chp/checkpoints/smoke_pi05_${JOB_NAME} \
  --job_name=smoke_${JOB_NAME} \
  --wandb.enable=false 2>&1 | tee -a "$LOG"

echo "=== full 30k BC training ===" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="$GPU_SET" "$PY" -m accelerate.commands.launch \
  --num_processes "$NUM_GPUS" \
  --num_machines 1 \
  --mixed_precision bf16 \
  --main_process_port "$PORT" \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.root="$DATASET_ROOT" \
  --policy.type=pi05 \
  --policy.pretrained_path="$PI05_BASE" \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.push_to_hub=false \
  --policy.private=false \
  --batch_size=8 \
  --steps=30000 \
  --save_freq=10000 \
  --log_freq=50 \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$JOB_NAME" \
  --wandb.enable=false 2>&1 | tee -a "$LOG"

echo "=== training finished ===" | tee -a "$LOG"
date | tee -a "$LOG"
