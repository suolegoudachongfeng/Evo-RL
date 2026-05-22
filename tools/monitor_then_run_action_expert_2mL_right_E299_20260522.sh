#!/usr/bin/env bash
set -euo pipefail

CKPT_ROOT="/mnt/project_eai/chp/checkpoints"
LORA_LOG="$CKPT_ROOT/2mL_right_E299_prompt_v2_from_E249_evorl_lora_r16_20260522_2gpu_5k.log"
FULL_LOG="$CKPT_ROOT/2mL_right_E299_prompt_v2_from_E249_evorl_full_ft_20260522_2gpu_5k.log"
EXPERT_SCRIPT="$CKPT_ROOT/run_action_expert_only_2mL_right_E299_prompt_v2_from_E249_evorl_2gpu_5k_20260522.sh"
MONITOR_LOG="$CKPT_ROOT/monitor_action_expert_after_lora_full_E299_20260522.log"

exec > >(tee -a "$MONITOR_LOG") 2>&1

echo "===== Monitor start: $(date) ====="
echo "Waiting for both LoRA and full fine-tune to finish, then launching action-expert-only on GPUs 4,5."

while true; do
  if grep -q "===== LoRA E299 finished" "$LORA_LOG" 2>/dev/null && grep -q "===== Full FT E299 finished" "$FULL_LOG" 2>/dev/null; then
    echo "Both LoRA and full fine-tune completed: $(date)"
    break
  fi
  if grep -q -E "Traceback|RuntimeError|CUDA out of memory|Error:" "$LORA_LOG" 2>/dev/null; then
    echo "Potential error detected in LoRA log. Inspect $LORA_LOG"
    tail -n 80 "$LORA_LOG" || true
  fi
  if grep -q -E "Traceback|RuntimeError|CUDA out of memory|Error:" "$FULL_LOG" 2>/dev/null; then
    echo "Potential error detected in full fine-tune log. Inspect $FULL_LOG"
    tail -n 80 "$FULL_LOG" || true
  fi
  sleep 120
done

echo "===== Launching action-expert-only: $(date) ====="
GPU_SET=4,5 PORT=30441 bash "$EXPERT_SCRIPT"
echo "===== Monitor finished: $(date) ====="
