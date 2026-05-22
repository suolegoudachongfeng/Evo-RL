#!/usr/bin/env bash
set -euo pipefail

REMOTE=${REMOTE:-wuwen}
NAME=pi05_2mL_right_E299_success_only_prompt_v2_bc_030000
REMOTE_MODEL=/mnt/project_eai/chp/checkpoints/policy_train_pi05_2mL_right_E299_success_only_prompt_v2_bc_pi05_20260522_4gpu_030000/checkpoints/030000/pretrained_model
LOCAL_ROOT=/media/deepcybo/T9/evorl_checkpoint
LOCAL_DIR="$LOCAL_ROOT/$NAME"
LINK=/home/deepcybo/Workspace/Evo-RL/checkpoints/$NAME
CONFIG=/home/deepcybo/Workspace/Evo-RL/src/lerobot/robots/nero_dual_arm/configs/deploy_${NAME}_4090.yaml
WAIT_INTERVAL_S=${WAIT_INTERVAL_S:-600}

echo "Waiting for $REMOTE:$REMOTE_MODEL/model.safetensors"
while true; do
  if ssh "$REMOTE" "test -f '$REMOTE_MODEL/model.safetensors'"; then
    echo "$(date '+%F %T') checkpoint is ready."
    break
  fi
  echo "$(date '+%F %T') not ready yet; sleeping ${WAIT_INTERVAL_S}s."
  sleep "$WAIT_INTERVAL_S"
done

mkdir -p "$LOCAL_ROOT" /home/deepcybo/Workspace/Evo-RL/checkpoints "$(dirname "$CONFIG")"
rsync -a --delete "$REMOTE:$REMOTE_MODEL/" "$LOCAL_DIR/pretrained_model/"
ln -sfn "$LOCAL_DIR" "$LINK"

cat > "$CONFIG" <<YAML
# NERO pi05 policy deployment config for the 4090 client.
# Model: 2 mL right-column E299 success-only prompt-v2 pure-BC policy, step 30000.

robot:
  type: nero_dual_arm
  id: nero_dual_arm
  robot_ip: 10.10.10.1
  robot_port: 4242
  use_gripper: true
  gripper_max_open: 0.1
  gripper_force: 2.0
  gripper_reverse: false
  close_threshold: 0.05
  control_mode: oculus
  debug: false
  action_send_freq_hz: 50.0
  cameras:
    left_wrist_image:
      type: intelrealsense
      serial_number_or_name: "412622270929"
      width: 424
      height: 240
      fps: 30
      use_depth: false
    right_wrist_image:
      type: intelrealsense
      serial_number_or_name: "412622270701"
      width: 424
      height: 240
      fps: 30
      use_depth: false
    head_image:
      type: intelrealsense
      serial_number_or_name: "339322074423"
      width: 424
      height: 240
      fps: 30
      use_depth: false

teleop: null

policy:
  type: pi05
  pretrained_path: /home/deepcybo/Workspace/Evo-RL/checkpoints/$NAME/pretrained_model
  device: cuda
  dtype: bfloat16
  push_to_hub: false
  # Inference-only knobs. You can change these without retraining.
  n_action_steps: 30
  num_inference_steps: 10

dataset:
  repo_id: nero_task3_step1/eval_2mL_right_E299_success_only_prompt_v2_bc_030000
  single_task: "2mL_two_vials_right_column. Object: two small 2 mL vials. Goal: pick up the two small vials one by one and place them upright into the rightmost column of the empty rack. Targets: the bottom-right corner hole and the top-right corner hole. Constraint: use the right column only; do not place either vial into the left column, center holes, or any other hole."
  auto_version_repo_id: true
  root: null
  fps: 30
  episode_time_s: 600
  reset_time_s: 300
  num_episodes: 3
  video: true
  push_to_hub: false
  private: false
  tags: ["nero", "dual-arm", "pi05", "deployment", "2mL-right", "E299", "success-only", "prompt-v2", "bc", "step-030000"]
  num_image_writer_processes: 0
  num_image_writer_threads_per_camera: 4
  video_encoding_batch_size: 1
  vcodec: libsvtav1
  rename_map: {}

display_data: false
display_ip: null
display_port: null
display_compressed_images: false
play_sounds: true
resume: false

policy_sync_to_teleop: false
policy_sync_parallel: true
intervention_state_machine_enabled: false
enable_episode_outcome_labeling: true
episode_success_key: s
episode_failure_key: f
require_episode_success_label: false
default_episode_success: null
enable_collector_policy_id: true
collector_policy_id_policy: $NAME
collector_policy_id_human: human
communication_retry_timeout_s: 2.0
communication_retry_interval_s: 0.1

acp_inference:
  enable: true
  use_cfg: false
  cfg_beta: 1.0
YAML

echo "Pulled to: $LOCAL_DIR"
echo "Symlink: $LINK -> $LOCAL_DIR"
echo "YAML: $CONFIG"
