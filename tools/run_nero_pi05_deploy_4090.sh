#!/usr/bin/env bash
set -euo pipefail

# Run from any directory without relying on a global PYTHONPATH export.
# Do not source this script; execute it with bash so the environment stays local.
REPO_ROOT="/home/deepcybo/Workspace/Evo-RL"
DEFAULT_CONFIG="src/lerobot/robots/nero_dual_arm/configs/deploy_pi05_2mL_right_4090.yaml"

CONFIG_PATH="${1:-$DEFAULT_CONFIG}"
if [[ $# -gt 0 ]]; then
  shift
fi

cd "$REPO_ROOT"
PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}" \
  python -m lerobot.scripts.lerobot_record \
  --config_path="$CONFIG_PATH" \
  "$@"
