# Evo-RL NERO Training Handoff Report

## 1. Goal

This document explains how to continue the Evo-RL workflow for the NERO dual-arm robot project.
It is intended as a one-stop handoff note for another AI assistant or a new developer.

The current goal is to migrate existing NERO teleoperation data into the Evo-RL pipeline and run the offline training workflow:

```text
Legacy NERO/LeRobot dataset
-> Convert to Evo-RL-compatible schema
-> Add success/failure episode labels
-> Train a value function
-> Run value inference on the dataset
-> Write value / advantage / ACP indicator back to the dataset
-> Later train a text-conditioned policy with ACP
```

Evo-RL is not replacing the NERO hardware server. In this project, Evo-RL mainly provides:

- A LeRobot-compatible client interface for NERO.
- Dataset recording and schema support.
- Offline value-function training.
- Advantage and ACP indicator generation.
- Later policy training with advantage-conditioned text tags.

## 2. Machine Roles

### Local Mac

Local project path:

```text
/Users/chp/Workspace/ZGC_WBCD/Evo-RL
```

Use the Mac for:

- Code editing and Git management.
- SSH access.
- Downloading or uploading files if the training server has limited network access.
- Running Claude/Codex/other AI tools.

### NERO Robot Computer

SSH alias:

```bash
ssh nero-server
```

Known details:

```text
User: geist
Path: /home/geist/wbcd_workspace/Evo-RL
Conda env: dual_arm_data
```

Use this machine for:

- Starting the existing NERO server.
- Connecting to the physical robot.
- Collecting teleoperation data.
- Inspecting old datasets under the LeRobot cache.

### H100 Training Server

The current H100 server is reached through an SSH tunnel.

Known current server info:

```text
Hostname: is-dcvhnqjpnmha4esn-devmachine-0
IP: 172.27.113.169
GPU: NVIDIA H100 80GB HBM3
```

Recommended paths:

```text
/mnt/project_eai/chp/workspace/Evo-RL
/mnt/project_eai/chp/datasets
/mnt/project_eai/chp/checkpoints
/mnt/project_eai/chp/hf_cache
/mnt/project_eai/chp/envs/evorl
```

Avoid putting large project data, models, or environments under `/workspace` or `/root`.
Use `/mnt/project_eai/chp/...` instead.

## 3. SSH Access To H100

The H100 internal IP may change when the cloud instance is recreated.
The previous IP `172.27.113.181` is no longer valid. The current known IP is:

```text
172.27.113.169
```

The OpenSSH config should look like this:

```sshconfig
Host wuwen
  HostName 172.27.113.169
  User root
  Port 22
  IdentityFile ~/.ssh/id_ed25519
  IdentitiesOnly yes
  ProxyCommand ssh -W %h:%p -J shenzhaolong@ssh-jumper.cloud.infini-ai.com root@aic-dcvhhla5bblypwpq
```

For tools that cannot understand `ProxyCommand`, such as `ssh-mcp`, open a local tunnel:

```bash
ssh -fN -L 127.0.0.1:22269:172.27.113.169:22 \
  -J shenzhaolong@ssh-jumper.cloud.infini-ai.com \
  root@aic-dcvhhla5bblypwpq
```

Then test:

```bash
ssh -p 22269 root@127.0.0.1 \
  -i ~/.ssh/id_ed25519 \
  -o IdentitiesOnly=yes \
  'hostname; hostname -i; nvidia-smi'
```

Expected result:

```text
is-dcvhnqjpnmha4esn-devmachine-0
172.27.113.169
NVIDIA H100 80GB HBM3
```

Claude Desktop MCP can connect through the tunnel:

```json
"ssh-wuwen": {
  "command": "npx",
  "args": [
    "-y",
    "ssh-mcp",
    "--",
    "--host=127.0.0.1",
    "--port=22269",
    "--user=root",
    "--key=/Users/chp/.ssh/id_ed25519",
    "--timeout=60000"
  ]
}
```

## 4. Relationship Between Evo-RL And NERO Control

The low-level NERO robot control is still handled by the existing server-side stack:

```text
NERO SDK / pyAgxArm / hardware control
```

Evo-RL adds a LeRobot-compatible client layer. The practical runtime flow is:

```text
Start NERO server
-> Start Evo-RL recording client
-> Use Oculus / Quest teleoperation
-> Save LeRobot/Evo-RL-format dataset
```

In other words:

- The existing NERO server talks to the robot hardware.
- Evo-RL acts as a client for data collection and policy deployment.
- Evo-RL records extra fields needed for human-in-the-loop and offline RL.

## 5. Dataset Structure

The old project data mostly follows the LeRobot dataset format:

```text
dataset/
  data/
    chunk-000/
      file-000.parquet
  meta/
    info.json
    stats.json
    episodes/
      chunk-000/file-000.parquet
    tasks.parquet
  videos/
    ...
```

Core fields:

```text
observation.state
action
observation.images.*
task
episode_index
frame_index
timestamp
```

The current NERO data uses:

```text
observation.state: 28 dimensions
action: 14 dimensions
```

Evo-RL additionally expects:

```text
complementary_info.policy_action
complementary_info.is_intervention
complementary_info.state
episode_success
```

Field meaning:

```text
complementary_info.policy_action
The action proposed by the deployed policy. For old pure-human demos, this is filled with zeros.

complementary_info.is_intervention
Whether the human intervened at the current frame. For old pure-human demos without DAgger logic, this is 0.

complementary_info.state
Internal human-in-the-loop state field used by Evo-RL. For converted old data, this is 0.

episode_success
Episode-level success/failure label. The value function needs this supervision.
```

## 6. Existing Converted Dataset

One converted dataset is already available on the H100:

```text
/mnt/project_eai/chp/datasets/converted_datasets/2mL_empty_merged_20260427_evorl
```

Recommended repo id:

```text
nero_task3_step1/2mL_empty_merged_20260427_evorl
```

Source dataset:

```text
/home/geist/.cache/huggingface/lerobot/nero_task3_step1/2mL_empty_merged_20260427
```

Known properties:

```text
100 episodes
81000 frames
observation.state = 28D
action = 14D
success = 100
failure = 0
is_intervention = 0
```

Check it with:

```bash
cd /mnt/project_eai/chp/workspace/Evo-RL
source /mnt/project_eai/chp/env.sh
unset TRANSFORMERS_CACHE

/mnt/project_eai/chp/envs/evorl/bin/python -m lerobot.scripts.lerobot_dataset_report \
  --dataset nero_task3_step1/2mL_empty_merged_20260427_evorl \
  --root /mnt/project_eai/chp/datasets/converted_datasets/2mL_empty_merged_20260427_evorl
```

## 7. Converting Legacy NERO Datasets

Use:

```bash
python -m lerobot.scripts.lerobot_convert_nero_legacy_dataset
```

Example on the H100:

```bash
cd /mnt/project_eai/chp/workspace/Evo-RL
source /mnt/project_eai/chp/env.sh
unset TRANSFORMERS_CACHE

/mnt/project_eai/chp/envs/evorl/bin/python -m lerobot.scripts.lerobot_convert_nero_legacy_dataset \
  --dataset /path/to/old_dataset \
  --output-repo-id nero_task3_step1/your_dataset_evorl \
  --output-dir /mnt/project_eai/chp/datasets/converted_datasets/your_dataset_evorl
```

Important constraints:

- The old dataset must already use the 28D NERO `observation.state`.
- Old 14D-state datasets should not be converted with this script.
- The script does not delete the source dataset.
- Always output to a separate new directory.

Add success/failure labels:

```bash
/mnt/project_eai/chp/envs/evorl/bin/python -m lerobot.scripts.lerobot_patch_episode_success \
  --dataset /mnt/project_eai/chp/datasets/converted_datasets/your_dataset_evorl \
  --default-success success \
  --overwrite
```

If some episodes failed:

```bash
/mnt/project_eai/chp/envs/evorl/bin/python -m lerobot.scripts.lerobot_patch_episode_success \
  --dataset /mnt/project_eai/chp/datasets/converted_datasets/your_dataset_evorl \
  --default-success success \
  --failure-episodes 3,7,10-12 \
  --overwrite
```

## 8. Value Function Principle

The current default Evo-RL value model is:

```text
pistar06
```

This is not an open-source pi0.6 policy. In this repository, `pistar06` is a value-model stack.
It uses two Hugging Face backbones:

```text
google/siglip-so400m-patch14-384
google/gemma-3-270m
```

Roles:

```text
SigLIP
Processes visual observations and extracts image features.

Gemma
Processes task text and text-form state information.

Fusion/value head
Combines images, state, and task text to predict a scalar value.
```

Intuitively, the value function estimates how good the current state is with respect to task success.
Successful episodes provide positive trajectory supervision; failed episodes provide negative supervision.

## 9. H100 Environment

Before training, run:

```bash
cd /mnt/project_eai/chp/workspace/Evo-RL
source /mnt/project_eai/chp/env.sh
unset TRANSFORMERS_CACHE
```

The environment file should look like:

```bash
export EVO_RL_HOME=/mnt/project_eai/chp/workspace/Evo-RL
export HF_HOME=/mnt/project_eai/chp/hf_cache
export HF_HUB_CACHE=/mnt/project_eai/chp/hf_cache/hub
export TORCHDYNAMO_DISABLE=1
export PYTHONPATH=/mnt/project_eai/chp/workspace/Evo-RL/src:${PYTHONPATH:-}
```

Check GPU:

```bash
nvidia-smi
```

Check Python and CUDA:

```bash
/mnt/project_eai/chp/envs/evorl/bin/python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
PY
```

## 10. Hugging Face Cache Check

The H100 may not have internet access. Model weights should be cached under:

```text
/mnt/project_eai/chp/hf_cache
```

Check offline loading:

```bash
source /mnt/project_eai/chp/env.sh
unset TRANSFORMERS_CACHE

/mnt/project_eai/chp/envs/evorl/bin/python - <<'PY'
from transformers import AutoConfig, AutoImageProcessor, AutoTokenizer

print(AutoConfig.from_pretrained("google/siglip-so400m-patch14-384", local_files_only=True).__class__)
print(AutoImageProcessor.from_pretrained("google/siglip-so400m-patch14-384", local_files_only=True).__class__)
print(AutoConfig.from_pretrained("google/gemma-3-270m", local_files_only=True).__class__)
print(AutoTokenizer.from_pretrained("google/gemma-3-270m", local_files_only=True).__class__)
print("HF cache OK")
PY
```

If this fails, check:

- `HF_HOME`.
- `HF_HUB_CACHE`.
- Whether `TRANSFORMERS_CACHE` is accidentally set.
- Whether both model repos are fully downloaded.
- Whether Hugging Face gated access for Gemma has been accepted.

## 11. Value Training Smoke Test

Always run a 1-step smoke test before a long training run:

```bash
cd /mnt/project_eai/chp/workspace/Evo-RL
source /mnt/project_eai/chp/env.sh
unset TRANSFORMERS_CACHE

rm -rf /mnt/project_eai/chp/checkpoints/value_train_smoke_h100

/mnt/project_eai/chp/envs/evorl/bin/python -m lerobot.scripts.lerobot_value_train \
  --dataset.repo_id=nero_task3_step1/2mL_empty_merged_20260427_evorl \
  --dataset.root=/mnt/project_eai/chp/datasets/converted_datasets/2mL_empty_merged_20260427_evorl \
  --value.type=pistar06 \
  --value.dtype=bfloat16 \
  --value.freeze_vision_encoder=true \
  --value.freeze_language_model=true \
  --batch_size=1 \
  --num_workers=0 \
  --steps=1 \
  --save_checkpoint=false \
  --wandb.enable=false \
  --output_dir=/mnt/project_eai/chp/checkpoints/value_train_smoke_h100 \
  --job_name=value_train_smoke_h100
```

If this succeeds, the dataset, environment, GPU, and model cache are basically working.

## 12. Full Value Training

Recommended first full run:

```bash
cd /mnt/project_eai/chp/workspace/Evo-RL
source /mnt/project_eai/chp/env.sh
unset TRANSFORMERS_CACHE

/mnt/project_eai/chp/envs/evorl/bin/python -m lerobot.scripts.lerobot_value_train \
  --dataset.repo_id=nero_task3_step1/2mL_empty_merged_20260427_evorl \
  --dataset.root=/mnt/project_eai/chp/datasets/converted_datasets/2mL_empty_merged_20260427_evorl \
  --value.type=pistar06 \
  --value.dtype=bfloat16 \
  --value.freeze_vision_encoder=true \
  --value.freeze_language_model=true \
  --batch_size=16 \
  --num_workers=4 \
  --steps=8000 \
  --save_checkpoint=true \
  --save_freq=4000 \
  --wandb.enable=false \
  --output_dir=/mnt/project_eai/chp/checkpoints/value_train_2mL_empty_v1 \
  --job_name=value_train_2mL_empty_v1
```

On an H100 80GB, try larger batch sizes if stable:

```text
batch_size=32
batch_size=64
```

If CUDA OOM occurs, reduce `batch_size`.

## 13. Value Inference And ACP Fields

After value training, run value inference to write these fields back to the dataset:

```text
complementary_info.value_v1
complementary_info.advantage_v1
complementary_info.acp_indicator_v1
```

Back up the dataset first:

```bash
cp -a \
  /mnt/project_eai/chp/datasets/converted_datasets/2mL_empty_merged_20260427_evorl \
  /mnt/project_eai/chp/datasets/converted_datasets/2mL_empty_merged_20260427_evorl_backup_$(date +%Y%m%d_%H%M%S)
```

Run inference:

```bash
cd /mnt/project_eai/chp/workspace/Evo-RL
source /mnt/project_eai/chp/env.sh
unset TRANSFORMERS_CACHE

/mnt/project_eai/chp/envs/evorl/bin/python -m lerobot.scripts.lerobot_value_infer \
  --dataset.repo_id=nero_task3_step1/2mL_empty_merged_20260427_evorl \
  --dataset.root=/mnt/project_eai/chp/datasets/converted_datasets/2mL_empty_merged_20260427_evorl \
  --inference.checkpoint_path=/mnt/project_eai/chp/checkpoints/value_train_2mL_empty_v1 \
  --runtime.device=cuda \
  --runtime.batch_size=32 \
  --runtime.num_workers=4 \
  --acp.enable=true \
  --acp.n_step=50 \
  --acp.positive_ratio=0.3 \
  --acp.value_field=complementary_info.value_v1 \
  --acp.advantage_field=complementary_info.advantage_v1 \
  --acp.indicator_field=complementary_info.acp_indicator_v1 \
  --output_dir=/mnt/project_eai/chp/checkpoints/value_infer_2mL_empty_v1 \
  --job_name=value_infer_2mL_empty_v1
```

Important parameters:

```text
acp.n_step
N-step advantage horizon.

acp.positive_ratio
Fraction of high-advantage frames marked as positive ACP samples.

value_field
Output column for predicted value.

advantage_field
Output column for advantage.

indicator_field
Output column for binary ACP label.
```

## 14. Policy Training After Value Inference

ACP policy training requires a policy that supports task text, because ACP is injected into task text.

Suitable policy families:

```text
pi0
pi05
pi0_fast
evo1
Other VLA-style text-conditioned policies
```

Less suitable:

```text
ACT
```

ACT can still be used as a baseline, but it does not naturally consume text/task tags, so it cannot fully use ACP.

General policy training template:

```bash
/mnt/project_eai/chp/envs/evorl/bin/python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=nero_task3_step1/2mL_empty_merged_20260427_evorl \
  --dataset.root=/mnt/project_eai/chp/datasets/converted_datasets/2mL_empty_merged_20260427_evorl \
  --policy.type=<POLICY_TYPE> \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --batch_size=32 \
  --steps=30000 \
  --acp.enable=true \
  --acp.indicator_field=complementary_info.acp_indicator_v1 \
  --acp.indicator_dropout_prob=0.3 \
  --output_dir=/mnt/project_eai/chp/checkpoints/policy_train_v1 \
  --job_name=policy_train_v1 \
  --wandb.enable=false
```

The concrete `<POLICY_TYPE>` and pretrained model path depend on the selected policy.

## 15. Closed-Loop Data Collection

Evo-RL closed-loop collection means:

```text
Train policy/value on existing data
-> Deploy the policy
-> Let the human intervene when needed
-> Record the policy action and intervention flag
-> Add the new data to the next training iteration
```

During policy deployment:

```text
policy_action
The action that the policy wanted to execute.

action
The final action actually executed by the robot. This may be a human action during intervention.

is_intervention
1 when a human has taken over; 0 otherwise.
```

For old pure teleoperation data:

```text
policy_action = 0
is_intervention = 0
```

This is expected and correct.

## 16. Common Failure Modes

### H100 SSH Fails

Check whether the cloud instance IP changed.

Current known IP:

```text
172.27.113.169
```

Old invalid IP:

```text
172.27.113.181
```

### Claude MCP Cannot Connect To H100

Do not make `ssh-mcp` connect directly to the internal IP.
Start a local tunnel and connect MCP to `127.0.0.1:22269`.

### Value Training Hangs On Model Download

Likely causes:

- H100 cannot access the internet.
- Hugging Face cache is incomplete.
- Gemma gated model access was not accepted.
- `TRANSFORMERS_CACHE` points to an empty or wrong directory.

### Dataset Cannot Be Loaded

Check:

- `--dataset.repo_id`.
- `--dataset.root`.
- `meta/info.json`.
- `observation.state` is 28D.
- `action` is 14D.
- `episode_success` exists.

### Value Inference Writeback Fails

Recommendations:

- Back up the dataset first.
- Check `--inference.checkpoint_path`.
- Use unique field suffixes such as `_v1`, `_v2`, `_v3`.

## 17. Recommended Next Steps

The safest continuation sequence is:

```text
1. Connect to H100 and check nvidia-smi.
2. Confirm /mnt/project_eai/chp/workspace/Evo-RL exists.
3. source /mnt/project_eai/chp/env.sh.
4. unset TRANSFORMERS_CACHE.
5. Run the Hugging Face cache check.
6. Run dataset_report.
7. Run 1-step value_train smoke test.
8. Run full value_train.
9. Back up the dataset.
10. Run value_infer and write advantage fields.
11. Only then start policy training.
```

The most important rule is:

```text
Do not start long training before the 1-step smoke test succeeds.
```

