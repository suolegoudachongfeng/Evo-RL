# NERO Integration

[中文说明](./README_zh-CN.md)

This directory contains the first-stage NERO integration for Evo-RL/LeRobot.

The goal of this integration is practical:

- let Evo-RL recognize `robot.type=nero_dual_arm`
- let Evo-RL control a NERO dual-arm robot through your existing NERO server
- let Evo-RL use Quest/Oculus as a teleoperation source with `teleop.type=oculus_teleop`

This integration is intentionally **client-side**.
It does **not** bundle the full NERO hardware driver stack into Evo-RL.
Instead, it reuses an already working external NERO control server.

## What Is Included

In this repo, the NERO-related pieces are:

- `config_nero_dual_arm.py`
  - registers `robot.type=nero_dual_arm`
- `nero_interface_client.py`
  - zerorpc client for the external NERO server
- `nero_dual_arm.py`
  - LeRobot `Robot` adapter for NERO

Quest teleoperation support is included in:

- `/Users/chp/Workspace/ZGC_WBCD/chp/Evo-RL/src/lerobot/teleoperators/oculus_teleop/`
  - registers `teleop.type=oculus_teleop`
  - reads Quest controller poses
  - converts them into bimanual delta end-effector actions

## Architecture

This integration assumes the following control chain:

```text
Quest / Oculus
    |
    v
Evo-RL Teleoperator (`oculus_teleop`)
    |
    v
Evo-RL Robot (`nero_dual_arm`)
    |
    v
zerorpc client
    |
    v
External NERO server
    |
    v
pyAgxArm / CAN / NERO
```

The important design choice is:

- Evo-RL handles the **workflow layer**
  - teleoperation
  - data recording
  - replay
  - policy execution
- your existing NERO server handles the **hardware layer**
  - `pyAgxArm`
  - CAN
  - gripper
  - low-level servo / IK logic

## What You Still Need Outside This Repo

Before using this integration, you need a working NERO server environment.

Typically this means you already have a previous NERO project that provides:

- `nero_interface_server.py`
- `pyAgxArm`
- CAN setup for `can_left` / `can_right`
- gripper support

This repo does **not** replace that server.
You should keep using your existing server project and launch the server from there.

## Dependencies

### Python dependencies

Install Evo-RL with the NERO/Quest extra dependencies:

```bash
pip install -e ".[agilex_teleop]"
```

This extra currently adds:

- `zerorpc`
- `pure-python-adb`
- `scipy`

If you also use RealSense cameras from Evo-RL, install the RealSense extra too:

```bash
pip install -e ".[agilex_teleop,intelrealsense]"
```

### System dependencies

For Quest teleoperation you also need `adb` on the machine running Evo-RL:

```bash
sudo apt install android-tools-adb
```

## Current Scope And Limitations

This integration is intentionally narrow.

What works in scope:

- `lerobot-teleoperate`
- `lerobot-record`
- `lerobot-replay`
- NERO robot access through an external zerorpc server
- Quest/Oculus as a teleoperator

What is **not** bundled here:

- NERO server code
- `pyAgxArm`
- CAN activation scripts
- Quest APK bundle

Important note about the Quest APK:

- the reader code is included here
- but the Quest teleop APK is **not** bundled in this repo
- if the APK is not already installed on your headset, install it from your previous NERO teleop project first

## Data Schema Used By This Integration

To stay compatible with the previous NERO ACT project, the NERO robot adapter uses:

Observation keys:

- `left_ee_pose.x`
- `left_ee_pose.y`
- `left_ee_pose.z`
- `left_ee_pose.rx`
- `left_ee_pose.ry`
- `left_ee_pose.rz`
- `right_ee_pose.x`
- `right_ee_pose.y`
- `right_ee_pose.z`
- `right_ee_pose.rx`
- `right_ee_pose.ry`
- `right_ee_pose.rz`
- `left_gripper_cmd_bin`
- `right_gripper_cmd_bin`
- camera observations

Action keys:

- `left_delta_ee_pose.x`
- `left_delta_ee_pose.y`
- `left_delta_ee_pose.z`
- `left_delta_ee_pose.rx`
- `left_delta_ee_pose.ry`
- `left_delta_ee_pose.rz`
- `right_delta_ee_pose.x`
- `right_delta_ee_pose.y`
- `right_delta_ee_pose.z`
- `right_delta_ee_pose.rx`
- `right_delta_ee_pose.ry`
- `right_delta_ee_pose.rz`
- `left_gripper_cmd_bin`
- `right_gripper_cmd_bin`

This makes it easier to port over your previous teleoperation and ACT data conventions.

## Deployment Modes

There are two supported deployment styles.

### Mode 1: Single machine

You run both:

- the external NERO server
- Evo-RL commands

on the same computer.

In this case:

- `robot.robot_ip` should usually be `127.0.0.1`
- Quest can be connected to the same computer through USB or wireless ADB

This is the simplest setup.

### Mode 2: Two machines

You run:

- the external NERO server on the robot-side machine
- Evo-RL on another machine

In this case:

- `robot.robot_ip` must point to the server machine IP
- Quest must be connected to the Evo-RL machine, not the server machine
- the Evo-RL machine must be able to reach the NERO server over the network

This setup is useful when:

- the robot-side machine is dedicated to CAN and low-level control
- the Evo-RL machine is dedicated to teleop, cameras, recording, and policy execution

## Step 1: Start The External NERO Server

Start the server from your existing NERO control project.

The exact command depends on your previous project, but conceptually it looks like:

```bash
python nero/teleop/interface/nero_interface_server.py --ip 0.0.0.0 --port 4242
```

Before starting it, make sure:

- the NERO robot is powered correctly
- the CAN interfaces are configured
- `can_left` and `can_right` are available if your server expects those names
- the grippers are available if you plan to use them

## Step 2: Verify Quest Connectivity

On the machine running Evo-RL, check that Quest is visible:

```bash
adb devices
```

Two common modes:

- USB mode
  - use a cable
  - in this case `teleop.ip` can be omitted
- wireless ADB mode
  - connect Quest over network
  - set `teleop.ip=<quest_ip>`

If you use wireless ADB and are not sure about the Quest IP:

```bash
adb shell ip route
```

## Step 3: Use The Long-Term YAML Config

For day-to-day recording, prefer editing this config file instead of rewriting all CLI arguments:

```text
src/lerobot/robots/nero_dual_arm/configs/record_nero_dual_arm.yaml
```

Run recording with:

```bash
lerobot-record \
  --config_path=src/lerobot/robots/nero_dual_arm/configs/record_nero_dual_arm.yaml
```

The config contains the robot server address, Quest settings, camera serials, dataset settings, timing, and teleop tuning values.

Edit these fields before use:

- `robot.robot_ip`
  - set to `127.0.0.1` for single-machine mode
  - set to the robot-side server IP for two-machine mode
- `teleop.ip`
  - set to the Quest IP for wireless ADB
  - set to `null` for USB ADB
- `robot.cameras.*.serial_number_or_name`
  - replace the example RealSense serial numbers with your own
- `dataset.repo_id`
  - set the base dataset name, for example `nero_task3_step1/8mL_empty`
- `dataset.auto_version_repo_id`
  - keep this as `true` if you want legacy-style names like `nero_task3_step1/8mL_empty_20260424_v01`
- `dataset.single_task`
  - write the task description saved into the dataset

When `dataset.auto_version_repo_id=true`, `lerobot-record` checks the local dataset folder before creating a dataset and automatically picks the next `vXX` suffix. For example, `nero_task3_step1/8mL_empty` becomes `nero_task3_step1/8mL_empty_YYYYMMDD_vXX` at runtime. If `dataset.root` is `null`, the dataset is saved under the default LeRobot cache; if `dataset.root` is set, it is treated as the dataset storage root.

You can still override any field from the command line for quick tests:

```bash
lerobot-record \
  --config_path=src/lerobot/robots/nero_dual_arm/configs/record_nero_dual_arm.yaml \
  --dataset.num_episodes=1 \
  --dataset.episode_time_s=10
```

## Step 4: Teleoperate NERO From Evo-RL

Minimal example:

```bash
lerobot-teleoperate \
  --robot.type=nero_dual_arm \
  --robot.robot_ip=127.0.0.1 \
  --robot.robot_port=4242 \
  --robot.use_gripper=true \
  --teleop.type=oculus_teleop \
  --teleop.ip=192.168.110.62 \
  --teleop.use_gripper=true
```

If Quest is connected by USB to the same machine, use:

```bash
lerobot-teleoperate \
  --robot.type=nero_dual_arm \
  --robot.robot_ip=127.0.0.1 \
  --robot.robot_port=4242 \
  --robot.use_gripper=true \
  --teleop.type=oculus_teleop \
  --teleop.use_gripper=true
```

### Optional teleop tuning

You can tune the Quest-to-robot mapping with:

- `--teleop.left_pose_scaler='[1.2,1.2]'`
- `--teleop.right_pose_scaler='[1.2,1.2]'`
- `--teleop.left_channel_signs='[-1,-1,1,1,1,1]'`
- `--teleop.right_channel_signs='[-1,-1,1,1,1,1]'`

These values are especially useful when migrating from your previous ACT/NERO project.

For long-term use, put these values in `record_nero_dual_arm.yaml` under:

```yaml
teleop:
  left_pose_scaler: [1.2, 1.2]
  right_pose_scaler: [1.2, 1.2]
  left_channel_signs: [-1, -1, 1, 1, 1, 1]
  right_channel_signs: [-1, -1, 1, 1, 1, 1]
```

## Step 5: Record A Dataset

Recommended config-based command:

```bash
lerobot-record \
  --config_path=src/lerobot/robots/nero_dual_arm/configs/record_nero_dual_arm.yaml
```

For a one-off override:

```bash
lerobot-record \
  --config_path=src/lerobot/robots/nero_dual_arm/configs/record_nero_dual_arm.yaml \
  --dataset.repo_id=<your_name>/nero_test \
  --dataset.num_episodes=5
```

### Camera config in YAML

The example YAML already includes three RealSense cameras:

```yaml
robot:
  cameras:
    left_wrist_image:
      type: intelrealsense
      serial_number_or_name: "412622270929"
      width: 424
      height: 240
      fps: 30
    right_wrist_image:
      type: intelrealsense
      serial_number_or_name: "412622270701"
      width: 424
      height: 240
      fps: 30
    head_image:
      type: intelrealsense
      serial_number_or_name: "339322074423"
      width: 424
      height: 240
      fps: 30
```

Replace the serial numbers with your own camera serials.

## Step 6: Replay A Recorded Episode

```bash
lerobot-replay \
  --robot.type=nero_dual_arm \
  --robot.robot_ip=127.0.0.1 \
  --robot.robot_port=4242 \
  --robot.use_gripper=true \
  --dataset.repo_id=<your_name>/nero_demo \
  --dataset.episode=0
```

## Configuration Reference

### Robot config

`nero_dual_arm` currently supports these main fields:

- `robot_ip`
  - IP of the external NERO server
- `robot_port`
  - zerorpc port of the external NERO server
- `use_gripper`
  - whether to send gripper commands
- `gripper_max_open`
  - max opening width used for normalization
- `gripper_force`
  - force used when commanding the grippers
- `gripper_reverse`
  - invert gripper direction if needed
- `close_threshold`
  - reserved threshold for binary close/open logic
- `debug`
  - if true, action computation still runs but robot servo commands are skipped
- `cameras`
  - camera config dictionary

### Teleop config

`oculus_teleop` currently supports these main fields:

- `ip`
  - Quest IP for wireless ADB
  - omit this field for USB ADB mode
- `use_gripper`
  - whether to map trigger values to gripper commands
- `left_pose_scaler`
- `right_pose_scaler`
  - `[position_scale, orientation_scale]`
- `left_channel_signs`
- `right_channel_signs`
  - per-axis sign flips for `[x, y, z, rx, ry, rz]`

## Troubleshooting

### 1. `Failed to connect to NERO server`

Check:

- the external server is actually running
- `robot_ip` and `robot_port` are correct
- firewall or LAN rules are not blocking the port

### 2. `adb devices` does not show the Quest

Check:

- developer mode is enabled on the headset
- USB debugging is allowed
- if using wireless ADB, both devices are on the same network

### 3. Teleop direction is wrong

Tune:

- `left_channel_signs`
- `right_channel_signs`
- `left_pose_scaler`
- `right_pose_scaler`

This is expected during first-time alignment.

### 4. Quest reader asks for an APK or fails to install

This repo does not ship the Quest APK.
Reuse the APK installation flow from your previous NERO teleop project first.

### 5. Robot moves but recording is missing images

Check:

- camera config is attached under `robot.cameras`
- all camera serials are correct
- the required camera dependencies are installed

## Recommended First Validation Order

Use this order to reduce debugging time:

1. Verify the external NERO server works on its own.
2. Verify `adb devices` sees the Quest on the Evo-RL machine.
3. Run `lerobot-teleoperate` without cameras.
4. Add cameras and rerun `lerobot-teleoperate`.
5. Run `lerobot-record` for a tiny dataset.
6. Run `lerobot-replay` on one episode.

That usually gives the fastest path to a stable setup.
