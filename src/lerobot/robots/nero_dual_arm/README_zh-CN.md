# NERO 接入说明

这份文档介绍 Evo-RL 中 `nero_dual_arm` 的使用方式。

当前这套接入的目标很明确：

- 让 Evo-RL 识别 `robot.type=nero_dual_arm`
- 让 Evo-RL 通过已有的 NERO server 控制双臂 NERO
- 让 Evo-RL 使用 Quest/Oculus 作为遥操作设备，也就是 `teleop.type=oculus_teleop`

这是一版**客户端侧接入**。
它没有把完整的 NERO 底层驱动、`pyAgxArm`、CAN 激活脚本和 server 逻辑全部搬进 Evo-RL。
它的设计是复用你已经能工作的外部 NERO server，然后让 Evo-RL 作为 client 去连接它。

## 包含哪些内容

NERO 机器人接入在当前目录下：

- `config_nero_dual_arm.py`
  - 注册 `robot.type=nero_dual_arm`
  - 定义 NERO robot 的配置字段，比如 `robot_ip`、`robot_port`、`use_gripper`、`cameras`
- `nero_interface_client.py`
  - 通过 `zerorpc` 连接外部 NERO server
  - 负责调用 server 暴露的读状态、发动作、夹爪控制等接口
- `nero_dual_arm.py`
  - 实现 LeRobot 标准 `Robot` 接口
  - 让 `lerobot-teleoperate`、`lerobot-record`、`lerobot-replay` 能使用 NERO

Quest/Oculus 遥操作接入在：

```text
Evo-RL/src/lerobot/teleoperators/oculus_teleop/
```

它负责：

- 注册 `teleop.type=oculus_teleop`
- 读取 Quest 左右手柄姿态和按键
- 输出双臂末端增量动作

## 总体架构

当前推荐的运行链路是：

```text
Quest / Oculus
    |
    v
Evo-RL Teleoperator: oculus_teleop
    |
    v
Evo-RL Robot: nero_dual_arm
    |
    v
zerorpc client
    |
    v
外部 NERO server
    |
    v
pyAgxArm / CAN / NERO 机械臂
```

也就是说：

- Evo-RL 负责上层工作流
  - 遥操作
  - 数据采集
  - 数据回放
  - 策略推理
- 外部 NERO server 负责底层硬件控制
  - `pyAgxArm`
  - CAN 通信
  - 夹爪控制
  - 底层 servo / IK 逻辑

## 你还需要准备什么

使用这套代码之前，你需要有一个已经能工作的 NERO server 环境。

通常也就是你之前的 NERO/ACT 项目里提供的这些内容：

- `nero_interface_server.py`
- `pyAgxArm`
- CAN 设备配置，例如 `can_left` 和 `can_right`
- 夹爪控制支持

当前 Evo-RL 中的 NERO 接入不会替代这个 server。
你仍然需要从原来的 NERO server 项目里启动 server。

## 安装依赖

### Python 依赖

在 Evo-RL 根目录执行：

```bash
pip install -e ".[agilex_teleop]"
```

这个 extra 当前包含：

- `zerorpc`
- `pure-python-adb`
- `scipy`

如果你还要在 Evo-RL 里使用 RealSense 相机，可以安装：

```bash
pip install -e ".[agilex_teleop,intelrealsense]"
```

### 系统依赖

如果要使用 Quest/Oculus 遥操作，运行 Evo-RL 的电脑需要安装 `adb`：

```bash
sudo apt install android-tools-adb
```

## 当前支持范围

当前已经接入的能力：

- `lerobot-teleoperate`
- `lerobot-record`
- `lerobot-replay`
- 通过外部 `zerorpc` server 控制 NERO
- 使用 Quest/Oculus 作为遥操作输入设备

当前没有内置的内容：

- NERO server 代码
- `pyAgxArm`
- CAN 激活脚本
- Quest teleop APK

关于 Quest APK 需要特别注意：

- 当前仓库包含 Quest reader 代码
- 但没有打包 Quest teleop APK
- 如果你的 Quest 里还没安装 teleop APK，需要先沿用旧项目里的 APK 安装流程

## 数据字段约定

为了兼容之前 NERO + ACT 项目的数据习惯，当前 NERO adapter 使用下面这些字段。

观测字段：

```text
left_ee_pose.x
left_ee_pose.y
left_ee_pose.z
left_ee_pose.rx
left_ee_pose.ry
left_ee_pose.rz
right_ee_pose.x
right_ee_pose.y
right_ee_pose.z
right_ee_pose.rx
right_ee_pose.ry
right_ee_pose.rz
left_gripper_cmd_bin
right_gripper_cmd_bin
```

如果配置了相机，观测里还会包含相机图像。

动作字段：

```text
left_delta_ee_pose.x
left_delta_ee_pose.y
left_delta_ee_pose.z
left_delta_ee_pose.rx
left_delta_ee_pose.ry
left_delta_ee_pose.rz
right_delta_ee_pose.x
right_delta_ee_pose.y
right_delta_ee_pose.z
right_delta_ee_pose.rx
right_delta_ee_pose.ry
right_delta_ee_pose.rz
left_gripper_cmd_bin
right_gripper_cmd_bin
```

这些字段会进入 LeRobotDataset，因此后续 ACT 或其他策略训练也会依赖这些名字。

## 部署方式

### 单机部署

如果只用一台电脑，这台电脑同时运行：

- 外部 NERO server
- Evo-RL 命令

这时：

- `robot.robot_ip` 通常填 `127.0.0.1`
- Quest 可以通过 USB 或无线 ADB 连接到这台电脑

这是最简单的调试方式。

### 双机部署

如果使用两台电脑：

- 机械臂侧电脑运行外部 NERO server
- 另一台电脑运行 Evo-RL

这时：

- `robot.robot_ip` 必须填 server 电脑的 IP
- Quest 必须连接到运行 Evo-RL 的电脑
- 运行 Evo-RL 的电脑必须能访问 server 电脑的 `4242` 端口

这种方式适合把底层 CAN 控制和上层采集、相机、策略推理分开。

## 第一步：启动外部 NERO server

在你之前的 NERO 控制项目里启动 server。

命令形式大致如下，具体路径以你的旧项目为准：

```bash
python nero/teleop/interface/nero_interface_server.py --ip 0.0.0.0 --port 4242
```

启动前请确认：

- NERO 机械臂已经正确上电
- CAN 设备已经激活
- 如果 server 代码使用 `can_left` / `can_right`，这两个接口必须存在
- 如果要用夹爪，夹爪也需要能被 server 正常初始化

## 第二步：检查 Quest 连接

在运行 Evo-RL 的电脑上检查 Quest：

```bash
adb devices
```

常见连接方式有两种。

USB 模式：

```text
Quest 通过 USB 线连接到运行 Evo-RL 的电脑
```

这种情况下可以不写 `--teleop.ip`。

无线 ADB 模式：

```text
Quest 和运行 Evo-RL 的电脑在同一个网络中
```

这种情况下需要设置：

```bash
--teleop.ip=<quest_ip>
```

如果不确定 Quest IP，可以运行：

```bash
adb shell ip route
```

## 第三步：使用长期 YAML 配置

长期采集数据时，建议直接修改这份配置文件，不要每次都在命令行里写一长串参数：

```text
src/lerobot/robots/nero_dual_arm/configs/record_nero_dual_arm.yaml
```

使用这份配置启动采集：

```bash
lerobot-record \
  --config_path=src/lerobot/robots/nero_dual_arm/configs/record_nero_dual_arm.yaml
```

这份 YAML 里包含：

- NERO server 地址
- Quest/Oculus 参数
- 相机序列号
- 数据集名称
- episode 时间
- 遥操作方向和尺度

正式使用前，通常需要改这些字段：

- `robot.robot_ip`
  - 单机部署用 `127.0.0.1`
  - 双机部署填 server 电脑的 IP
- `teleop.ip`
  - 无线 ADB 模式填 Quest IP
  - USB ADB 模式填 `null`
- `robot.cameras.*.serial_number_or_name`
  - 替换成你自己的 RealSense 相机序列号
- `dataset.repo_id`
  - 改成你的数据集名称，比如 `your_name/nero_demo`
- `dataset.single_task`
  - 改成当前采集任务描述

如果只是临时调试，可以在命令行覆盖 YAML 里的某些字段：

```bash
lerobot-record \
  --config_path=src/lerobot/robots/nero_dual_arm/configs/record_nero_dual_arm.yaml \
  --dataset.num_episodes=1 \
  --dataset.episode_time_s=10
```

## 第四步：使用 Quest 遥操作 NERO

如果 NERO server 和 Evo-RL 在同一台电脑上运行，可以先用这个最小命令：

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

如果 Quest 通过 USB 连接，可以省略 `--teleop.ip`：

```bash
lerobot-teleoperate \
  --robot.type=nero_dual_arm \
  --robot.robot_ip=127.0.0.1 \
  --robot.robot_port=4242 \
  --robot.use_gripper=true \
  --teleop.type=oculus_teleop \
  --teleop.use_gripper=true
```

如果 NERO server 在另一台电脑上，把 `robot.robot_ip` 改成 server 电脑的 IP：

```bash
--robot.robot_ip=<server_ip>
```

### 遥操作方向和尺度调节

Quest 到机械臂的映射可以通过下面这些参数调：

```bash
--teleop.left_pose_scaler='[0.3,0.3]'
--teleop.right_pose_scaler='[0.3,0.3]'
--teleop.left_channel_signs='[-1,-1,1,1,1,1]'
--teleop.right_channel_signs='[-1,-1,1,1,1,1]'
```

含义：

- `pose_scaler` 是 `[位置缩放, 姿态缩放]`
- `channel_signs` 是 `[x, y, z, rx, ry, rz]` 每个通道的正负号

如果你发现手柄往左，机械臂往右，通常就是 `channel_signs` 需要调整。

长期使用时，建议直接把这些值写进 `record_nero_dual_arm.yaml`：

```yaml
teleop:
  left_pose_scaler: [0.3, 0.3]
  right_pose_scaler: [0.3, 0.3]
  left_channel_signs: [-1, -1, 1, 1, 1, 1]
  right_channel_signs: [-1, -1, 1, 1, 1, 1]
```

## 第五步：采集数据

推荐使用配置文件启动：

```bash
lerobot-record \
  --config_path=src/lerobot/robots/nero_dual_arm/configs/record_nero_dual_arm.yaml
```

临时覆盖部分参数：

```bash
lerobot-record \
  --config_path=src/lerobot/robots/nero_dual_arm/configs/record_nero_dual_arm.yaml \
  --dataset.repo_id=<your_name>/nero_test \
  --dataset.num_episodes=5
```

### 在 YAML 里配置 RealSense 相机

示例 YAML 已经包含三路 RealSense 相机：

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

请把示例里的相机序列号替换成你自己的设备序列号。

## 第六步：回放数据

回放第 0 个 episode：

```bash
lerobot-replay \
  --robot.type=nero_dual_arm \
  --robot.robot_ip=127.0.0.1 \
  --robot.robot_port=4242 \
  --robot.use_gripper=true \
  --dataset.repo_id=<your_name>/nero_demo \
  --dataset.episode=0
```

## 参数说明

### `nero_dual_arm` 参数

- `robot_ip`
  - 外部 NERO server 的 IP
  - 单机部署一般是 `127.0.0.1`
- `robot_port`
  - 外部 NERO server 的 `zerorpc` 端口，默认 `4242`
- `use_gripper`
  - 是否发送夹爪命令
- `gripper_max_open`
  - 夹爪最大开口宽度，用于把归一化夹爪命令转成实际宽度
- `gripper_force`
  - 发送夹爪命令时使用的力
- `gripper_reverse`
  - 如果夹爪方向反了，可以设为 `true`
- `close_threshold`
  - 预留给二值夹爪开合逻辑的阈值
- `debug`
  - 如果设为 `true`，会跳过机械臂 servo 命令，适合只测试数据链路
- `cameras`
  - 相机配置字典

### `oculus_teleop` 参数

- `ip`
  - Quest 的 IP
  - USB 模式下可以不填
- `use_gripper`
  - 是否把 trigger 映射成夹爪命令
- `left_pose_scaler`
  - 左手柄 `[位置缩放, 姿态缩放]`
- `right_pose_scaler`
  - 右手柄 `[位置缩放, 姿态缩放]`
- `left_channel_signs`
  - 左手柄 `[x, y, z, rx, ry, rz]` 的正负号
- `right_channel_signs`
  - 右手柄 `[x, y, z, rx, ry, rz]` 的正负号

## 常见问题

### 连接不上 NERO server

检查：

- 外部 server 是否已经启动
- `robot.robot_ip` 是否正确
- `robot.robot_port` 是否正确
- 防火墙是否拦截了端口
- 双机部署时两台电脑是否在同一网络中

### `adb devices` 看不到 Quest

检查：

- Quest 是否开启开发者模式
- 头显里是否允许了 USB debugging
- 如果是无线 ADB，两台设备是否在同一网络
- 是否已经执行过 `adb connect <quest_ip>:5555`

### 遥操作方向不对

优先调：

- `left_channel_signs`
- `right_channel_signs`
- `left_pose_scaler`
- `right_pose_scaler`

第一次接 Quest 时需要调方向和尺度，这是正常现象。

### 提示 Quest APK 不存在或安装失败

当前 Evo-RL 仓库没有打包 Quest APK。
请先使用你旧 NERO teleop 项目里的 APK 安装流程，把 teleop APK 安装到 Quest 上。

### 机械臂能动，但数据集里没有图像

检查：

- 是否配置了 `robot.cameras`
- 相机序列号是否正确
- 是否安装了 RealSense 依赖
- 相机是否被其他程序占用

## 推荐调试顺序

建议按这个顺序排查，不要一上来就直接跑完整采集：

1. 先确认外部 NERO server 能独立控制机械臂。
2. 在运行 Evo-RL 的电脑上确认 `adb devices` 能看到 Quest。
3. 先不加相机，跑 `lerobot-teleoperate`。
4. 加入相机，再跑一次 `lerobot-teleoperate`。
5. 采一个很短的 `lerobot-record` 数据集。
6. 用 `lerobot-replay` 回放第一个 episode。

这样最容易定位问题到底出在 server、Quest、RPC、相机，还是 Evo-RL 配置上。
