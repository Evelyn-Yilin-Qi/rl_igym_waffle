# 观察空间 (Observation Space) 详细说明

## 概述

WaffleDrive RL 环境的观察空间是一个 **44 维向量**，由以下四个部分组成：

```
Observation = [LiDAR(36维) | User Input(2维) | Base Velocity(2维) | Action History(4维)]
```

---

## 1. LiDAR 距离测量 (维度 0-35, 共 36 维)

### 基本信息
- **维度索引**: `[0:36]`
- **数据类型**: `float32`
- **取值范围**: `[0.0, 1.0]`
- **物理单位**: 归一化距离（原始单位为米）

### 计算公式

```python
lidar_normalized = clip(lidar_ranges / lidar_max_range, 0.0, 1.0)
```

其中：
- `lidar_ranges`: 原始 LiDAR 距离测量值（单位：米）
- `lidar_max_range`: 最大测量范围 = **3.0 米**（配置在 `cfg/task/WaffleDrive.yaml`）

### 物理含义

- **射线数量**: 36 条射线
- **扫描角度**: 均匀分布在 `[0, 2π)` 范围内，即每 10° 一条射线
- **射线方向**: 相对于机器人局部坐标系，0° 对应机器人前进方向（x 轴正方向）
- **测量方法**: 使用 `omni.physx.get_physx_scene_query_interface().raycast_closest()` 进行真实射线检测

### 取值含义

| 取值 | 物理含义 | 说明 |
|------|---------|------|
| `0.0` | 距离 = 0 米 | 射线直接命中障碍物或边界墙 |
| `0.0 < x < 1.0` | 距离 = x × 3.0 米 | 在测量范围内检测到障碍物 |
| `1.0` | 距离 ≥ 3.0 米 | 超出最大测量范围，未检测到障碍物 |

### 示例

- `lidar[0] = 0.5` → 机器人前方 1.5 米处有障碍物
- `lidar[9] = 1.0` → 机器人右侧 90° 方向 3 米内无障碍物
- `lidar[18] = 0.0` → 机器人后方直接碰撞

---

## 2. User Input 用户意图向量 (维度 36-37, 共 2 维)

### 基本信息
- **维度索引**: `[36:38]`
- **数据类型**: `float32`
- **取值范围**: `ux, uy ∈ [-1, 1]`，且满足 **√(ux² + uy²) = 1**（单位向量）
- **物理含义**: 从机器人当前位置指向目标位置的单位方向向量（在机器人局部坐标系中）

### 计算公式

```python
# 1. 计算环境坐标系中的方向向量
robot_local = robot_position - env_origin
goal_local = goal_position - env_origin
dir_vec = goal_local[:2] - robot_local[:2]  # (N, 2)

# 2. 归一化为单位向量
dist = ||dir_vec||
norm = max(dist, 1e-8)  # 避免除零
dir_unit = dir_vec / norm  # (ux, uy)

# 3. 转换到机器人局部坐标系
# 根据机器人当前朝向（yaw 角）进行旋转变换
ux, uy = rotate_to_robot_frame(dir_unit, robot_yaw)
```

### 物理含义

- **坐标系**: 机器人局部坐标系（x 轴为前进方向，y 轴为左侧方向）
- **单位向量**: 模长始终为 1，只表示方向，不表示距离
- **方向定义**:
  - `ux > 0`: 目标在机器人前方
  - `ux < 0`: 目标在机器人后方
  - `uy > 0`: 目标在机器人左侧
  - `uy < 0`: 目标在机器人右侧

### 取值含义

| 取值组合 | 物理含义 | 说明 |
|---------|---------|------|
| `ux = 1.0, uy = 0.0` | 目标在正前方 | 机器人只需前进 |
| `ux = 0.0, uy = 1.0` | 目标在正左侧 | 机器人需要左转 90° |
| `ux = -1.0, uy = 0.0` | 目标在正后方 | 机器人需要掉头 |
| `ux = 0.707, uy = 0.707` | 目标在左前方 45° | 机器人需要前进并左转 |
| `ux = -0.5, uy = -0.866` | 目标在右后方 120° | 机器人需要后退并右转 |

### 约束条件

- **单位向量约束**: `ux² + uy² = 1.0`（允许负值）
- **允许负值**: `ux` 和 `uy` 都可以为负，表示目标在机器人后方或右侧

---

## 3. Base Velocity 基础速度 (维度 38-39, 共 2 维)

### 基本信息
- **维度索引**: `[38:40]`
- **数据类型**: `float32`
- **取值范围**: `[-1.0, 1.0]`
- **物理含义**: 当前时刻测量的线速度和角速度（归一化后）

### 计算公式

```python
# 1. 从机器人速度张量中提取
v = velocities[:, 0]  # 线速度 v（机器人局部坐标系 x 方向）
w = velocities[:, 5]  # 角速度 w（绕 z 轴）

# 2. 归一化
v_norm = clip(v / max_v, -1.0, 1.0)
w_norm = clip(w / max_w, -1.0, 1.0)
```

其中：
- `max_v = 0.5 m/s`（最大线速度，配置在 `cfg/task/WaffleDrive.yaml`）
- `max_w = 1.0 rad/s`（最大角速度，配置在 `cfg/task/WaffleDrive.yaml`）

### 物理含义

#### v (线速度)
- **物理单位**: 米/秒 (m/s)
- **坐标系**: 机器人局部坐标系 x 方向（前进方向）
- **取值含义**:
  - `v_norm > 0`: 机器人前进
  - `v_norm < 0`: 机器人后退
  - `v_norm = 0`: 机器人静止（线速度方向）

#### w (角速度)
- **物理单位**: 弧度/秒 (rad/s)
- **旋转轴**: 绕 z 轴（垂直向上）
- **取值含义**:
  - `w_norm > 0`: 逆时针旋转（从上方俯视）
  - `w_norm < 0`: 顺时针旋转（从上方俯视）
  - `w_norm = 0`: 不旋转

### 取值含义

| 取值 | 物理速度 | 说明 |
|------|---------|------|
| `v_norm = 1.0` | `v = 0.5 m/s` | 最大前进速度 |
| `v_norm = -1.0` | `v = -0.5 m/s` | 最大后退速度 |
| `v_norm = 0.0` | `v = 0.0 m/s` | 静止 |
| `w_norm = 1.0` | `w = 1.0 rad/s` | 最大逆时针角速度（约 57.3°/s） |
| `w_norm = -1.0` | `w = -1.0 rad/s` | 最大顺时针角速度 |
| `w_norm = 0.0` | `w = 0.0 rad/s` | 不旋转 |

### 示例

- `[v_norm=0.6, w_norm=0.3]` → 机器人以 0.3 m/s 前进，同时以 0.3 rad/s 逆时针旋转
- `[v_norm=-0.2, w_norm=-0.8]` → 机器人以 0.1 m/s 后退，同时以 0.8 rad/s 顺时针旋转
- `[v_norm=0.0, w_norm=1.0]` → 机器人原地逆时针旋转

### 重要说明

- **测量值 vs 命令值**: Base Velocity 是**实际测量的速度**，不是发送给机器人的命令。由于物理限制、摩擦等因素，测量值可能与命令值不同。
- **来源**: 从 `robots.get_velocities()` 获取，返回 `[vx, vy, vz, wx, wy, wz]`，其中 `vx` 是线速度，`wz` 是角速度。

---

## 4. Action History 动作历史 (维度 40-43, 共 4 维)

### 基本信息
- **维度索引**: `[40:44]`
- **数据类型**: `float32`
- **取值范围**: `[-1.0, 1.0]`
- **物理含义**: 历史时刻测量的线速度和角速度（归一化后）

### 计算公式

```python
# 1. 获取历史时刻的测量速度（物理值）
v_t_minus_1 = prev_measured_v  # t-1 时刻的测量线速度
w_t_minus_1 = prev_measured_w  # t-1 时刻的测量角速度
v_t_minus_2 = action_history[:, 2]  # t-2 时刻的测量线速度（从历史中获取）
w_t_minus_2 = action_history[:, 3]  # t-2 时刻的测量角速度（从历史中获取）

# 2. 归一化
action_history_normalized[:, 0] = clip(v_t_minus_1 / max_v, -1.0, 1.0)  # v_t-1
action_history_normalized[:, 1] = clip(w_t_minus_1 / max_w, -1.0, 1.0)  # w_t-1
action_history_normalized[:, 2] = clip(v_t_minus_2 / max_v, -1.0, 1.0)  # v_t-2
action_history_normalized[:, 3] = clip(v_t_minus_2 / max_w, -1.0, 1.0)  # w_t-2
```

其中：
- `max_v = 0.5 m/s`
- `max_w = 1.0 rad/s`

### 维度组成

```
Action History = [v_t-1, w_t-1, v_t-2, w_t-2]
```

- **`[40]`: v_t-1** - t-1 时刻（上一时刻）的测量线速度
- **`[41]`: w_t-1** - t-1 时刻（上一时刻）的测量角速度
- **`[42]`: v_t-2** - t-2 时刻（上上时刻）的测量线速度
- **`[43]`: w_t-2** - t-2 时刻（上上时刻）的测量角速度

### 物理含义

- **时间步**: 假设控制频率为 40 Hz，时间步长为 `dt = 0.025 s`
  - `t-1`: 25 毫秒前
  - `t-2`: 50 毫秒前
- **测量值**: Action History 存储的是**历史时刻的测量速度**，不是命令动作。这与 Base Velocity 一致，都是实际测量的速度值。

### 取值含义

与 Base Velocity 相同，每个值的含义如下：

| 维度 | 取值 | 物理含义 |
|------|------|---------|
| `v_t-1` | `1.0` | 上一时刻以最大速度前进 |
| `v_t-1` | `-1.0` | 上一时刻以最大速度后退 |
| `w_t-1` | `1.0` | 上一时刻以最大角速度逆时针旋转 |
| `w_t-1` | `-1.0` | 上一时刻以最大角速度顺时针旋转 |

### 更新逻辑

```python
# 在每个时间步：
# 1. 获取当前测量速度
current_v = robot_velocities[:, 0]
current_w = robot_velocities[:, 5]

# 2. 更新历史（移动窗口）
action_history[:, 2] = action_history[:, 0]  # t-2 = 旧的 t-1
action_history[:, 3] = action_history[:, 1]  # t-2 = 旧的 t-1
action_history[:, 0] = prev_measured_v       # t-1 = 上一时刻的测量值
action_history[:, 1] = prev_measured_w       # t-1 = 上一时刻的测量值

# 3. 保存当前测量值供下次使用
prev_measured_v = current_v
prev_measured_w = current_w
```

### 重要说明

- **测量值 vs 命令值**: Action History 存储的是**历史测量速度**，不是历史命令动作。这与 Base Velocity 保持一致，都是实际测量的速度值。
- **用途**: 提供速度的时间序列信息，帮助 RL 模型理解机器人的运动趋势和加速度信息。

---

## 完整观察向量示例

假设某个时刻的完整观察向量为：

```python
obs = [
    # LiDAR (36维)
    0.8, 0.9, 1.0, 1.0, 1.0, 0.7, 0.5, 0.3, 0.2, 0.1,  # 前方和左侧
    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,  # 后方
    1.0, 1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3,  # 右侧
    0.2, 0.1, 0.0, 0.1, 0.2, 0.3,                       # 回到前方
    
    # User Input (2维)
    0.707, 0.707,  # 目标在左前方 45°
    
    # Base Velocity (2维)
    0.6, 0.3,  # 当前以 0.3 m/s 前进，0.3 rad/s 逆时针旋转
    
    # Action History (4维)
    0.5, 0.2,  # t-1: 0.25 m/s 前进，0.2 rad/s 逆时针
    0.4, 0.1   # t-2: 0.2 m/s 前进，0.1 rad/s 逆时针
]
```

---

## 总结

| 组件 | 维度 | 取值范围 | 物理含义 | 归一化基准 |
|------|------|---------|---------|-----------|
| **LiDAR** | 36 | [0, 1] | 36 个方向的障碍物距离 | `max_range = 3.0 m` |
| **User Input** | 2 | [-1, 1], 模长=1 | 目标方向单位向量 | 单位向量 |
| **Base Velocity** | 2 | [-1, 1] | 当前测量速度 (v, w) | `max_v=0.5 m/s`, `max_w=1.0 rad/s` |
| **Action History** | 4 | [-1, 1] | 历史测量速度 (v_t-1, w_t-1, v_t-2, w_t-2) | 同上 |
| **总计** | **44** | - | - | - |

---

## 参考配置

相关配置参数位于 `cfg/task/WaffleDrive.yaml`：

```yaml
robot_limits:
  max_v: 0.5  # 线速度上限 m/s
  max_w: 1.0  # 角速度上限 rad/s

lidar:
  num_rays: 36
  max_range: 3.0  # 米
```

---

## 代码实现位置

- **观察组装**: `envs/observations.py` - `assemble_observations()`
- **Base Velocity 提取**: `envs/observations.py` - `get_base_velocity_from_tensor()`
- **User Input 计算**: `envs/user_intent.py` - `compute_user_intent_torch()`
- **LiDAR 计算**: `scripts/2_RL_stage1_test.py` - `compute_lidar_ranges()`（测试脚本）或 `envs/igym_waffle_env.py` - `_compute_lidar_ranges()`（RL 环境）
