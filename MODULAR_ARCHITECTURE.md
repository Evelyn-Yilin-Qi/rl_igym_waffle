
## 目录结构

```
rl_igym_waffle/
├── core_network/        ❗❗❗如果后续要用别的架构就要改这里   # 核心网络架构模块
│   ├── __init__.py      # 工厂函数：create_policy(), create_value()
│   ├── base.py          # 基础接口：BasePolicy, BaseValue
│   ├── simple_fc.py     # 简单全连接架构（LiDAR分支 + 状态分支）
│   └── essay_base.py    # 论文架构（1D Conv + LSTM）
│    （❗或许我们还有很多别的架构）
│
├── algorithms/          # RL算法模块
│   ├── __init__.py      # 工厂函数：create_algorithm()
│   ├── base.py          # 算法基础接口：BaseRLAlgorithm
│   └── ppo.py           # PPO算法实现
│
├── rewards/             ❗❗❗如果要改奖励函数的参数请动这里 # 奖励函数模块
│   ├── __init__.py      # 导出所有奖励组件
│   ├── base.py          # 奖励组件基类：BaseRewardComponent
│   ├── composer.py      # 奖励组合器：RewardComposer（统一管理）  ❗组装所有enable为 Ture的奖励函数
│   ├── obstacle.py      # 障碍物惩罚
│   ├── heading.py       # 航向惩罚
│   ├── smoothness.py    # 动作平滑惩罚
│   ├── static.py        # 静态惩罚（静止不动）
│   └── centrifugal.py   # 离心力惩罚（可选）❗暂时没用上
│
├── envs/                # 环境观察模块
│   ├── __init__.py
│   ├── observations.py  # 观察向量组装：LiDAR、碰撞检测、44维观察
│   └── user_intent.py   # 用户意图计算（目标方向）
│
├── sim/                 # 仿真环境模块
│   ├── robot/           # 机器人相关
│   │   ├── tb3_config.py    # TB3配置（速度限制、尺寸等）
│   │   └── ...
│   └── scenes/          # 场景管理
│       ├── scene_manager.py  # 场景管理器（创建、重置障碍物）
│       ├── scene_base.py     # 场景基类和常量
│       └── ...
│
├── process_settings/    # 训练流程设置模块
│   ├── __init__.py
│   └── env_setup.py     # 环境初始化类：EnvironmentSetup
│
├── sft/                 ❗❗❗SFT相关逻辑都在这里
│   ├── rule_controller.py  # 规则控制器：监督label怎么生成
│   ├── supervised.py       # 监督训练：MSE训练策略网络   要是觉得需要改就动这里
│   ├── logging.py          # SFT/RL打印工具 正式的Stage里面就不print了
│   └── config.py           # SFT相关配置读取工具（轻量）
│
├── config/              ❗❗❗ 改要使用的re function； network；RL 策略等# 配置模块（YAML格式）  
│   ├── network_config.yaml    # 网络架构配置（支持enabled切换）
│   ├── algorithm_config.yaml  # RL算法配置（支持enabled切换）
│   └── reward_config.yaml     # 奖励函数配置（支持enabled/weight）
│
├── utils/               # 工具函数模块
│   └── config_utils.py  # 配置读取工具：get_enabled_component()
│
├── scripts/              # 训练脚本   （有关SFT以及RL的一些参数代码的命名我都改了，注释里面写了原名和现名，改是为了统一加上好理解
│   ├── stage1_rl_sft.py       # ❗❗❗❗❗Stage1 SFT + RL 之后修改stage1主要用这个  只有空场景 目前env中写死了16个
│   │                             只能说跑通了,效果完全没验证过                         env改的时候建议【4，16，32，64这种的】
│   ├── stage2_rl_sft.py       # ❗❗❗❗❗Stage2 SFT + RL 之后修改stage2主要用这个  四个场景 目前env中写死了16个
│   │                             只能说跑通了,效果完全没验证过                         env改的时候建议【4，16，32，64这种的】
│   ├── sft_rl_tb3_V3_modular.py ❗❗❗❗❗ 如果要做任何运行测试，可以在这里做，但只运行一个env，四个场景都可选择可以在代码中切换，main注释写清楚了
│   │                           
│   ├── sft_rl_tb3_V3.py       # 冯的版本对齐到V3  这个可以运行，但是只是box单一场景 (test)
│   ├── sft_rl_tb3.py          # 冯原始的脚本  (test)
│   └── PPO_ckpt_test.py       # checkpoint打包/测试脚本  反正现在没有可用的模型我就没动，这个代码没有做适配
│
│   # 旧脚本（别用了）
│   ├── recent_old_version/stage1_sft_rl.py
│   ├── recent_old_version/stage2_sft_rl.py
│   └── test/stage1_rl.py, test/stage2_rl.py, test/*.py
│
└── cfg/                 # 别再用了，只是为了几个老脚本留着了,新脚本都没引用 
    ├── WaffleDrive.yaml  # 
```

---


这个工程在做的事情，在isaac gym 仿真平台中使用不同的场景scene，然后并行加载多个env环境做TB3小车的强化学习训练，让小车学会跟随user intent行走的同时学会丝滑转向避障等。
训练在stage1的时候只会加载一个类型的scene，即空场景, 使用这一个场景生成多个env环境
训练在stage2的时候会加载四个类型的scene，除了空场景之外，还有带有障碍物的场景，使用四个场景生成env环境训练

# 模块化架构设计方案

## 📋 目录

- [项目概述](#项目概述)
- [目录结构](#目录结构)
- [核心模块详解](#核心模块详解)
  - [core_network/ - 核心网络架构模块](#core_network---核心网络架构模块)
  - [algorithms/ - RL算法模块](#algorithms---rl算法模块)
  - [rewards/ - 奖励函数模块](#rewards---奖励函数模块)
  - [envs/ - 环境观察模块](#envs---环境观察模块)
  - [sim/ - 仿真环境模块](#sim---仿真环境模块)
  - [process_settings/ - 训练流程设置模块](#process_settings---训练流程设置模块)
  - [sft/ - SFT模块](#sft---sft模块)
  - [config/ - 配置模块](#config---配置模块)
  - [utils/ - 工具函数模块](#utils---工具函数模块)
- [核心接口设计](#核心接口设计)
- [配置文件说明](#配置文件说明)
- [使用示例](#使用示例)
- [当前开发必看入口（最重要）](#当前开发必看入口最重要)

---

## 项目概述

本项目是一个基于 **Isaac Sim** 的 **TurtleBot3 (TB3)** 机器人强化学习训练框架，采用模块化设计，支持：

- ✅ **多种网络架构**：`simple_fc`（简单全连接）、`essay_base`（1D Conv + LSTM）
- ✅ **多种RL算法**：PPO（已实现），可扩展 SAC、TD3 等
- ✅ **模块化奖励函数**：障碍物惩罚、航向惩罚、平滑度惩罚、静态惩罚、离心力惩罚等
- ✅ **多场景训练**：EMPTY、BOX、CYLINDER、DOOR 等场景
- ✅ **配置驱动**：通过 YAML 配置文件切换架构、算法、奖励函数，无需修改代码

---


## 核心模块详解

### `core_network/` - 核心网络架构模块

**职责**：定义和管理策略网络（Policy）和价值网络（Value）的架构。

**核心功能**：
- 提供统一的网络接口（`BasePolicy`、`BaseValue`）
- 实现多种网络架构（`simple_fc`、`essay_base`）
- 通过工厂函数创建网络实例

**文件说明**：
- **`base.py`**：定义 `BasePolicy` 和 `BaseValue` 基类
  - `BasePolicy.forward()` 返回 `(mean, log_std)` 用于连续动作空间
  - `BaseValue.forward()` 返回状态价值
- **`simple_fc.py`**：简单全连接架构
  - LiDAR分支：36维 → FC → 64维
  - 状态分支：8维 → FC → 64维
  - 融合后 → FC → 输出
- **`essay_base.py`**：论文架构（1D Conv + LSTM）
  - LiDAR分支：1D Conv(36→16→7) → FC0(28→64) → Tanh
  - 状态特征：8维
  - 融合：Concat(64+8=72) → LSTM(72→64) → FC1(64→64) → Tanh
  - 输出：Policy(mean, log_std) 或 Value

**使用方式**：
```python
from core_network import create_policy, create_value

# 从配置读取网络类型和参数
network_type, network_params = get_enabled_component(network_cfg.networks)
network_params['obs_dim'] = 44
network_params['act_dim'] = 2
network_params['lidar_dim'] = 36
network_params['state_dim'] = 8

# 创建网络
policy = create_policy(network_type, **network_params)
value = create_value(network_type, **network_params)
```

---

### `algorithms/` - RL算法模块

**职责**：实现强化学习算法，管理策略更新和经验回放。

**核心功能**：
- 实现PPO算法（Proximal Policy Optimization）
- 提供统一的算法接口（`BaseRLAlgorithm`）
- 支持模型保存和加载

**文件说明**：
- **`base.py`**：定义 `BaseRLAlgorithm` 基类
  - `select_action()`：选择动作
  - `store_transition()`：存储经验
  - `update()`：更新策略
  - `save()` / `load()`：模型保存/加载
- **`ppo.py`**：PPO算法实现
  - 使用GAE（Generalized Advantage Estimation）计算优势函数
  - 支持梯度裁剪、数值稳定性检查
  - 返回训练损失（actor_loss, critic_loss, entropy, approx_kl）用于TensorBoard

**使用方式**：
```python
from algorithms import create_algorithm

# 从配置读取算法类型和参数
algorithm_type, algorithm_params = get_enabled_component(algo_cfg.algorithms)

# 创建算法实例
rl_agent = create_algorithm(
    algorithm_type,
    policy=policy,
    value=value,
    config=algorithm_params,
    device=device
)

# 训练循环
action, log_prob = rl_agent.select_action(obs)
rl_agent.store_transition(obs, action, reward, next_obs, done, log_prob)
losses = rl_agent.update()  # 返回训练损失
```

---

### `rewards/` - 奖励函数模块

**职责**：定义和管理各种奖励/惩罚组件，统一计算总奖励。

**核心功能**：
- 模块化奖励组件（每个组件独立计算）
- 奖励组合器（`RewardComposer`）统一管理和组合
- 支持配置驱动的启用/禁用和权重调整

**文件说明**：
- **`base.py`**：定义 `BaseRewardComponent` 基类
  - `compute()` 方法返回 `(reward, info)`，其中 `info` 用于TensorBoard
- **`composer.py`**：`RewardComposer` 类
  - 自动加载配置中启用的奖励组件
  - 统一调用所有组件的 `compute()` 方法
  - 按权重组合总奖励
  - 收集所有组件的 `info` 用于日志记录
- **`obstacle.py`**：障碍物惩罚
  - 根据距离障碍物的距离给予惩罚（`rc`, `rcrit`）
  - 碰撞时给予高额惩罚（`rcol = -100.0`）
- **`heading.py`**：航向惩罚
  - 惩罚机器人朝向与目标方向的偏差
- **`smoothness.py`**：动作平滑惩罚
  - 惩罚动作变化过大（加速度惩罚）
- **`static.py`**：静态惩罚
  - 惩罚机器人静止不动
- **`centrifugal.py`**：离心力惩罚（可选）
  - 惩罚过大的离心力（可选：linear、threshold、quadratic模式）
- **`distance.py`**：距离奖励（负距离项）
- **`goal.py`**：到达奖励 + 超时惩罚

**使用方式**：
```python
from rewards import RewardComposer

# 从配置创建奖励组合器
reward_composer = RewardComposer(reward_cfg.rewards)

# 计算奖励（自动组合所有启用的组件）
reward, reward_info = reward_composer.compute(
    lidar_ranges=lidar_ranges,
    robot_positions=robot_positions,
    robot_orientations=robot_orientations,
    goal_positions=goal_positions,
    env_origins=env_origins,
    base_vel=base_vel,
    action_history=action_history,
    prev_action=prev_action,
    collision_flags=collision_flags,
    max_v=max_v,
    max_w=max_w
)

# reward_info 包含所有组件的详细信息，用于TensorBoard
```

---

### `envs/` - 环境观察模块

**职责**：处理观察空间的数据处理和组装。

**核心功能**：
- LiDAR数据处理和碰撞检测
- 组装44维观察向量
- 用户意图计算

**文件说明**：
- **`observations.py`**：
  - `compute_lidar_ranges()`：计算LiDAR距离数据（36维）
  - `check_obstacle_collision()`：检测碰撞（基于LiDAR数据）
  - `assemble_observations()`：组装44维观察向量
    - LiDAR(36) + UserIntent(2) + BaseVel(2) + ActionHistory(4) = 44维
- **`user_intent.py`**：
  - `compute_user_intent_torch()`：计算用户意图（目标方向向量，2维）
  - **UserIntent 统一口径（全工程）**：
    - `ux`：自车正前方为正方向（forward +X）
    - `uy`：自车左侧90度为正方向（left +Y）
    - 模型输入使用 `user_intent_ego`（自车坐标系），而非环境/世界坐标系向量

**使用方式**：
```python
from envs.observations import assemble_observations

# 组装观察向量
obs = assemble_observations(
    robot_positions=robot_positions,
    robot_orientations=robot_orientations,
    goal_positions=goal_positions,
    env_origins=env_origins,
    lidar_ranges=lidar_ranges,
    base_vel=base_vel,
    action_history=action_history,
    max_v=max_v,
    max_w=max_w,
    device=device
)
```

---

### `sim/` - 仿真环境模块

**职责**：管理Isaac Sim仿真环境，包括机器人、场景、物理引擎等。

**核心功能**：
- 机器人配置和初始化
- 场景管理（障碍物创建、重置）
- 物理引擎配置

**文件说明**：
- **`robot/tb3_config.py`**：
  - TB3机器人配置（速度限制、尺寸、USD路径等）
- **`scenes/scene_manager.py`**：`SceneManager` 类
  - `create_scene_obstacles()`：创建场景障碍物（BOX、CYLINDER、DOOR）
  - `reset_scene_obstacles()`：重置障碍物位置和可见性
  - `sample_goal_positions()`：采样目标位置
  - 支持场景类型：`EMPTY`、`BOX`、`CYLINDER`、`DOOR`

**使用方式**：
```python
from sim.scenes import SceneManager

# 创建场景管理器
scene_manager = SceneManager(
    num_envs=num_envs,
    env_size=env_cfg.env_size,
    env_origins=env_origins,
    stage=stage
)

# 创建场景障碍物
scene_manager.create_scene_obstacles(
    scene_types=scene_types,
    rng=rng,
    show_visual_walls_list=None  # None表示自动根据场景类型决定
)

# 重置场景
scene_manager.reset_scene_obstacles(
    env_ids=np.arange(num_envs),
    rng=rng
)
```

---

### `process_settings/` - 训练流程设置模块

**职责**：封装训练环境的初始化逻辑，减少代码重复。

**核心功能**：
- 统一的环境初始化流程
- 自动计算环境布局（接近正方形）
- 机器人、场景、目标的统一设置

**文件说明**：
- **`env_setup.py`**：`EnvironmentSetup` 类
  - `setup_all()`：静态方法，一键初始化所有环境
  - 负责：
    1. 初始化 World 和 Physics
    2. 计算环境布局（自动计算列数和行数，接近正方形）
    3. 加载和初始化机器人
    4. 创建差速控制器
    5. 创建场景管理器和障碍物
    6. 创建目标标记

**使用方式**：
```python
from process_settings.env_setup import EnvironmentSetup

# 一键初始化所有环境
env_setup = EnvironmentSetup.setup_all(
    env_cfg=env_cfg,
    num_envs=16,
    scene_types=["EMPTY"] * 16,  # 或混合场景
    simulation_app=simulation_app,
    rng=rng
)

# 获取初始化后的对象
world = env_setup.world
robot_view = env_setup.robot_view
controller = env_setup.controller
scene_manager = env_setup.scene_manager
goal_markers = env_setup.goal_markers
env_origins = env_setup.env_origins
```

---

### `sft/` - SFT模块

**职责**：把“监督阶段”从主训练脚本里拆出来，集中管理规则采样、监督训练和日志打印。

**你最常改的文件**：
- **`rule_controller.py`**  
  - 改“监督采样时机器人怎么走”的规则（避障、转向、速度分配）
  - 想改变 SFT label 的行为，就改这里
- **`supervised.py`**  
  - 改监督训练过程（loss/batch/epoch/训练细节）
  - 当前是对 policy mean 做 MSE 训练
- **`logging.py`**  
  - 改 SFT/RL阶段终端输出格式
- **`config.py`**  
  - 轻量配置读取工具（非主配置）

**与主脚本关系**：
- `scripts/stage1_rl_sft.py` / `scripts/stage2_rl_sft.py` 会调用这里的模块
- 训练主循环负责环境交互；`sft/` 负责监督阶段逻辑

---

### `config/` - 配置模块

**职责**：通过YAML配置文件管理所有模块的参数，支持配置驱动的切换。

**核心功能**：
- 网络架构配置（支持 `enabled` 切换）
- RL算法配置（支持 `enabled` 切换）
- 奖励函数配置（支持 `enabled` 和 `weight` 调整）
- 新脚本统一读取 `config/*`，**不再读取 `cfg/*`**

**文件说明**：
- **`network_config.yaml`**：
  ```yaml
  networks:
    components:
      - name: "essay_base"
        enabled: true
        params: {}
      - name: "simple_fc"
        enabled: false
        params:
          hidden_dim: 128
  ```
- **`algorithm_config.yaml`**：
  ```yaml
  algorithms:
    components:
      - name: "ppo"
        enabled: true
        params:
          lr: 3e-4
          gamma: 0.98
          ...
  ```
- **`reward_config.yaml`**：
  ```yaml
  rewards:
    components:
      - name: "obstacle"
        enabled: true
        weight: 1.0
        params:
          rcol: -100.0
          ...
  ```

**使用方式**：
```python
from utils.config_utils import get_enabled_component
from omegaconf import OmegaConf

# 加载配置
network_cfg = OmegaConf.load("config/network_config.yaml")

# 获取启用的组件
network_type, network_params = get_enabled_component(network_cfg.networks)
```

---

### `utils/` - 工具函数模块

**职责**：提供通用的工具函数。

**文件说明**：
- **`config_utils.py`**：
  - `get_enabled_component(config, component_type="components")`：从配置中提取启用的组件
  - **功能**：
    1. 从配置的 `components` 列表中找到 `enabled: true` 的组件
    2. 检查是否有且仅有一个启用的组件（防止配置错误）
    3. 返回组件的 `name` 和 `params`
  - **使用位置**：`scripts/stage1_rl_sft.py`、`scripts/stage2_rl_sft.py`、`scripts/sft_rl_tb3_V3.py`
  - **使用场景**：
    - 从 `network_config.yaml` 中提取启用的网络架构（如 `"essay_base"`）
    - 从 `algorithm_config.yaml` 中提取启用的算法（如 `"ppo"`）
  - **为什么需要这个函数**：
    - 配置文件使用 `components` 列表格式，需要遍历找到 `enabled: true` 的项
    - 统一处理配置错误（没有启用的组件、多个启用的组件）
    - 简化代码：训练脚本只需调用一次函数，无需手动遍历和检查

---

## 核心接口设计

### 1. 核心网络架构接口（`core_network/base.py`）

```python
class BasePolicy(nn.Module):
    """策略网络基类 - 统一接口：输入44维，输出2维"""
    def __init__(self, obs_dim=44, act_dim=2, **kwargs):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
    
    def forward(self, obs):
        """
        返回 (mean, log_std) 用于连续动作空间
        Args:
            obs: (batch, 44) 观察向量
        Returns:
            mean: (batch, 2) 动作均值
            log_std: (batch, 2) 动作对数标准差
        """
        raise NotImplementedError

class BaseValue(nn.Module):
    """价值网络基类"""
    def __init__(self, obs_dim=44, **kwargs):
        super().__init__()
        self.obs_dim = obs_dim
    
    def forward(self, obs):
        """
        返回状态价值
        Args:
            obs: (batch, 44) 观察向量
        Returns:
            value: (batch, 1) 状态价值
        """
        raise NotImplementedError
```

### 2. RL算法接口（`algorithms/base.py`）

```python
class BaseRLAlgorithm:
    """RL算法基类 - 统一接口"""
    def __init__(self, policy, value, config, device=None):
        self.policy = policy
        self.value = value
        self.config = config
        self.device = device
    
    def select_action(self, obs):
        """
        选择动作
        Returns:
            action: (batch, 2) 动作
            log_prob: (batch,) 动作对数概率
        """
        raise NotImplementedError
    
    def store_transition(self, obs, act, rew, next_obs, done, log_prob):
        """存储经验"""
        raise NotImplementedError
    
    def update(self):
        """
        更新策略
        Returns:
            dict: 训练损失信息（用于TensorBoard）
        """
        raise NotImplementedError
    
    def save(self, path):
        """保存模型"""
        raise NotImplementedError
    
    def load(self, path):
        """加载模型"""
        raise NotImplementedError
```

### 3. 奖励函数接口（`rewards/base.py`）

```python
class BaseRewardComponent:
    """奖励组件基类"""
    def __init__(self, config):
        self.config = config
        self.name = self.__class__.__name__
    
    def compute(self, **kwargs):
        """
        计算奖励分量
        Args:
            **kwargs: 环境状态信息（lidar_ranges, robot_positions, ...）
        Returns:
            reward: (num_envs,) 奖励值
            info: dict 额外信息（用于TensorBoard）
        """
        raise NotImplementedError
```

---

## 配置文件说明

### 为什么使用"组件+enabled"设计？

**问题背景**：
- 传统方式：直接在配置中写 `type: "essay_base"`，要切换架构需要修改配置
- 问题：切换架构时，需要删除旧配置、重写新配置，容易出错，也不方便对比不同架构的参数

**组件设计的优势**：
1. **保留所有选项**：所有架构/算法的配置都保留在文件中，方便对比和切换
2. **一键切换**：只需修改 `enabled: true/false`，无需删除或重写配置
3. **防止配置丢失**：切换架构时不会丢失之前的配置参数
4. **便于实验**：可以快速在不同架构/算法之间切换，进行对比实验

**使用示例**：
```python
# 从配置中提取启用的组件
network_type, network_params = get_enabled_component(network_cfg.networks)
# 如果 essay_base 的 enabled: true，则返回 ("essay_base", {})
# 如果 simple_fc 的 enabled: true，则返回 ("simple_fc", {"hidden_dim": 128})
```

### `config/network_config.yaml`

用于配置网络架构，支持通过 `enabled: true/false` 切换架构。

**设计说明**：
- 所有可用的网络架构都列在 `components` 列表中
- 每个架构都有 `name`（架构名称）、`enabled`（是否启用）、`params`（架构参数）
- **只能有一个架构的 `enabled: true`**，`get_enabled_component()` 会自动检查并返回启用的架构
- 要切换架构，只需将当前架构的 `enabled` 改为 `false`，目标架构的 `enabled` 改为 `true`

```yaml
networks:
  components:
    - name: "essay_base"  # 论文基础架构：1D Conv + LSTM
      enabled: true       # 当前启用这个架构
      params:
        # essay_base 架构不需要额外参数（所有维度都是固定的）
    
    - name: "simple_fc"  # 简单全连接架构
      enabled: false      # 当前未启用，但配置保留在这里
      params:
        hidden_dim: 128  # 隐藏层维度
```

**切换示例**：
```yaml
# 要切换到 simple_fc，只需修改：
    - name: "essay_base"
      enabled: false  # 改为 false
    - name: "simple_fc"
      enabled: true   # 改为 true
```

### `config/algorithm_config.yaml`

用于配置RL算法，支持通过 `enabled: true/false` 切换算法。

**设计说明**：
- 与网络配置类似，所有可用的算法都列在 `components` 列表中
- 每个算法都有 `name`、`enabled`、`params`
- **只能有一个算法的 `enabled: true`**

```yaml
algorithms:
  components:
    - name: "ppo"  # Proximal Policy Optimization
      enabled: true
      params:
        lr: 3e-4
        gamma: 0.98
        lamda: 0.90
        clip_eps: 0.2
        k_epochs: 3
        batch_size: 64
        entropy_coef: 0.01
```

### `config/reward_config.yaml`

用于配置奖励函数组件，支持 `enabled` 和 `weight` 调整。

```yaml
rewards:
  components:
    - name: "obstacle"
      enabled: true
      weight: 1.0
      params:
        dcol: 0.2
        dcrit: 0.5
        rc: -1.0      # 临界距离惩罚
        rcrit: -1.0   # 临界阈值惩罚
        rcol: -100.0  # 碰撞惩罚
    
    - name: "heading"
      enabled: true
      weight: 1.0
      params:
        phi_thresh: 0
        rh: -2.8
        rl: -1
    
    - name: "smoothness"
      enabled: true
      weight: 1.0
      params:
        ras1: -0.02
        ras2: -0.02
    
    - name: "static"
      enabled: true
      weight: 1.0
      params:
        v_thresh: 0.1
        w_thresh: 0.2
        penalty_value: -1.0
    
    - name: "centrifugal"
      enabled: false  # 暂时不启用
      weight: 1.0
      params:
        mass: 1.373
        penalty_type: "linear"
        scale: 1.0
        threshold: 0.5
```

---

## 当前开发必看入口（最重要）

下面是“改需求时最该动的地方”，按优先级排序：

1. **SFT规则逻辑改这里**：`sft/rule_controller.py`
   - `compute_velocity_commands()` 是规则控制器输出 label 的核心。
   - 想改“监督阶段机器人怎么走、如何避障、如何转向”就改这里。

2. **SFT监督训练改这里**：`sft/supervised.py`
   - `train_supervised_policy()` 定义了监督训练方式（当前是 MSE 训练 policy mean）。
   - 想改 batch、loss、优化轮数策略就改这里。

3. **主训练流程入口**：
   - `scripts/stage1_rl_sft.py`：16 env + EMPTY 场景，先 SFT 再 PPO
   - `scripts/stage2_rl_sft.py`：16 env + 4场景混合，先 SFT 再 PPO，默认从 stage1 模型继续训

4. **网络架构切换（不改脚本）**：`config/network_config.yaml`
   - 通过 `enabled: true/false` 选择当前架构（如 `simple_fc` / `simple_fc_sft` / `essay_base`）。

5. **奖励组装开关（不改脚本）**：`config/reward_config.yaml`
   - 通过每个 reward component 的 `enabled` 控制是否参与总奖励组合。
   - 通过 `weight` 做权重缩放。

6. **奖励公式参数改这里**：
   - 先改 `config/reward_config.yaml` 的 `params`（优先）
   - 若需要改公式本身，再改 `rewards/*.py` 对应组件实现
   - 例如：障碍物惩罚改 `rewards/obstacle.py`，目标奖励改 `rewards/goal.py`

7. **checkpoint 产物位置与命名**
   - 各训练脚本里的 `checkpoint_dir` 决定输出目录（通常在 `checkpoints/` 下）
   - 当前 SFT/RL脚本保存文件名带时间戳，精确到分钟（`YYYYMMDD_HHMM`）
   - 常见命名：`ppo_supervised_final_*.pth`、`ppo_epoch_XXX_*.pth`、`ppo_final_*.pth`

8. **`cfg/` 目录状态**
   - `cfg/*` 是 legacy 兼容目录，保留给老脚本。
   - 新增脚本或新需求开发，不应再依赖 `cfg/*`。

---

## 使用示例

### 完整训练流程（`scripts/stage1_rl_sft.py`）

```python
# 1. 加载配置
network_cfg = load_config("config/network_config.yaml")
algo_cfg = load_config("config/algorithm_config.yaml")
reward_cfg = load_config("config/reward_config.yaml")

# 2. 脚本内定义环境参数（新流程不依赖 cfg/WaffleDrive.yaml）
num_envs = 16
scene_types = ["EMPTY"] * 16  # stage1
env_cfg = OmegaConf.create({"sim": {"dt": 0.025}})

# 3. 初始化环境
env_setup = EnvironmentSetup(...)

# 4. 创建网络
network_type, network_params = get_enabled_component(network_cfg.networks)
network_params.update({
    'obs_dim': 44,
    'act_dim': 2,
    'lidar_dim': 36,
    'state_dim': 8
})
policy = create_policy(network_type, **network_params)
value = create_value(network_type, **network_params)

# 5. 创建RL算法
algorithm_type, algorithm_params = get_enabled_component(algo_cfg.algorithms)
rl_agent = create_algorithm(
    algorithm_type,
    policy=policy,
    value=value,
    config=algorithm_params,
    device=device
)

# 6. 创建奖励组合器
reward_composer = RewardComposer(reward_cfg.rewards)

# 7. 训练循环（SFT阶段收集 -> 监督训练 -> PPO阶段更新）
for step in range(max_steps):
    # 获取观察
    obs = assemble_observations(...)
    
    # 选择动作
    action, log_prob = rl_agent.select_action(obs)
    
    # 执行动作，获取下一状态
    # ... (Isaac Sim环境交互)
    
    # 计算奖励（SFT阶段也会算，但SFT loss不使用reward）
    reward, reward_info = reward_composer.compute(...)
    
    # 存储经验
    rl_agent.store_transition(obs, action, reward, next_obs, done, log_prob)
    
    # PPO阶段更新策略
    if step % update_freq == 0:
        losses = rl_agent.update()
        # 记录TensorBoard
        log_to_tensorboard(reward_info, losses)
```

---

## 模块优势

1. **高内聚低耦合**：各模块独立，通过标准接口交互
2. **配置驱动**：无需改代码，修改YAML即可切换架构、算法、奖励函数
3. **易于扩展**：
   - 新增网络架构：实现 `BasePolicy`/`BaseValue`，在配置中添加
   - 新增RL算法：实现 `BaseRLAlgorithm`，在配置中添加
   - 新增奖励组件：实现 `BaseRewardComponent`，在配置中添加
4. **代码复用**：`EnvironmentSetup` 统一环境初始化，减少重复代码
5. **便于协作**：训练入口、配置入口、奖励入口都已集中，可快速定位修改点

---

## 总结

本工程采用模块化设计，各模块职责清晰，通过配置驱动实现灵活切换。新增功能只需实现对应接口并在配置中添加，无需修改现有代码，大大提高了代码的可维护性和可扩展性。
