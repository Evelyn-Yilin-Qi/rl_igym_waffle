# 3_RL_stage1_test.py 实现计划（最终版）

## 核心策略

**在 `2_RL_stage1_test.py` 的基础上直接添加 RL 训练组件，不使用 `VecEnvRLGames`。**

这样可以：
1. ✅ 避免所有导入顺序问题（`VecEnvRLGames` 会触发 `domain_randomization` 的过早导入）
2. ✅ 保留已验证可运行的环境创建逻辑
3. ✅ 只需替换 PID 控制为 RL 策略，添加训练组件

---

## 一、核心变化对比

| 模块 | 2_RL_stage1_test.py | 3_RL_stage1_test.py |
|------|---------------------|---------------------|
| **环境创建** | `World` + `ArticulationView` + `SceneManager` | ✅ **保持不变** |
| **观察计算** | 手动计算 `assemble_observations()` | ✅ **保持不变** |
| **控制方式** | PID 控制（`k_v`, `k_yaw`） | ❌ **替换为 RL 策略输出** |
| **动作应用** | `DifferentialController` | ✅ **保持不变**（但输入来自 RL） |
| **重置管理** | 简单重置逻辑 | ✅ **添加 active_mask + cooldown（10步）** |
| **训练组件** | 无 | ✅ **新增：RL 模型、PPO Agent、Memory、Tensorboard** |
| **奖励计算** | 无 | ✅ **新增：RewardCalculator** |

---

## 二、实现步骤

### 步骤 1：保留环境创建逻辑（完全不变）

```python
# ✅ 保留：World + ArticulationView 环境创建
world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=render_dt)
PhysicsContext().substeps = 8
world.scene.add_default_ground_plane()

# ✅ 保留：加载机器人
for i in range(num_envs):
    tb3_root = f"/World/envs/env_{i}/TB3"
    add_reference_to_stage(usd_path=TB3_USD, prim_path=tb3_root)

# ✅ 保留：创建 ArticulationView
robots = ArticulationView(...)

# ✅ 保留：创建 SceneManager
scene_manager = SceneManager(...)
```

### 步骤 2：保留观察计算逻辑（完全不变）

```python
# ✅ 保留：手动计算 LiDAR
lidar_ranges = compute_lidar_ranges(...)

# ✅ 保留：手动组装观察
obs = assemble_observations(
    robot_positions=pos,
    robot_orientations=rot,
    goal_positions=goal_pos,
    env_origins=env_origins,
    lidar_ranges=lidar_ranges,
    base_vel=base_vel_np,
    action_history=action_history,
    max_v=max_v,
    max_w=max_w,
    lidar_max_range=lidar_max_range,
    user_input=user_intent_np
)
```

### 步骤 3：添加 RL 训练组件

```python
# ✅ 新增：导入 RL 组件
from models.cnn_lstm_policy_base import WaffleCNNLSTMPolicyBase
from agents.ppo_agent_base import PPOAgentBase
from skrl.memories.torch import RandomMemory
from gym import spaces
from torch.utils.tensorboard import SummaryWriter
from envs.rewards import RewardCalculator

# ✅ 新增：定义观察和动作空间
observation_space = spaces.Box(
    low=-np.inf, high=np.inf,
    shape=(44,), dtype=np.float32
)
action_space = spaces.Box(
    low=-1.0, high=1.0,
    shape=(2,), dtype=np.float32
)

# ✅ 新增：创建 RL 模型
device = "cuda:0" if torch.cuda.is_available() else "cpu"
models = {
    "policy": WaffleCNNLSTMPolicyBase(...),
    "value": WaffleCNNLSTMPolicyBase(...)
}

# ✅ 新增：创建 Memory
rollouts = 8  # 从配置读取
memory_size = rollouts * num_envs  # 8 * 4 = 32
memory = RandomMemory(
    memory_size=memory_size,
    num_envs=num_envs,
    device=device,
    replacement=True
)

# ✅ 新增：创建 PPO Agent
agent = PPOAgentBase(
    models=models,
    memory=memory,
    observation_space=observation_space,
    action_space=action_space,
    device=device,
    cfg=full_cfg  # 包含 task 和 train 的完整配置
)

# ✅ 新增：创建奖励计算器
reward_calculator = RewardCalculator(
    num_envs=num_envs,
    device=device,
    ra=ra, rl=rl, rh=rh, phi_thresh=phi_thresh,
    ras=ras, rc=rc, rcrit=rcrit, rcol=rcol,
    d_col=d_col, d_crit=d_crit,
    max_v=max_v, max_w=max_w
)

# ✅ 新增：创建 Tensorboard
writer = SummaryWriter(log_dir="runs/stage1_test")
```

### 步骤 4：替换控制逻辑（PID → RL）

```python
# ❌ 移除：PID 控制
# v_cmd = np.clip(k_v * dist, 0.0, v_max_cmd)
# w_cmd = np.clip(k_yaw * yaw_err, -max_w, max_w)

# ✅ 新增：RL 策略推理
actions, _, _ = agent.act(obs_torch, step, TOTAL_STEPS)  # [num_envs, 2], 归一化 [-1, 1]
actions_np = actions.cpu().numpy()

# ✅ 新增：冷却期处理（强制 inactive 环境的动作为 0）
actions_clamped = actions_np.copy()
actions_clamped[~active_mask.cpu().numpy()] = 0.0

# ✅ 新增：反归一化动作（从 [-1, 1] 到 [v, w]）
v_cmd = actions_clamped[:, 0] * max_v  # 线速度
w_cmd = actions_clamped[:, 1] * max_w  # 角速度

# ✅ 保留：转换为轮速（使用 DifferentialController）
targets = np.zeros((num_envs, robots.num_dof), dtype=np.float32)
for i in range(num_envs):
    action = diff_ctrl.forward(command=np.array([float(v_cmd[i]), float(w_cmd[i])], dtype=np.float32))
    targets[i, left_idx] = float(action.joint_velocities[0])
    targets[i, right_idx] = float(action.joint_velocities[1])

robots.set_joint_velocity_targets(targets)
world.step(render=True)
```

### 步骤 5：添加奖励计算

```python
# ✅ 新增：计算奖励
rewards_torch = reward_calculator.compute_rewards(
    robot_positions=robot_pos_torch,
    robot_orientations=robot_rot_torch,
    robot_velocities=robot_velocities_torch,
    goal_positions=goal_pos_torch,
    env_origins=env_origins_torch,
    lidar_ranges=torch.from_numpy(lidar_ranges).float(),
    action_history=torch.from_numpy(action_history).float(),
    max_v=max_v,
    max_w=max_w,
    lidar_max_range=lidar_max_range
)
rewards = rewards_torch.cpu().numpy()
```

### 步骤 6：添加训练循环逻辑

```python
# ✅ 新增：active_mask + cooldown 机制
active_mask = torch.ones(num_envs, dtype=torch.bool, device=device)
cooldown_steps = torch.zeros(num_envs, dtype=torch.int32, device=device)
COOLDOWN_STEPS = 10

# ✅ 新增：统计信息
episode_rewards = torch.zeros(num_envs, device=device)
episode_lengths = torch.zeros(num_envs, dtype=torch.int32, device=device)

# ✅ 主训练循环
for step in range(TOTAL_STEPS):
    # 1. RL 策略推理
    actions, _, _ = agent.act(obs_torch, step, TOTAL_STEPS)
    
    # 2. 冷却期处理
    actions_clamped = actions_np.copy()
    actions_clamped[~active_mask.cpu().numpy()] = 0.0
    
    # 3. 反归一化并应用到机器人
    v_cmd = actions_clamped[:, 0] * max_v
    w_cmd = actions_clamped[:, 1] * max_w
    # ... 转换为轮速并应用 ...
    
    # 4. 计算奖励
    rewards = reward_calculator.compute_rewards(...)
    
    # 5. 检查重置条件
    to_reset = reset_timeout | reached | collision
    dones = to_reset.astype(np.float32)
    
    # 6. 处理重置和冷却期
    reset_env_ids = np.nonzero(to_reset)[0]
    if len(reset_env_ids) > 0:
        active_mask[reset_env_ids] = False
        cooldown_steps[reset_env_ids] = COOLDOWN_STEPS
        # ... 重置逻辑 ...
    
    # 更新冷却计数器
    cooldown_steps[~active_mask] -= 1
    cooldown_finished = (cooldown_steps <= 0) & (~active_mask)
    if cooldown_finished.any():
        active_mask[cooldown_finished] = True
    
    # 7. 获取下一步观察
    next_obs = assemble_observations(...)
    
    # 8. 记录经验到 memory
    rewards_masked = rewards.copy()
    rewards_masked[~active_mask.cpu().numpy()] = 0.0
    dones_masked = dones.copy()
    dones_masked[~active_mask.cpu().numpy()] = 0.0
    
    agent.record_transition(
        states=obs_torch,
        actions=actions,
        rewards=torch.from_numpy(rewards_masked).float().to(device),
        next_states=next_obs_torch,
        terminated=torch.from_numpy(dones_masked).float().to(device),
        truncated=torch.zeros_like(dones_torch, dtype=torch.bool).to(device)
    )
    
    # 9. 每 rollouts 步进行一次 PPO 更新
    if (step + 1) % rollouts == 0:
        agent.update(step, TOTAL_STEPS)
        
        # 记录到 Tensorboard
        if active_mask.any():
            avg_reward = rewards[active_mask.cpu().numpy()].mean()
            writer.add_scalar("Reward/Average", avg_reward, step)
            avg_episode_reward = episode_rewards[active_mask].mean().item()
            writer.add_scalar("Episode/Reward", avg_episode_reward, step)
            avg_episode_length = episode_lengths[active_mask].float().mean().item()
            writer.add_scalar("Episode/Length", avg_episode_length, step)
        
        reset_rate = dones.mean()
        writer.add_scalar("Episode/ResetRate", reset_rate, step)
        active_rate = active_mask.float().mean().item()
        writer.add_scalar("Environment/ActiveRate", active_rate, step)
    
    # 10. 定期保存 Checkpoint
    if (step + 1) % CHECKPOINT_INTERVAL == 0:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_{step+1}.pt")
        agent.save(checkpoint_path)
    
    # 11. 更新观察（为下一步准备）
    obs_torch = next_obs_torch
```

---

## 三、关键实现细节

### 3.1 动作处理流程

```
RL 策略输出 (归一化 [-1, 1])
    ↓
冷却期处理（inactive 环境 → 0）
    ↓
反归一化（× max_v, × max_w）
    ↓
DifferentialController（转换为轮速）
    ↓
应用到机器人
```

### 3.2 active_mask + cooldown 机制

- **重置时**：`active_mask[env_id] = False`，`cooldown_steps[env_id] = 10`
- **冷却期内**：
  - `actions[~active_mask] = 0.0`（强制不动）
  - `rewards[~active_mask] = 0.0`（不贡献奖励）
  - `dones[~active_mask] = 0.0`（不触发 RNN 状态清零）
- **冷却期结束**：`active_mask[cooldown_finished] = True`

### 3.3 经验记录策略

- 记录所有环境的转换，但：
  - `rewards[~active_mask] = 0.0`（不贡献奖励）
  - `dones[~active_mask] = 0.0`（不触发 RNN 状态清零）
- 这样 inactive 环境的转换不会影响训练

---

## 四、与 2_RL_stage1_test.py 的差异总结

### 4.1 完全保留的内容

- ✅ `World` + `ArticulationView` 环境创建
- ✅ `SceneManager` 场景管理
- ✅ `compute_lidar_ranges()` LiDAR 计算
- ✅ `assemble_observations()` 观察组装
- ✅ `DifferentialController` 差速控制
- ✅ 重置逻辑（超时/到达目标/碰撞）
- ✅ 观察输出（每隔1秒输出 env0）
- ✅ 调试输出（每0.5秒输出 env0 状态）

### 4.2 替换的内容

- ❌ PID 控制（`k_v`, `k_yaw`）→ ✅ RL 策略输出

### 4.3 新增的内容

- ✅ RL 模型（`WaffleCNNLSTMPolicyBase`）
- ✅ PPO Agent（`PPOAgentBase`）
- ✅ Memory（`RandomMemory`）
- ✅ 奖励计算器（`RewardCalculator`）
- ✅ `active_mask` + `cooldown` 机制
- ✅ Tensorboard 记录
- ✅ Checkpoint 保存
- ✅ 训练循环（记录经验、更新策略）

---

## 五、优势

1. **无导入顺序问题**：不使用 `VecEnvRLGames`，避免 `domain_randomization` 的过早导入
2. **已验证可运行**：基于 `2_RL_stage1_test.py`，环境创建逻辑已验证
3. **最小改动**：只需替换控制逻辑，添加训练组件
4. **灵活可控**：完全控制环境创建、观察计算、奖励计算流程

---

## 六、实现检查清单

- [ ] 环境创建成功（4 个 empty 场景）
- [ ] 观察形状正确（[4, 44]）
- [ ] 动作形状正确（[4, 2]），范围 [-1, 1]
- [ ] RL 策略推理正常（无 NaN/Inf）
- [ ] 环境步进正常（机器人移动）
- [ ] 奖励计算正常（有正值和负值）
- [ ] 重置机制正常（超时/到达目标/碰撞）
- [ ] active_mask 机制正常（重置后冷却 10 步）
- [ ] 经验记录正常（只记录 active 环境）
- [ ] PPO 更新正常（每 8 步更新一次）
- [ ] Tensorboard 记录正常（有数据输出）
- [ ] Checkpoint 保存正常（文件生成）

---

## 总结

核心策略：**在 `2_RL_stage1_test.py` 的基础上直接添加 RL 训练组件，不使用 `VecEnvRLGames`。**

关键变化：
1. **保留**：所有环境创建和观察计算逻辑
2. **替换**：PID 控制 → RL 策略输出
3. **新增**：RL 模型、Agent、Memory、奖励计算器、训练循环、Tensorboard、Checkpoint

这样避免了所有导入顺序问题，同时保持了代码的简洁和可控性。
