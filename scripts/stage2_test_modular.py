"""
RL Stage 2 PPO训练脚本 - 模块化版本
基于Stage1训练的模型继续训练，使用4种场景（EMPTY, CYLINDER, DOOR, BOX）
使用模块化架构：core_network, algorithms, rewards
"""
import os
import sys
import time
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
from omni.isaac.kit import SimulationApp

# 配置项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ==================== 设备配置 ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=== 设备配置 ===")
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU可用内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("未检测到GPU，使用CPU运行")

# 初始化Isaac Sim
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.physics_context import PhysicsContext
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.objects import VisualSphere
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController

# 导入自定义模块
from sim.robot import (
    TB3_USD, WHEEL_RADIUS, WHEEL_BASE,
    apply_massapi_all_tb3, configure_wheel_joints
)
from sim.scenes import (
    SCENE_EMPTY, SCENE_BOX, SCENE_CYLINDER, SCENE_DOOR,
    SceneManager, yaw_from_quat_wxyz, quat_wxyz_from_yaw, wrap_to_pi
)
from envs.observations import (
    assemble_observations, 
    get_base_velocity_from_tensor,
    compute_lidar_ranges,
    check_obstacle_collision,
    get_obs
)
from envs.user_intent import compute_user_intent_torch

# 导入模块化组件
from core_network import create_policy, create_value
from algorithms import create_algorithm
from rewards import RewardComposer

# ==================== 工具函数 ====================
def load_config(cfg_path):
    """加载配置文件"""
    candidates = []
    if os.path.isabs(cfg_path):
        candidates.append(cfg_path)
    else:
        candidates.extend([
            os.path.join(PROJECT_ROOT, cfg_path),
            os.path.abspath(cfg_path),
            os.path.join(os.getcwd(), cfg_path),
        ])

    found = None
    for path in candidates:
        if os.path.exists(path):
            found = os.path.abspath(path)
            break

    if found is None:
        raise FileNotFoundError(f"Config file not found. Tried: {candidates}")

    cfg = OmegaConf.load(found)
    return cfg


def find_latest_checkpoint(checkpoint_dir, pattern="ppo_*.pth"):
    """
    查找checkpoint目录中最新的模型文件
    
    Args:
        checkpoint_dir: checkpoint目录路径
        pattern: 文件匹配模式，默认"ppo_*.pth"
    
    Returns:
        最新checkpoint文件的完整路径，如果不存在则返回None
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    # 查找所有匹配的checkpoint文件
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, pattern))
    
    if not checkpoint_files:
        return None
    
    # 按修改时间排序，返回最新的
    latest = max(checkpoint_files, key=os.path.getmtime)
    return latest


def load_checkpoint(rl_agent, checkpoint_path=None, checkpoint_dir=None):
    """
    加载checkpoint模型
    
    Args:
        rl_agent: RL算法实例
        checkpoint_path: 指定的checkpoint文件路径（如果为None，则自动查找最新的）
        checkpoint_dir: checkpoint目录（仅在checkpoint_path为None时使用）
    
    Returns:
        是否成功加载
    """
    if checkpoint_path is None:
        if checkpoint_dir is None:
            # 默认查找stage1的checkpoint目录
            checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints", "ppo_stage1_test_modular")
        checkpoint_path = find_latest_checkpoint(checkpoint_dir)
    
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        print(f"[警告] 未找到checkpoint文件: {checkpoint_path}")
        print(f"[信息] 将从头开始训练")
        return False
    
    try:
        rl_agent.load(checkpoint_path)
        print(f"[成功] 已加载checkpoint: {checkpoint_path}")
        return True
    except Exception as e:
        print(f"[错误] 加载checkpoint失败: {e}")
        print(f"[信息] 将从头开始训练")
        return False


def save_model(rl_agent, checkpoint_dir, tag):
    """保存模型"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = os.path.join(checkpoint_dir, f"ppo_{tag}.pth")
    rl_agent.save(save_path)
    print(f"[Checkpoint] Saved => {save_path}")

# ==================== 主函数 ====================
def main(checkpoint_path=None):
    """
    主函数
    
    Args:
        checkpoint_path: 可选的checkpoint文件路径。如果为None，则自动查找最新的stage1模型
    """
    # 加载配置（模块化配置）
    env_cfg = load_config("cfg/WaffleDrive.yaml")  # 环境和物理配置
    network_cfg = load_config("config/network_config.yaml")  # 核心网络配置
    algo_cfg = load_config("config/algorithm_config.yaml")  # RL算法配置
    reward_cfg = load_config("config/reward_config.yaml")  # 奖励函数配置
    
    # 配置参数（从env_cfg读取）
    num_envs = 16  # Stage2：16个环境，每个场景各4个（EMPTY, CYLINDER, DOOR, BOX各4个）
    env_size = float(env_cfg.env.scene.env_size)
    env_gap = 2.0
    env_spacing = env_size + env_gap
    reset_dist = float(env_cfg.env.resetDist)
    wall_thickness = float(env_cfg.env.scene.wall_thickness)
    wall_height = float(env_cfg.env.scene.wall_height)
    show_visual_walls = bool(env_cfg.env.scene.show_visual_walls)
    TIMEOUT_SECONDS = 60.0
    
    # 物理和机器人参数
    physics_dt = float(env_cfg.sim.dt)
    render_dt = 1.0 / 30.0
    max_v = float(env_cfg.env.robot_limits.max_v)
    max_w = float(env_cfg.env.robot_limits.max_w)
    
    # LiDAR参数
    lidar_num_rays = int(env_cfg.env.lidar.num_rays)
    lidar_max_range = float(env_cfg.env.lidar.max_range)
    
    # 碰撞模型参数（轮椅胶囊形）
    DCOL = 0.2  # 碰撞阈值（20cm）
    DCRIT = 0.5  # 临界阈值（50cm）
    
    # 随机数初始化
    seed = 0
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # GPU随机数种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # 初始化仿真世界
    world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=render_dt)
    PhysicsContext().substeps = 8
    world.scene.add_default_ground_plane()
    stage = world.scene.stage
    
    # 环境原点 - 自动计算接近正方形的布局
    import math
    # 找到最接近 sqrt(num_envs) 的因数作为列数
    target_cols = int(math.sqrt(num_envs))
    num_cols = target_cols
    # 从目标值向下找，找到能整除 num_envs 的最大因数
    for cols in range(target_cols, 0, -1):
        if num_envs % cols == 0:
            num_cols = cols
            break
    num_rows = num_envs // num_cols
    print(f"\n=== 环境布局 ===")
    print(f"环境数量: {num_envs}")
    print(f"布局: {num_cols}列 × {num_rows}行")
    
    env_origins = np.zeros((num_envs, 3), dtype=np.float32)
    for i in range(num_envs):
        ix = i % num_cols      # X方向索引（列）
        iy = i // num_cols     # Y方向索引（行）
        env_origins[i, 0] = ix * env_spacing
        env_origins[i, 1] = iy * env_spacing
    
    # 加载机器人
    for i in range(num_envs):
        tb3_root = f"/World/envs/env_{i}/TB3"
        add_reference_to_stage(usd_path=TB3_USD, prim_path=tb3_root)
    
    # 等待场景稳定
    for _ in range(180):
        world.step(render=True)
    
    # 创建机器人视图
    robots = ArticulationView(
        prim_paths_expr="/World/envs/env_.*/TB3/a__namespace_base_footprint",
        name="tb3_view",
        reset_xform_properties=False,
    )
    world.scene.add(robots)
    world.reset()
    robots.initialize()
    
    # 配置机器人物理属性
    apply_massapi_all_tb3()
    for _ in range(10):
        world.step(render=True)
    
    # 配置轮子关节
    left_idx, right_idx = configure_wheel_joints(robots)
    if left_idx is None or right_idx is None:
        print("[ERROR] Could not find wheel joints")
        simulation_app.close()
        return
    
    # 差速控制器
    diff_ctrl = DifferentialController(
        name="tb3_diff_ctrl",
        wheel_radius=WHEEL_RADIUS,
        wheel_base=WHEEL_BASE,
        max_linear_speed=max_v,
        max_angular_speed=max_w,
    )
    
    # ==================== Stage2场景配置：4种场景均匀分配 ====================
    scene_list = [SCENE_EMPTY, SCENE_CYLINDER, SCENE_DOOR, SCENE_BOX]
    scene_types = [scene_list[i % 4] for i in range(num_envs)]  # 循环分配
    
    print(f"\n=== Stage2场景配置 ===")
    scene_count = {}
    for i, scene_type in enumerate(scene_types):
        scene_count[scene_type] = scene_count.get(scene_type, 0) + 1
    for scene_type, count in sorted(scene_count.items()):
        print(f"{scene_type}: {count}个环境")
    
    # 场景管理器 - Stage2：4种场景均匀分配
    scene_manager = SceneManager(
        num_envs=num_envs,
        env_size=env_size,
        env_origins=env_origins,
        stage=stage
    )
    scene_manager.wall_thickness = wall_thickness
    scene_manager.wall_height = wall_height
    # 传入None让scene_manager自动决定：EMPTY场景不显示墙，其他场景显示
    scene_manager.create_scene_obstacles(scene_types=scene_types, show_visual_walls=None)
    
    # 重要：初始化所有环境的障碍物位置（从地下移上来）
    # 因为 create_scene_obstacles 创建时所有障碍物都在 z=-10.0，需要 reset 来设置正确位置
    scene_manager.reset_scene_obstacles(env_ids=np.arange(num_envs), rng=rng)
    
    # 目标位置和标记
    goal_pos = np.zeros((num_envs, 3), dtype=np.float32)
    markers = []
    for i in range(num_envs):
        goal_pos[i] = scene_manager.get_goal_config(i, rng=rng)
        m = world.scene.add(
            VisualSphere(
                prim_path=f"/World/envs/env_{i}/GoalMarker",
                name=f"goal_marker_{i}",
                position=goal_pos[i].tolist(),
                radius=0.05,
                color=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            )
        )
        markers.append(m)
    
    # 初始化变量
    episode_start_time = np.zeros((num_envs,), dtype=np.float32)
    action_history = np.zeros((num_envs, 4), dtype=np.float32)  # [v_t-1, w_t-1, v_t-2, w_t-2]
    prev_measured_v = np.zeros((num_envs,), dtype=np.float32)
    prev_measured_w = np.zeros((num_envs,), dtype=np.float32)
    
    # 动作历史缓存（用于平滑惩罚）
    prev_act = np.zeros((num_envs, 2), dtype=np.float32)  # a_t-1
    prev_prev_act = np.zeros((num_envs, 2), dtype=np.float32)  # a_t-2
    
    # 重置所有机器人
    for i in range(num_envs):
        spawn_pos, spawn_yaw = scene_manager.get_robot_spawn_config(i, rng=rng)
        spawn_rot = quat_wxyz_from_yaw(np.array([spawn_yaw], dtype=np.float32))
        idx = np.array([i], dtype=np.int32)
        robots.set_world_poses(
            positions=spawn_pos.reshape(1, 3),
            orientations=spawn_rot.reshape(1, 4),
            indices=idx,
        )
        robots.set_velocities(np.zeros((1, 6), dtype=np.float32), indices=idx)
        robots.set_joint_velocities(np.zeros((1, robots.num_dof), dtype=np.float32), indices=idx)
        robots.set_joint_velocity_targets(np.zeros((1, robots.num_dof), dtype=np.float32), indices=idx)
        episode_start_time[i] = 0.0
    
    # ==================== 初始化模块化组件 ====================
    # 1. 创建核心网络
    network_type = network_cfg.network.type
    network_params = network_cfg.network.params.copy() if network_cfg.network.params else {}
    
    # 自动设置所有架构通用的参数（这些不应该在配置文件中重复）
    network_params['obs_dim'] = 44  # 固定44维观察
    network_params['act_dim'] = 2   # 固定2维动作
    network_params['lidar_dim'] = lidar_num_rays  # 从环境配置读取
    network_params['state_dim'] = 8  # 固定：UserIntent(2) + BaseVel(2) + ActionHistory(4)
    
    policy = create_policy(network_type, **network_params)
    value = create_value(network_type, **network_params)
    
    # 2. 创建RL算法
    algorithm_type = algo_cfg.algorithm.type
    algorithm_params = algo_cfg.algorithm.params
    rl_agent = create_algorithm(
        algorithm_type,
        policy=policy,
        value=value,
        config=algorithm_params,
        device=device
    )
    
    # 3. 加载checkpoint（如果存在）
    print(f"\n=== 模型加载 ===")
    checkpoint_dir_stage1 = os.path.join(PROJECT_ROOT, "checkpoints", "ppo_stage1_test_modular")
    checkpoint_loaded = load_checkpoint(rl_agent, checkpoint_path=checkpoint_path, checkpoint_dir=checkpoint_dir_stage1)
    if checkpoint_loaded:
        print("[信息] 从Stage1模型继续训练")
    else:
        print("[信息] 从头开始训练")
    
    # 4. 创建奖励组合器
    reward_composer = RewardComposer(reward_cfg.rewards)
    
    for _ in range(8):
        robots.set_joint_velocity_targets(np.zeros((num_envs, robots.num_dof), dtype=np.float32))
        robots.set_velocities(np.zeros((num_envs, 6), dtype=np.float32))
        robots.set_joint_velocities(np.zeros((num_envs, robots.num_dof), dtype=np.float32))
        world.step(render=True)
    
    # 训练参数
    update_freq = 64  # 每64步更新一次策略
    step_count = -1
    reward_buffer = deque(maxlen=100)  # 奖励缓存，用于监控训练
    total_epochs = 300           # Stage2：总共训练300轮（每轮一次ppo.update）
    save_every = 10             # 每10轮保存一次checkpoint
    update_count = 0            # 已完成的更新轮数
    checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints", "ppo_stage2_test_modular")
    writer = SummaryWriter(log_dir=os.path.join(PROJECT_ROOT, "runs", "ppo_stage2_test_modular"))
    reward_component_sums = {}
    reward_component_count = 0
    
    # 主循环
    current_time = 0.0
    last_obs_print = time.time()
    last_debug_print = time.time()

    temp_cur_act = np.zeros((num_envs, 2), dtype=np.float32)
    temp_logprob = np.zeros((num_envs,), dtype=np.float32)
    
    print(f"\n=== 开始Stage2训练 ===")
    print(f"环境数量: {num_envs}")
    print(f"总训练轮数: {total_epochs}")
    print(f"每{update_freq}步更新一次策略")
    print(f"每{save_every}轮保存一次checkpoint")
    
    while simulation_app.is_running():
        current_time += physics_dt
        step_count += 1
        
        # 获取机器人状态
        pos, rot = robots.get_world_poses()
        yaw = yaw_from_quat_wxyz(rot)
        
        # 局部坐标计算
        car_local = pos[:, :2] - env_origins[:, :2]  # (num_envs, 2)
        x_local = (pos[:, 0] - env_origins[:, 0]).astype(np.float32)
        y_local = (pos[:, 1] - env_origins[:, 1]).astype(np.float32)
        
        # 目标偏移计算
        goal_local = goal_pos[:, :2] - env_origins[:, :2]
        gx_local = goal_local[:, 0].astype(np.float32)
        gy_local = goal_local[:, 1].astype(np.float32)
        
        # 到目标的距离和方位角误差
        ex = gx_local - x_local
        ey = gy_local - y_local
        goal_heading = np.arctan2(ey, ex)
        
        # ==================== 观察数据采集 ====================
        # 1. LiDAR数据
        lidar_ranges = compute_lidar_ranges(
            robot_positions=pos,
            robot_orientations=rot,
            env_origins=env_origins,
            lidar_num_rays=lidar_num_rays,
            lidar_max_range=lidar_max_range,
            scene_manager=scene_manager
        )
        
        # 2. 基础速度
        robot_velocities_np = robots.get_velocities()
        # 移到GPU计算（如果可用）
        robot_velocities_torch = torch.from_numpy(robot_velocities_np).float().to(device)
        base_vel_np = get_base_velocity_from_tensor(
            robot_velocities_torch,
            max_v=max_v,
            max_w=max_w
        )  # 转回CPU用于后续numpy计算
        
        # 3. 当前测量速度
        current_measured_v = robot_velocities_np[:, 0]
        current_measured_w = robot_velocities_np[:, 5]
        
        # 4. 用户意图
        # 张量移到GPU
        robot_pos_torch = torch.from_numpy(pos).float().to(device)
        robot_rot_torch = torch.from_numpy(rot).float().to(device)
        goal_pos_torch = torch.from_numpy(goal_pos).float().to(device)
        env_origins_torch = torch.from_numpy(env_origins).float().to(device)
        _, user_intent_env, _ = compute_user_intent_torch(
            robot_pos_torch, torch.tensor(yaw), goal_pos_torch, env_origins_torch, normalize=True
        )
        user_intent_np = user_intent_env.cpu().numpy().astype(np.float32)  # 转回CPU
        
        # 5. 组装观察向量
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
        
        # ==================== 障碍物碰撞检测 ====================
        obstacle_collision, critical_flag, min_dist = check_obstacle_collision(lidar_ranges, DCOL, DCRIT)
        
        # ==================== RL动作选择 ====================
        v_cmd = np.zeros(num_envs, dtype=np.float32)
        w_cmd = np.zeros(num_envs, dtype=np.float32)
        log_probs = np.zeros(num_envs, dtype=np.float32)
        current_act = np.zeros((num_envs, 2), dtype=np.float32)  # 归一化动作
        
        for i in range(num_envs):
            # 碰撞时强制停止
            if obstacle_collision[i]:
                act = np.array([0.0, 0.0])
                log_prob = 0.0
            else:
                # RL选择动作（归一化的[-1,1]）
                if step_count % 10 == 0:
                    act, log_prob = rl_agent.select_action(obs[i])
                    temp_cur_act[i] = act
                    temp_logprob[i] = log_prob
                else:
                    # 持续控制
                    act = temp_cur_act[i]
                    obs_cur = torch.FloatTensor(obs[i]).unsqueeze(0).to(device)
                    
                    # NaN检测：检查输入观察值
                    if torch.isnan(obs_cur).any():
                        print(f"[ERROR] NaN detected in observation at step {step_count}, env {i}")
                        print(f"  obs min: {obs_cur.min().item():.6f}, max: {obs_cur.max().item():.6f}")
                        print(f"  obs contains inf: {torch.isinf(obs_cur).any().item()}")
                        # 使用零动作作为fallback
                        act = np.array([0.0, 0.0])
                        log_prob = 0.0
                    else:
                        mean, log_std = rl_agent.policy(obs_cur)
                        
                        # NaN检测：检查网络输出
                        if torch.isnan(mean).any() or torch.isnan(log_std).any():
                            print(f"[ERROR] NaN detected in policy output at step {step_count}, env {i}")
                            print(f"  mean min: {mean.min().item():.6f}, max: {mean.max().item():.6f}")
                            print(f"  mean contains inf: {torch.isinf(mean).any().item()}")
                            print(f"  log_std min: {log_std.min().item():.6f}, max: {log_std.max().item():.6f}")
                            print(f"  log_std contains inf: {torch.isinf(log_std).any().item()}")
                            # 检查网络参数
                            for name, param in rl_agent.policy.named_parameters():
                                if torch.isnan(param).any():
                                    print(f"  [ERROR] NaN in policy parameter: {name}")
                            # 使用零动作作为fallback
                            act = np.array([0.0, 0.0])
                            log_prob = 0.0
                        else:
                            std = log_std.exp()
                            # 检查std是否有效
                            if torch.isnan(std).any() or (std <= 0).any():
                                print(f"[ERROR] Invalid std at step {step_count}, env {i}")
                                print(f"  std min: {std.min().item():.6f}, max: {std.max().item():.6f}")
                                act = np.array([0.0, 0.0])
                                log_prob = 0.0
                            else:
                                dist = Normal(mean, std)
                                log_prob = dist.log_prob(
                                    torch.tensor(act).unsqueeze(0).to(device)
                                ).sum().detach().cpu().numpy()
            
            log_probs[i] = log_prob
            current_act[i] = act
            
            # 反归一化到实际速度范围
            v_cmd[i] = act[0] * max_v
            w_cmd[i] = act[1] * max_w
            
            # 速度裁剪
            v_cmd[i] = np.clip(v_cmd[i], 0.0, max_v)
            w_cmd[i] = np.clip(w_cmd[i], -max_w, max_w)
        
        # ==================== 动作下发 ====================
        targets = np.zeros((num_envs, robots.num_dof), dtype=np.float32)
        for i in range(num_envs):
            action = diff_ctrl.forward(command=np.array([float(v_cmd[i]), float(w_cmd[i])], dtype=np.float32))
            targets[i, left_idx] = float(action.joint_velocities[0])
            targets[i, right_idx] = float(action.joint_velocities[1])

        # 更新动作历史：旧的 t-1 -> t-2，新的 v_cmd/w_cmd -> t-1
        action_history[:, 2:] = action_history[:, 0:2]  # 旧的 t-1 -> t-2
        action_history[:, 0] = v_cmd
        action_history[:, 1] = w_cmd
        
        robots.set_joint_velocity_targets(targets)
        world.step(render=True)
        
        # ==================== 重置条件检查 ====================
        dist = np.sqrt(ex * ex + ey * ey).astype(np.float32)
        yaw_err = wrap_to_pi(goal_heading - yaw).astype(np.float32)
        reset_timeout = (current_time - episode_start_time) >= TIMEOUT_SECONDS
        reached = dist <= reset_dist
        
        boundary_collision = np.zeros((num_envs,), dtype=bool)
        for i in range(num_envs):
            boundary_collision[i] = scene_manager.check_boundary_collision(i, np.array([x_local[i], y_local[i]]))
        
        # 总重置条件：障碍物碰撞 + 边界碰撞 + 超时 + 到达目标
        to_reset = obstacle_collision | boundary_collision | reset_timeout | reached
        
        # ==================== 奖励计算与经验存储 ====================
        next_obs = get_obs(robots, env_origins, goal_pos, lidar_num_rays, lidar_max_range, scene_manager, max_v, max_w, action_history, device=device)
        
        # 使用模块化奖励组合器计算奖励
        # 注意：传入测量速度用于离心力计算（如果启用）
        rewards, reward_info = reward_composer.compute(
            dist=dist,
            yaw_err=yaw_err,
            v_cmd=v_cmd,
            w_cmd=w_cmd,
            v_measured=current_measured_v,  # 实际测量的线速度（用于离心力计算）
            w_measured=current_measured_w,  # 实际测量的角速度（用于离心力计算）
            collision=obstacle_collision,
            reached=reached,
            timeout=reset_timeout,
            min_dist=min_dist,
            current_act=current_act,
            prev_act=prev_act,
            prev_prev_act=prev_prev_act
        )
        
        # 累计奖励分量用于TB统计
        reward_component_count += len(rewards)
        for k, v in reward_info.items():
            if k not in reward_component_sums:
                reward_component_sums[k] = 0.0
            reward_component_sums[k] += float(np.sum(v))
        
        for i in range(num_envs):
            reward_buffer.append(rewards[i])
            
            # 存储经验（动作归一化）
            rl_agent.store_transition(
                obs=obs[i],
                act=current_act[i],
                rew=rewards[i],
                next_obs=next_obs[i],
                done=to_reset[i],
                log_prob=log_probs[i]
            )
        
        # ==================== 策略更新 ====================
        if step_count % update_freq == 0 and len(rl_agent.buffer["obs"]) >= rl_agent.batch_size:
            buffer_len = len(rl_agent.buffer["obs"])
            train_info = rl_agent.update()
            update_count += 1
            avg_reward = np.mean(reward_buffer) if reward_buffer else 0.0
            print(f"\n=== PPO策略更新 ===")
            print(f"更新步数: {step_count}, 已完成轮数: {update_count}/{total_epochs}, 最近100步平均奖励: {avg_reward:.2f}")
            print(f'Actor Loss: {train_info["actor_loss"]:.4f}, Critic Loss: {train_info["critic_loss"]:.4f}')
            print(f'Entropy: {train_info["entropy"]:.4f}, Approx KL: {train_info["approx_kl"]:.4f}')

            # TensorBoard记录：训练loss信息
            writer.add_scalar("train/actor_loss", train_info["actor_loss"], update_count)
            writer.add_scalar("train/critic_loss", train_info["critic_loss"], update_count)
            writer.add_scalar("train/entropy", train_info["entropy"], update_count)
            writer.add_scalar("train/approx_kl", train_info["approx_kl"], update_count)
            
            # TensorBoard记录：奖励分量均值、总奖励、rollout信息
            if reward_component_count > 0:
                for k, v_sum in reward_component_sums.items():
                    writer.add_scalar(f"reward/{k}", v_sum / reward_component_count, update_count)
            writer.add_scalar("reward/total_mean", avg_reward, update_count)
            writer.add_scalar("rollout/steps_per_update", update_freq, update_count)
            writer.add_scalar("rollout/buffer_size", buffer_len, update_count)
            writer.add_scalar("rollout/step_count", step_count, update_count)

            # 重置累计器
            for k in reward_component_sums.keys():
                reward_component_sums[k] = 0.0
            reward_component_count = 0
            
            # 定期保存checkpoint
            if update_count % save_every == 0:
                save_model(rl_agent, checkpoint_dir, f"epoch_{update_count:03d}")

            # 达到总轮数，保存最终模型并退出
            if update_count >= total_epochs:
                # 添加时间戳后缀（精确到日期和小时）
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H")
                save_model(rl_agent, checkpoint_dir, f"final_{timestamp}")
                break
        
        # ==================== 机器人重置 ====================
        if np.any(to_reset):
            reset_ids = np.nonzero(to_reset)[0]
            
            # ==================== 处理根节点速度 ====================
            current_base_vels = robots.get_velocities()  # 形状: (num_envs, 6)
            current_base_vels[reset_ids] = 0.0
            robots.set_velocities(current_base_vels)

            # ==================== 处理关节实时速度 ====================
            current_joint_vels = robots.get_joint_velocities()  # 形状: (num_envs, num_dof)
            current_joint_vels[reset_ids] = 0.0
            robots.set_joint_velocities(current_joint_vels)
            
            # 重置每个环境
            for i in reset_ids:
                # 重置场景障碍物（随机化位置/大小）
                scene_manager.reset_scene_obstacles(np.array([i]), rng=rng)
                
                # 重置机器人位置和朝向
                spawn_pos, spawn_yaw = scene_manager.get_robot_spawn_config(i, rng=rng)
                spawn_rot = quat_wxyz_from_yaw(np.array([spawn_yaw], dtype=np.float32))
                idx = np.array([i], dtype=np.int32)
                
                robots.set_world_poses(
                    positions=spawn_pos.reshape(1, 3),
                    orientations=spawn_rot.reshape(1, 4),
                    indices=idx,
                )
                robots.set_velocities(np.zeros((1, 6), dtype=np.float32), indices=idx)
                robots.set_joint_velocities(np.zeros((1, robots.num_dof), dtype=np.float32), indices=idx)
                robots.set_joint_velocity_targets(np.zeros((1, robots.num_dof), dtype=np.float32), indices=idx)
                
                # 重置目标
                goal_pos[i] = scene_manager.get_goal_config(i, rng=rng)
                markers[i].set_world_pose(position=goal_pos[i].tolist(), orientation=[1, 0, 0, 0])
                
                # 重置计时和状态
                episode_start_time[i] = current_time
                action_history[i] = np.zeros(4, dtype=np.float32)  # 重置为4维零向量 [v_t-1, w_t-1, v_t-2, w_t-2]
                prev_measured_v[i] = 0.0
                prev_measured_w[i] = 0.0
                
                # 重置动作历史（平滑惩罚用）
                prev_act[i] = np.zeros(2)
                prev_prev_act[i] = np.zeros(2)
            
            # 稳定步骤
            for _ in range(8):
                robots.set_joint_velocity_targets(np.zeros((num_envs, robots.num_dof), dtype=np.float32))
                robots.set_velocities(np.zeros((num_envs, 6), dtype=np.float32))
                robots.set_joint_velocities(np.zeros((num_envs, robots.num_dof), dtype=np.float32))
                world.step(render=True)
        
        # 更新动作历史
        prev_measured_v = current_measured_v.copy()
        prev_measured_w = current_measured_w.copy()
        
        # 更新平滑惩罚用的动作历史
        prev_prev_act = prev_act.copy()
        prev_act = current_act.copy()
        
        # ==================== 打印输出 ====================
        now = time.time()
        # 每隔1秒打印观察
        if now - last_obs_print >= 1.0:
            obs_env0 = obs[0]
            non_lidar_obs = obs_env0[36:44]
            lidar_first2 = obs_env0[0:2]
            
            last_obs_print = now
        
        # 每0.5秒打印调试信息
        if now - last_debug_print > 0.5:
            last_debug_print = now
        
        time.sleep(0.001)
    
    simulation_app.close()
    writer.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage2 PPO训练脚本")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="指定checkpoint文件路径（可选，默认自动查找最新的stage1模型）")
    args = parser.parse_args()
    
    main(checkpoint_path=args.checkpoint)
