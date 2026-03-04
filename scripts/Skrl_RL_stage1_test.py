"""
RL Stage 1 完整训练脚本
基于 2_RL_stage1_test.py，添加 RL 训练组件
4个empty场景，使用RL策略训练，包含完整的PPO训练流程
"""
import os
import sys
import time
import numpy as np
import torch
from omegaconf import OmegaConf

# 添加项目根目录到 Python 路径（必须在导入其他模块之前）
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from omni.isaac.kit import SimulationApp

# 初始化 Isaac Sim
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.physics_context import PhysicsContext
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.objects import VisualSphere
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
from omni.physx import get_physx_scene_query_interface
from pxr import Gf

# 导入 sim 模块
from sim.robot import (
    TB3_USD, WHEEL_RADIUS, WHEEL_BASE,
    apply_massapi_all_tb3, configure_wheel_joints
)
from sim.scenes import (
    SCENE_EMPTY,
    SceneManager, yaw_from_quat_wxyz, quat_wxyz_from_yaw, wrap_to_pi
)

# 导入观察模块
from envs.observations import assemble_observations, get_base_velocity_from_tensor
from envs.user_intent import compute_user_intent_torch
from envs.rewards import RewardCalculator

# 导入 RL 组件
from models.cnn_lstm_policy_base import WaffleCNNLSTMPolicyBase
from agents.ppo_agent_base import PPOAgentBase
from skrl.memories.torch import RandomMemory
from gym import spaces

# 导入 Tensorboard
from torch.utils.tensorboard import SummaryWriter


def load_config(cfg_path="cfg/task/WaffleDrive.yaml"):
    """从配置文件加载参数"""
    # 如果路径是相对路径，基于项目根目录
    if not os.path.isabs(cfg_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        cfg_path = os.path.join(project_root, cfg_path)
    cfg = OmegaConf.load(cfg_path)
    return cfg


def compute_lidar_ranges(robot_positions, robot_orientations, env_origins, 
                         lidar_num_rays, lidar_max_range, scene_manager):
    """
    计算真实的 LiDAR 范围（使用 ray casting）
    
    Args:
        robot_positions: (N, 3) 机器人世界坐标位置
        robot_orientations: (N, 4) 机器人朝向（四元数 wxyz）
        env_origins: (N, 3) 环境原点
        lidar_num_rays: LiDAR 射线数量
        lidar_max_range: LiDAR 最大测量范围
        scene_manager: SceneManager 实例，用于获取障碍物信息
    
    Returns:
        lidar_ranges: (N, lidar_num_rays) LiDAR 距离测量值
    """
    num_envs = robot_positions.shape[0]
    lidar_ranges = np.ones((num_envs, lidar_num_rays), dtype=np.float32) * lidar_max_range
    
    # 获取 PhysX 场景查询接口
    physx_interface = get_physx_scene_query_interface()
    
    # LiDAR 高度（机器人中心高度）
    lidar_height = 0.1  # 10cm，略高于地面
    
    for env_id in range(num_envs):
        robot_pos = robot_positions[env_id]
        robot_rot = robot_orientations[env_id]
        env_origin = env_origins[env_id]
        
        # 计算机器人局部位置（用于边界碰撞检测）
        robot_pos_local = np.array([
            robot_pos[0] - env_origin[0],
            robot_pos[1] - env_origin[1]
        ])
        
        # 从四元数提取 yaw 角
        w, x, y, z = robot_rot[0], robot_rot[1], robot_rot[2], robot_rot[3]
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        
        # 生成 LiDAR 射线方向（在机器人局部坐标系中，均匀分布在 360 度）
        angles = np.linspace(0, 2 * np.pi, lidar_num_rays, endpoint=False)
        
        for ray_idx, angle in enumerate(angles):
            # 射线方向（在机器人局部坐标系中）
            ray_dir_local = np.array([np.cos(angle), np.sin(angle), 0.0])
            
            # 转换到世界坐标系（考虑机器人朝向）
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            ray_dir_world = np.array([
                ray_dir_local[0] * cos_yaw - ray_dir_local[1] * sin_yaw,
                ray_dir_local[0] * sin_yaw + ray_dir_local[1] * cos_yaw,
                0.0
            ])
            
            # 射线起点（机器人位置 + LiDAR 高度）
            ray_origin = Gf.Vec3f(
                float(robot_pos[0]),
                float(robot_pos[1]),
                float(robot_pos[2] + lidar_height)
            )
            
            # 射线方向（归一化的方向向量）
            ray_dir_norm = np.linalg.norm(ray_dir_world)
            if ray_dir_norm > 1e-6:
                ray_dir_normalized = ray_dir_world / ray_dir_norm
            else:
                ray_dir_normalized = np.array([1.0, 0.0, 0.0])
            
            ray_dir = Gf.Vec3f(
                float(ray_dir_normalized[0]),
                float(ray_dir_normalized[1]),
                float(ray_dir_normalized[2])
            )
            
            # 执行 ray cast
            hit = physx_interface.raycast_closest(
                ray_origin, 
                ray_dir, 
                float(lidar_max_range),
                bothSides=False
            )
            
            if hit:
                distance = hit.get('distance', lidar_max_range)
                lidar_ranges[env_id, ray_idx] = min(float(distance), lidar_max_range)
            else:
                # 检查是否与边界墙碰撞
                ray_end_local = robot_pos_local + ray_dir_local[:2] * lidar_max_range
                
                if scene_manager.check_boundary_collision(env_id, ray_end_local):
                    env_half = scene_manager.env_size * 0.5
                    dist_to_boundary = lidar_max_range
                    
                    # 检查四个边界
                    if ray_dir_local[0] > 0:
                        t = (env_half - robot_pos_local[0]) / ray_dir_local[0] if ray_dir_local[0] > 1e-6 else lidar_max_range
                        if 0 < t < dist_to_boundary:
                            dist_to_boundary = t
                    elif ray_dir_local[0] < 0:
                        t = (-env_half - robot_pos_local[0]) / ray_dir_local[0] if ray_dir_local[0] < -1e-6 else lidar_max_range
                        if 0 < t < dist_to_boundary:
                            dist_to_boundary = t
                    
                    if ray_dir_local[1] > 0:
                        t = (env_half - robot_pos_local[1]) / ray_dir_local[1] if ray_dir_local[1] > 1e-6 else lidar_max_range
                        if 0 < t < dist_to_boundary:
                            dist_to_boundary = t
                    elif ray_dir_local[1] < 0:
                        t = (-env_half - robot_pos_local[1]) / ray_dir_local[1] if ray_dir_local[1] < -1e-6 else lidar_max_range
                        if 0 < t < dist_to_boundary:
                            dist_to_boundary = t
                    
                    lidar_ranges[env_id, ray_idx] = min(dist_to_boundary, lidar_max_range)
                else:
                    lidar_ranges[env_id, ray_idx] = lidar_max_range
    
    return lidar_ranges


def main():
    # ==================== 配置参数 ====================
    # 训练参数
    TOTAL_STEPS = 10000  # 总采样步数
    CHECKPOINT_INTERVAL = 1000  # Checkpoint 保存间隔
    CHECKPOINT_DIR = "checkpoints/stage1_test"
    TENSORBOARD_DIR = "runs/stage1_test"
    
    # 重置冷却期参数
    COOLDOWN_STEPS = 10  # 冷却期步数
    
    # 观察输出参数
    OBS_PRINT_INTERVAL = 1.0  # 每隔多少秒输出一次观察（秒）
    DEBUG_PRINT_INTERVAL = 0.5  # 每隔多少秒输出一次调试信息（秒）
    
    # 创建必要的目录
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(TENSORBOARD_DIR, exist_ok=True)
    
    # ==================== 加载配置 ====================
    cfg = load_config()
    
    # ==================== 从配置读取环境参数 ====================
    num_envs = int(cfg.env.numEnvs)
    env_size = float(cfg.env.scene.env_size)
    env_gap = 2.0
    env_spacing = env_size + env_gap
    reset_dist = float(cfg.env.resetDist)
    wall_thickness = float(cfg.env.scene.wall_thickness)
    wall_height = float(cfg.env.scene.wall_height)
    show_visual_walls = bool(cfg.env.scene.show_visual_walls)
    TIMEOUT_SECONDS = 60.0
    
    # ==================== 从配置读取物理参数 ====================
    physics_dt = float(cfg.sim.dt)  # 0.025，对应40Hz
    render_dt = 1.0 / 60.0
    
    # ==================== 从配置读取机器人限制 ====================
    max_v = float(cfg.env.robot_limits.max_v)
    max_w = float(cfg.env.robot_limits.max_w)
    
    # ==================== 从配置读取 LiDAR 参数 ====================
    lidar_num_rays = int(cfg.env.lidar.num_rays)
    lidar_max_range = float(cfg.env.lidar.max_range)
    
    # ==================== 从配置读取奖励函数参数 ====================
    reward_cfg = cfg.env.rewards
    ra = float(reward_cfg.get("ra", 0.5))
    rl = float(reward_cfg.get("rl", -0.5))
    rh = float(reward_cfg.get("rh", -0.5))
    phi_thresh = float(reward_cfg.get("phi_thresh", 0.2))
    ras = float(reward_cfg.get("ras", -0.02))
    rc = float(reward_cfg.get("rc", -1.0))
    rcrit = float(reward_cfg.get("rcrit", -1.0))
    rcol = float(reward_cfg.get("rcol", -100.0))
    d_col = float(reward_cfg.get("d_col", 0.12))
    d_crit = float(reward_cfg.get("d_crit", 0.35))
    
    # ==================== 从配置读取训练参数 ====================
    # 使用项目根目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    train_cfg_path = os.path.join(project_root, "cfg/train/WaffleDrivePPO.yaml")
    train_cfg = OmegaConf.load(train_cfg_path)
    rollouts = int(train_cfg.agent.rollouts)  # 8
    
    # ==================== 初始化随机数生成器 ====================
    rng = np.random.default_rng(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # ==================== 1. 创建 World 和物理环境 ====================
    print("=" * 80)
    print("🌍 创建 World 和物理环境...")
    print("=" * 80)
    
    world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=render_dt)
    PhysicsContext().substeps = 8
    world.scene.add_default_ground_plane()
    stage = world.scene.stage
    
    # ==================== 2. 计算环境原点 ====================
    env_origins = np.zeros((num_envs, 3), dtype=np.float32)
    for i in range(num_envs):
        ix = i % 2
        iy = i // 2
        env_origins[i, 0] = ix * env_spacing
        env_origins[i, 1] = iy * env_spacing
    
    # ==================== 3. 加载机器人模型 ====================
    for i in range(num_envs):
        tb3_root = f"/World/envs/env_{i}/TB3"
        add_reference_to_stage(usd_path=TB3_USD, prim_path=tb3_root)
    
    # 等待场景稳定
    for _ in range(180):
        world.step(render=True)
    
    # ==================== 4. 创建机器人视图 ====================
    robots = ArticulationView(
        prim_paths_expr="/World/envs/env_.*/TB3/a__namespace_base_footprint",
        name="tb3_view",
        reset_xform_properties=False,
    )
    world.scene.add(robots)
    world.reset()
    robots.initialize()
    
    # ==================== 5. 配置机器人物理属性 ====================
    apply_massapi_all_tb3()
    for _ in range(10):
        world.step(render=True)
    
    # ==================== 6. 配置轮子关节 ====================
    left_idx, right_idx = configure_wheel_joints(robots)
    if left_idx is None or right_idx is None:
        print("[ERROR] Could not find wheel joints in dof_names.")
        print(robots.dof_names)
        simulation_app.close()
        return
    
    # ==================== 7. 创建差速控制器 ====================
    diff_ctrl = DifferentialController(
        name="tb3_diff_ctrl",
        wheel_radius=WHEEL_RADIUS,
        wheel_base=WHEEL_BASE,
        max_linear_speed=max_v,
        max_angular_speed=max_w,
    )
    
    # ==================== 8. 创建场景管理器 ====================
    scene_types = [SCENE_EMPTY] * num_envs
    show_visual_walls_list = [show_visual_walls] * num_envs
    
    scene_manager = SceneManager(
        num_envs=num_envs,
        env_size=env_size,
        env_origins=env_origins,
        stage=stage
    )
    scene_manager.wall_thickness = wall_thickness
    scene_manager.wall_height = wall_height
    scene_manager.create_scene_obstacles(
        scene_types=scene_types,
        show_visual_walls=show_visual_walls_list
    )
    
    # ==================== 9. 确定设备 ====================
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # ==================== 10. 创建奖励计算器 ====================
    reward_calculator = RewardCalculator(
        num_envs=num_envs,
        device=device,
        ra=ra, rl=rl, rh=rh, phi_thresh=phi_thresh,
        ras=ras, rc=rc, rcrit=rcrit, rcol=rcol,
        d_col=d_col, d_crit=d_crit,
        max_v=max_v, max_w=max_w
    )
    
    # ==================== 11. 初始化目标位置和标记 ====================
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
    
    # ==================== 12. 创建 RL 模型 ====================
    print("\n" + "=" * 80)
    print("🧠 创建 RL 模型...")
    print("=" * 80)
    
    num_observations = 44  # LiDAR[36] + User Input[2] + Base Velocity[2] + Action History[4]
    num_actions = 2  # [v, w]
    
    observation_space = spaces.Box(
        low=-np.inf, high=np.inf,
        shape=(num_observations,), dtype=np.float32
    )
    action_space = spaces.Box(
        low=-1.0, high=1.0,
        shape=(num_actions,), dtype=np.float32
    )
    
    models = {
        "policy": WaffleCNNLSTMPolicyBase(
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            clip_actions=True
        ),
        "value": WaffleCNNLSTMPolicyBase(
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            clip_actions=True
        )
    }
    
    print("✅ 模型创建完成")
    print(f"  - Policy Model: {type(models['policy']).__name__}")
    print(f"  - Value Model: {type(models['value']).__name__}")
    print(f"  - Device: {device}")
    
    # ==================== 13. 创建 Memory ====================
    print("\n" + "=" * 80)
    print("💾 创建 Memory...")
    print("=" * 80)
    
    memory_size = rollouts * num_envs  # 8 * 4 = 32
    
    memory = RandomMemory(
        memory_size=memory_size,
        num_envs=num_envs,
        device=device,
        replacement=True
    )
    
    print("✅ Memory 创建完成")
    print(f"  - Memory Size: {memory_size}")
    print(f"  - Rollouts: {rollouts}")
    print(f"  - Num Envs: {num_envs}")
    
    # ==================== 14. 创建 PPO Agent ====================
    print("\n" + "=" * 80)
    print("🤖 创建 PPO Agent...")
    print("=" * 80)
    
    # 创建完整配置（用于 agent）
    full_cfg = OmegaConf.create({
        "task": cfg,
        "train": train_cfg
    })
    
    agent = PPOAgentBase(
        models=models,
        memory=memory,
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        cfg=full_cfg
    )
    
    print("✅ PPO Agent 创建完成")
    
    # ==================== 15. 创建 Tensorboard ====================
    print("\n" + "=" * 80)
    print("📊 初始化 Tensorboard...")
    print("=" * 80)
    
    writer = SummaryWriter(log_dir=TENSORBOARD_DIR)
    print("✅ Tensorboard 初始化完成")
    print(f"  - Log Dir: {TENSORBOARD_DIR}")
    
    # ==================== 16. 初始化训练状态 ====================
    print("\n" + "=" * 80)
    print("🔄 初始化训练状态...")
    print("=" * 80)
    
    # Active mask 和 cooldown（在训练脚本中维护）
    active_mask = torch.ones(num_envs, dtype=torch.bool, device=device)
    cooldown_steps = torch.zeros(num_envs, dtype=torch.int32, device=device)
    
    # 观察输出控制
    last_obs_print = time.time()
    last_debug_print = time.time()
    
    # 统计信息
    episode_rewards = torch.zeros(num_envs, device=device)
    episode_lengths = torch.zeros(num_envs, dtype=torch.int32, device=device)
    
    # Episode 开始时间
    episode_start_time = np.zeros((num_envs,), dtype=np.float32)
    
    # 动作历史（用于观察）
    action_history = np.zeros((num_envs, 4), dtype=np.float32)
    prev_measured_v = np.zeros((num_envs,), dtype=np.float32)
    prev_measured_w = np.zeros((num_envs,), dtype=np.float32)
    
    print("✅ 训练状态初始化完成")
    print(f"  - Active Mask: {active_mask.shape}")
    print(f"  - Cooldown Steps: {COOLDOWN_STEPS}")
    
    # ==================== 17. 初始重置所有环境 ====================
    for i in range(num_envs):
        spawn_pos, spawn_yaw = scene_manager.get_robot_spawn_config(i, rng=rng)
        spawn_rot = quat_wxyz_from_yaw(np.array([spawn_yaw], dtype=np.float32))
        
        idx = np.array([i], dtype=np.int32)
        robots.set_world_poses(
            positions=spawn_pos.reshape(1, 3),
            orientations=spawn_rot.reshape(1, 4),
            indices=idx,
        )
        robots.set_velocities(
            velocities=np.zeros((1, 6), dtype=np.float32),
            indices=idx,
        )
        robots.set_joint_velocities(
            velocities=np.zeros((1, robots.num_dof), dtype=np.float32),
            indices=idx,
        )
        robots.set_joint_velocity_targets(
            np.zeros((1, robots.num_dof), dtype=np.float32),
            indices=idx,
        )
        
        episode_start_time[i] = 0.0
    
    # ==================== 18. 主训练循环 ====================
    print("\n" + "=" * 80)
    print("🎯 开始训练...")
    print("=" * 80)
    print(f"  - Total Steps: {TOTAL_STEPS}")
    print(f"  - Checkpoint Interval: {CHECKPOINT_INTERVAL}")
    print(f"  - Rollouts: {rollouts}")
    print("=" * 80 + "\n")
    
    # 仿真时间跟踪
    current_time = 0.0
    
    # 初始化 skrl agent（在训练循环前）
    agent.agent.pre_interaction(timestep=0, timesteps=TOTAL_STEPS)
    
    # 获取初始观察
    pos, rot = robots.get_world_poses()
    yaw = yaw_from_quat_wxyz(rot)
    x_local = (pos[:, 0] - env_origins[:, 0]).astype(np.float32)
    y_local = (pos[:, 1] - env_origins[:, 1]).astype(np.float32)
    goal_offsets = goal_pos[:, :2] - env_origins[:, :2]
    gx_local = goal_offsets[:, 0].astype(np.float32)
    gy_local = goal_offsets[:, 1].astype(np.float32)
    
    # 计算初始观察
    lidar_ranges = compute_lidar_ranges(
        robot_positions=pos,
        robot_orientations=rot,
        env_origins=env_origins,
        lidar_num_rays=lidar_num_rays,
        lidar_max_range=lidar_max_range,
        scene_manager=scene_manager
    )
    
    robot_velocities_np = robots.get_velocities()
    robot_velocities_torch = torch.from_numpy(robot_velocities_np).float()
    base_vel_np = get_base_velocity_from_tensor(
        robot_velocities_torch,
        max_v=max_v,
        max_w=max_w
    )
    
    robot_pos_torch = torch.from_numpy(pos).float()
    robot_rot_torch = torch.from_numpy(rot).float()
    goal_pos_torch = torch.from_numpy(goal_pos).float()
    env_origins_torch = torch.from_numpy(env_origins).float()
    
    _, user_intent_env, _ = compute_user_intent_torch(
        robot_pos_torch,
        robot_rot_torch,
        goal_pos_torch,
        env_origins_torch,
        normalize=True
    )
    user_intent_np = user_intent_env.cpu().numpy().astype(np.float32)
    
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
    obs_torch = torch.from_numpy(obs).float().to(device)
    
    for step in range(TOTAL_STEPS):
        # ========== 1. RL 策略推理 ==========
        # skrl 的 act() 返回格式：根据 skrl 1.1.0 的文档，PPO 的 act() 返回 (actions, log_probs, values)
        # 但 values 可能是字典（包含 RNN 状态），需要正确提取
        act_result = agent.act(obs_torch, step, TOTAL_STEPS)
        
        # 根据 skrl 的标准格式，act() 应该返回 (actions, log_probs, values) 元组
        if isinstance(act_result, tuple) and len(act_result) >= 2:
            actions = act_result[0]  # actions: (num_envs, action_dim)
            log_probs = act_result[1] if len(act_result) > 1 else None  # log_probs: (num_envs,)
            values_raw = act_result[2] if len(act_result) > 2 else None  # values: 可能是 tensor 或 dict
            
            # 处理 values：如果是字典，提取实际的 value tensor
            if values_raw is not None:
                if isinstance(values_raw, dict):
                    # 如果 values 是字典，可能包含 'values' 键或直接是 value tensor
                    # 根据模型定义，value 模型返回 (value, {"rnn": ...})
                    # skrl 可能将整个返回值作为 values，需要提取实际的 value tensor
                    if 'values' in values_raw:
                        values = values_raw['values']
                    elif 'value' in values_raw:
                        values = values_raw['value']
                    else:
                        # 如果字典中没有 'values' 或 'value'，可能是 RNN 状态字典
                        # 需要单独调用 value 模型获取 values
                        with torch.no_grad():
                            value_output = agent.agent.models["value"].act({"states": obs_torch})
                            if isinstance(value_output, tuple):
                                values = value_output[0]  # 提取 value tensor，忽略 RNN 状态
                            else:
                                values = value_output
                elif isinstance(values_raw, torch.Tensor):
                    values = values_raw
                else:
                    # 如果 values_raw 既不是 dict 也不是 tensor，尝试转换
                    values = torch.tensor(values_raw, device=device, dtype=torch.float32)
            else:
                # 如果没有 values，需要单独获取
                with torch.no_grad():
                    value_output = agent.agent.models["value"].act({"states": obs_torch})
                    if isinstance(value_output, tuple):
                        values = value_output[0]  # 提取 value tensor
                    else:
                        values = value_output
        else:
            raise ValueError(f"Unexpected act() return format: {type(act_result)}, value: {act_result}")
        
        # 确保所有值都是 tensor 且形状正确
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, device=device, dtype=torch.float32)
        if log_probs is not None and not isinstance(log_probs, torch.Tensor):
            log_probs = torch.tensor(log_probs, device=device, dtype=torch.float32)
        if values is not None and not isinstance(values, torch.Tensor):
            values = torch.tensor(values, device=device, dtype=torch.float32)
        
        # 确保 values 的形状是 (num_envs,)，如果不是则 squeeze
        if values is not None and values.dim() > 1:
            values = values.squeeze(-1)  # (num_envs, 1) -> (num_envs,)
        
        # 确保 log_probs 的形状是 (num_envs,)
        if log_probs is not None and log_probs.dim() > 1:
            log_probs = log_probs.squeeze(-1)
        
        actions_np = actions.detach().cpu().numpy()
        
        # ========== 2. 冷却期处理：强制 inactive 环境的动作为 0 ==========
        actions_clamped = actions_np.copy()
        actions_clamped[~active_mask.cpu().numpy()] = 0.0
        
        # ========== 3. 反归一化动作（从 [-1, 1] 到 [v, w]）==========
        v_cmd = actions_clamped[:, 0] * max_v  # 线速度
        w_cmd = actions_clamped[:, 1] * max_w  # 角速度
        
        # ========== 4. 转换为轮速并应用到机器人 ==========
        targets = np.zeros((num_envs, robots.num_dof), dtype=np.float32)
        for i in range(num_envs):
            action = diff_ctrl.forward(command=np.array([float(v_cmd[i]), float(w_cmd[i])], dtype=np.float32))
            targets[i, left_idx] = float(action.joint_velocities[0])
            targets[i, right_idx] = float(action.joint_velocities[1])
        
        robots.set_joint_velocity_targets(targets)
        world.step(render=True)
        
        # ========== 5. 更新仿真时间 ==========
        current_time += physics_dt
        
        # ========== 6. 获取机器人状态（用于奖励和观察）==========
        pos, rot = robots.get_world_poses()
        yaw = yaw_from_quat_wxyz(rot)
        x_local = (pos[:, 0] - env_origins[:, 0]).astype(np.float32)
        y_local = (pos[:, 1] - env_origins[:, 1]).astype(np.float32)
        goal_offsets = goal_pos[:, :2] - env_origins[:, :2]
        gx_local = goal_offsets[:, 0].astype(np.float32)
        gy_local = goal_offsets[:, 1].astype(np.float32)
        
        ex = gx_local - x_local
        ey = gy_local - y_local
        dist = np.sqrt(ex * ex + ey * ey).astype(np.float32)
        
        robot_velocities_np = robots.get_velocities()
        current_measured_v = robot_velocities_np[:, 0]
        current_measured_w = robot_velocities_np[:, 5]
        
        # ========== 7. 计算奖励 ==========
        robot_pos_torch = torch.from_numpy(pos).float().to(device)
        robot_rot_torch = torch.from_numpy(rot).float().to(device)
        goal_pos_torch = torch.from_numpy(goal_pos).float().to(device)
        env_origins_torch = torch.from_numpy(env_origins).float().to(device)
        robot_velocities_torch = torch.from_numpy(robot_velocities_np).float().to(device)
        
        # 计算 user_intent（用于奖励计算）
        _, user_intent_env, _ = compute_user_intent_torch(
            robot_pos_torch,
            robot_rot_torch,
            goal_pos_torch,
            env_origins_torch,
            normalize=True
        )
        
        # 获取当前的 lidar_ranges（需要先计算）
        lidar_ranges = compute_lidar_ranges(
            robot_positions=pos,
            robot_orientations=rot,
            env_origins=env_origins,
            lidar_num_rays=lidar_num_rays,
            lidar_max_range=lidar_max_range,
            scene_manager=scene_manager
        )
        
        rewards_torch = reward_calculator.compute_rewards(
            lidar_ranges=torch.from_numpy(lidar_ranges).float().to(device),
            robot_positions=robot_pos_torch,
            robot_velocities=robot_velocities_torch,
            robot_orientations=robot_rot_torch,
            goal_positions=goal_pos_torch,
            env_origins=env_origins_torch,
            actions=actions,  # 当前动作（归一化），已经在 device 上
            action_history=torch.from_numpy(action_history).float().to(device),
            user_intent_env=user_intent_env  # 已计算的 user intent，已经在 device 上
        )
        rewards = rewards_torch.detach().cpu().numpy()
        
        # ========== 8. 检查重置条件 ==========
        reset_timeout = (current_time - episode_start_time) >= TIMEOUT_SECONDS
        reached = dist <= reset_dist
        collision = np.zeros((num_envs,), dtype=bool)
        for i in range(num_envs):
            if scene_manager.check_boundary_collision(i, np.array([x_local[i], y_local[i]])):
                collision[i] = True
        
        to_reset = reset_timeout | reached | collision
        dones = to_reset.astype(np.float32)
        dones_torch = torch.from_numpy(dones).float()
        
        # ========== 9. 更新统计信息 ==========
        episode_rewards += torch.from_numpy(rewards).float().to(device)
        episode_lengths += 1
        
        # ========== 10. 处理重置和冷却期 ==========
        reset_env_ids = np.nonzero(to_reset)[0]
        
        if len(reset_env_ids) > 0:
            # 标记为 inactive，启动冷却期
            active_mask[reset_env_ids] = False
            cooldown_steps[reset_env_ids] = COOLDOWN_STEPS
            
            # 重置统计信息
            episode_rewards[reset_env_ids] = 0.0
            episode_lengths[reset_env_ids] = 0
            
            # 停止机器人
            robots.set_joint_velocity_targets(np.zeros((num_envs, robots.num_dof), dtype=np.float32))
            v_zero = np.zeros((num_envs, 6), dtype=np.float32)
            robots.set_velocities(v_zero)
            robots.set_joint_velocities(np.zeros((num_envs, robots.num_dof), dtype=np.float32))
            
            # 重置每个环境
            for i in reset_env_ids:
                scene_manager.reset_scene_obstacles(np.array([i]), rng=rng)
                spawn_pos, spawn_yaw = scene_manager.get_robot_spawn_config(i, rng=rng)
                spawn_rot = quat_wxyz_from_yaw(np.array([spawn_yaw], dtype=np.float32))
                
                idx = np.array([i], dtype=np.int32)
                robots.set_world_poses(
                    positions=spawn_pos.reshape(1, 3),
                    orientations=spawn_rot.reshape(1, 4),
                    indices=idx,
                )
                robots.set_velocities(
                    velocities=np.zeros((1, 6), dtype=np.float32),
                    indices=idx,
                )
                robots.set_joint_velocities(
                    velocities=np.zeros((1, robots.num_dof), dtype=np.float32),
                    indices=idx,
                )
                robots.set_joint_velocity_targets(
                    np.zeros((1, robots.num_dof), dtype=np.float32),
                    indices=idx,
                )
                
                goal_pos[i] = scene_manager.get_goal_config(i, rng=rng)
                markers[i].set_world_pose(position=goal_pos[i].tolist(), orientation=[1, 0, 0, 0])
                episode_start_time[i] = current_time
                action_history[i] = 0.0
                prev_measured_v[i] = 0.0
                prev_measured_w[i] = 0.0
            
            # 稳定步骤
            for _ in range(8):
                robots.set_joint_velocity_targets(np.zeros((num_envs, robots.num_dof), dtype=np.float32))
                robots.set_velocities(v_zero)
                robots.set_joint_velocities(np.zeros((num_envs, robots.num_dof), dtype=np.float32))
                world.step(render=True)
        
        # 更新冷却计数器
        cooldown_steps[~active_mask] -= 1
        cooldown_finished = (cooldown_steps <= 0) & (~active_mask)
        if cooldown_finished.any():
            active_mask[cooldown_finished] = True
        
        # ========== 11. 获取下一步观察 ==========
        lidar_ranges = compute_lidar_ranges(
            robot_positions=pos,
            robot_orientations=rot,
            env_origins=env_origins,
            lidar_num_rays=lidar_num_rays,
            lidar_max_range=lidar_max_range,
            scene_manager=scene_manager
        )
        
        robot_velocities_torch = torch.from_numpy(robot_velocities_np).float()
        base_vel_np = get_base_velocity_from_tensor(
            robot_velocities_torch,
            max_v=max_v,
            max_w=max_w
        )
        
        _, user_intent_env, _ = compute_user_intent_torch(
            robot_pos_torch,
            robot_rot_torch,
            goal_pos_torch,
            env_origins_torch,
            normalize=True
        )
        user_intent_np = user_intent_env.cpu().numpy().astype(np.float32)
        
        next_obs = assemble_observations(
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
        next_obs_torch = torch.from_numpy(next_obs).float().to(device)
        
        # ========== 12. 更新动作历史 ==========
        action_history[:, 2:] = action_history[:, :2].copy()
        action_history[:, 0] = prev_measured_v.copy()
        action_history[:, 1] = prev_measured_w.copy()
        prev_measured_v = current_measured_v.copy()
        prev_measured_w = current_measured_w.copy()
        
        # ========== 13. 记录经验到 memory ==========
        rewards_masked = rewards.copy()
        rewards_masked[~active_mask.cpu().numpy()] = 0.0
        dones_masked = dones.copy()
        dones_masked[~active_mask.cpu().numpy()] = 0.0
        dones_masked_torch = torch.from_numpy(dones_masked).float()
        
        # skrl 的 record_transition：记录基本经验
        # 根据 skrl 的标准用法，log_probs 和 values 应该通过 infos 参数传递
        # 这样 record_transition() 会自动将它们存储到 memory 中
        # ========== 13. 记录经验到 memory ==========
        # 先记录基本的环境交互数据
        agent.record_transition(
            states=obs_torch,
            actions=actions,
            rewards=torch.from_numpy(rewards_masked).float().to(device),
            next_states=next_obs_torch,
            terminated=dones_masked_torch.to(device),
            truncated=torch.zeros_like(dones_torch, dtype=torch.bool).to(device),
            infos={},  # 不通过 infos 传递 log_probs 和 values
            timestep=step,  # 当前时间步
            timesteps=TOTAL_STEPS  # 总时间步数
        )
        
        # 然后使用 memory 的 add_samples() 方法显式添加 log_probs 和 values
        # 确保每个环境的数据都正确存储，形状为 (num_envs,)
        if hasattr(agent.agent, 'memory') and agent.agent.memory is not None:
            memory = agent.agent.memory
            if hasattr(memory, 'add_samples') and log_probs is not None and values is not None:
                # 确保 log_probs 和 values 是 tensor，形状为 (num_envs,)
                # 每个环境独立采样，所以需要确保维度正确
                log_probs_tensor = log_probs if isinstance(log_probs, torch.Tensor) else torch.tensor(log_probs, device=device, dtype=torch.float32)
                values_tensor = values if isinstance(values, torch.Tensor) else torch.tensor(values, device=device, dtype=torch.float32)
                
                # 确保形状正确：(num_envs,)
                if log_probs_tensor.dim() > 1:
                    log_probs_tensor = log_probs_tensor.squeeze(-1)
                if values_tensor.dim() > 1:
                    values_tensor = values_tensor.squeeze(-1)
                
                # 确保设备正确
                log_probs_tensor = log_probs_tensor.to(device)
                values_tensor = values_tensor.to(device)
                
                # 存储到 memory（每个环境独立）
                try:
                    memory.add_samples(
                        log_prob=log_probs_tensor,
                        values=values_tensor
                    )
                except Exception as e:
                    print(f"[ERROR] Failed to add_samples to memory: {e}")
                    print(f"  log_probs shape: {log_probs_tensor.shape}, dtype: {log_probs_tensor.dtype}, device: {log_probs_tensor.device}")
                    print(f"  values shape: {values_tensor.shape}, dtype: {values_tensor.dtype}, device: {values_tensor.device}")
                    raise
        
        # ========== 14. 调用 post_interaction（检查是否需要更新）==========
        # skrl PPO 使用 post_interaction 来触发更新
        # 应该在每一步都调用，它会检查 memory 是否满了，如果满了才进行训练
        agent.agent.post_interaction(timestep=step, timesteps=TOTAL_STEPS)
        
        # ========== 15. 每 rollouts 步记录 Tensorboard ==========
        if (step + 1) % rollouts == 0:
            
            # 记录到 Tensorboard（在更新后）
            if active_mask.any():
                avg_reward = float(rewards[active_mask.cpu().numpy()].mean())
                writer.add_scalar("Reward/Average", avg_reward, step)
                avg_episode_reward = episode_rewards[active_mask].mean().item()
                writer.add_scalar("Episode/Reward", avg_episode_reward, step)
                avg_episode_length = episode_lengths[active_mask].float().mean().item()
                writer.add_scalar("Episode/Length", avg_episode_length, step)
            
            reset_rate = dones.mean()
            writer.add_scalar("Episode/ResetRate", reset_rate, step)
            active_rate = active_mask.float().mean().item()
            writer.add_scalar("Environment/ActiveRate", active_rate, step)
        
        # ========== 15. 定期保存 Checkpoint ==========
        if (step + 1) % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_{step+1}.pt")
            agent.save(checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")
        
        # ========== 16. 输出观察（每隔指定时间输出一次，仅环境0）==========
        now = time.time()
        if now - last_obs_print >= OBS_PRINT_INTERVAL:
            obs_env0 = obs[0]
            non_lidar_obs = obs_env0[36:44]
            lidar_first2 = obs_env0[0:2]
            
            print("\n" + "=" * 80)
            print(f"[E0 Observation] step={step}")
            print("-" * 80)
            print(f"Non-LiDAR Obs (8D):")
            print(f"  User Input [ux, uy]:              [{non_lidar_obs[0]:+.4f}, {non_lidar_obs[1]:+.4f}]")
            print(f"  Base Vel   [v, w]:                [{non_lidar_obs[2]:+.4f}, {non_lidar_obs[3]:+.4f}]")
            print(f"  Action Hist [v_t-1, w_t-1, v_t-2, w_t-2]:")
            print(f"                [{non_lidar_obs[4]:+.4f}, {non_lidar_obs[5]:+.4f}, {non_lidar_obs[6]:+.4f}, {non_lidar_obs[7]:+.4f}]")
            print(f"LiDAR (first 2 of 36):")
            print(f"  [lidar[0], lidar[1]]:   [{lidar_first2[0]:+.4f}, {lidar_first2[1]:+.4f}]")
            print(f"Active: {active_mask[0].item()}, Cooldown: {cooldown_steps[0].item()}")
            print("=" * 80 + "\n")
            
            last_obs_print = now
        
        # ========== 17. 调试输出 ==========
        if now - last_debug_print >= DEBUG_PRINT_INTERVAL:
            action_env0 = actions_np[0]
            reward_env0 = rewards[0]
            done_env0 = dones[0]
            
            print(
                f"[E0] step={step} "
                f"action=[{action_env0[0]:+.3f}, {action_env0[1]:+.3f}] "
                f"reward={reward_env0:+.3f} "
                f"done={done_env0} "
                f"active={active_mask[0].item()} "
                f"cooldown={cooldown_steps[0].item()}"
            )
            
            last_debug_print = now
        
        # ========== 18. 更新观察（为下一步准备）==========
        obs_torch = next_obs_torch
        
        # ========== 19. 检查应用是否仍在运行 ==========
        if not simulation_app.is_running():
            print("\n⚠️  SimulationApp 已停止，退出训练循环")
            break
    
    # ==================== 19. 训练结束 ====================
    print("\n" + "=" * 80)
    print("✅ 训练完成！")
    print("=" * 80)
    
    # 保存最终 Checkpoint
    final_checkpoint_path = os.path.join(CHECKPOINT_DIR, "checkpoint_final.pt")
    agent.save(final_checkpoint_path)
    print(f"💾 Final checkpoint saved: {final_checkpoint_path}")
    
    # 关闭 Tensorboard
    writer.close()
    print(f"📊 Tensorboard logs saved to: {TENSORBOARD_DIR}")
    
    # 关闭应用
    simulation_app.close()
    print("👋 应用已关闭")


if __name__ == "__main__":
    main()
