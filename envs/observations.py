"""
Observation Space Assembly
组装 44 维观察向量：LiDAR[36] + User Input[2] + Base Velocity[2] + Action History[4]

User Input（obs[36:38]）的工程统一定义：
- 来自 user_intent_ego（自车坐标系）
- ux: 小车正前方为正方向（forward +X）
- uy: 小车左侧90度为正方向（left +Y）
"""
import numpy as np
import torch
from .user_intent import compute_user_intent_torch

# 延迟导入Isaac Sim相关模块（避免在非Isaac Sim环境中导入失败）
try:
    from omni.physx import get_physx_scene_query_interface
    from pxr import Gf
    ISAAC_SIM_AVAILABLE = True
except ImportError:
    ISAAC_SIM_AVAILABLE = False
    get_physx_scene_query_interface = None
    Gf = None


def assemble_observations(
    robot_positions: np.ndarray,
    robot_orientations: np.ndarray,
    goal_positions: np.ndarray,
    env_origins: np.ndarray,
    lidar_ranges: np.ndarray,
    base_vel: np.ndarray,
    action_history: np.ndarray,
    max_v: float = 0.5,
    max_w: float = 1.0,
    lidar_max_range: float = 3.0,
    user_input: np.ndarray = None  # 可选：如果已计算则直接使用，避免重复计算
) -> np.ndarray:
    """
    组装 44 维观察向量
    
    Args:
        robot_positions: (N, 3) 机器人位置
        robot_orientations: (N, 4) 机器人朝向（四元数）
        goal_positions: (N, 3) 目标位置
        env_origins: (N, 3) 环境原点
        lidar_ranges: (N, 36) LiDAR 距离测量值
        base_vel: (N, 2) 归一化的测量基础速度 [v_norm, w_norm]
                  v: 测量的线速度，w: 测量的角速度
        action_history: (N, 4) 最近 2 个动作 [v_t-1, w_t-1, v_t-2, w_t-2]
        max_v: 最大线速度（用于归一化）
        max_w: 最大角速度（用于归一化）
        lidar_max_range: LiDAR 最大测量范围（用于归一化）
    
    Returns:
        obs: (N, 44) 观察向量
            - [0:36]: LiDAR 归一化距离 [0, 1]
            - [36:38]: User Input（user_intent_ego）单位向量 (ux, uy) ∈ [-1, 1]
                      ux: 车前方向分量，uy: 车左方向分量
            - [38:40]: Base Velocity 归一化 (v_norm, w_norm) ∈ [-1, 1]
                      测量的当前线速度 v 和角速度 w
            - [40:44]: Action History 归一化 (v_t-1, w_t-1, v_t-2, w_t-2) ∈ [-1, 1]
    """
    num_envs = lidar_ranges.shape[0]
    
    # 1. LiDAR 归一化 [0, 1]
    lidar_normalized = np.clip(lidar_ranges / lidar_max_range, 0.0, 1.0).astype(np.float32)
    
    # 2. User Input（user_intent_ego）单位向量 (ux, uy) ∈ [-1, 1]
    if user_input is None:
        # 如果未传入，则内部计算
        _, user_input_unit, _ = compute_user_intent_torch(
            torch.from_numpy(robot_positions),
            torch.from_numpy(robot_orientations),
            torch.from_numpy(goal_positions),
            torch.from_numpy(env_origins),
            normalize=True
        )
        user_input = user_input_unit.numpy().astype(np.float32)  # (N, 2)
    else:
        # 使用传入的 user_input（确保是 float32）
        user_input = np.asarray(user_input, dtype=np.float32)
    
    # 3. Base Velocity 归一化（已传入，直接使用）
    # base_vel 已经是归一化的 [v_norm, w_norm]，测量的线速度 v 和角速度 w
    base_vel_norm = base_vel.copy()
    
    # 4. Action History 归一化 [v_t-1, w_t-1, v_t-2, w_t-2]
    action_history_normalized = action_history.copy()
    action_history_normalized[:, 0] /= max_v  # v_t-1 归一化
    action_history_normalized[:, 1] /= max_w   # w_t-1 归一化
    action_history_normalized[:, 2] /= max_v  # v_t-2 归一化
    action_history_normalized[:, 3] /= max_w   # w_t-2 归一化
    action_history_normalized = np.clip(action_history_normalized, -1.0, 1.0).astype(np.float32)
    
    # 拼接所有观察
    obs = np.concatenate([
        lidar_normalized,      # (N, 36)
        user_input,            # (N, 2)
        base_vel_norm,         # (N, 2)
        action_history_normalized  # (N, 4) -> Total 44维
    ], axis=1).astype(np.float32)
    
    return obs


def get_base_velocity_from_tensor(velocities: torch.Tensor, max_v: float = 0.5, max_w: float = 1.0) -> np.ndarray:
    """
    从速度张量获取归一化的测量基础速度
    
    Base Velocity 是测量的当前线速度和角速度，用于作为 RL 模型的输入观察
    TB3 差速驱动机器人只有线速度 v 和角速度 w，没有 vx/vy 的概念
    
    Args:
        velocities: (N, 6) 速度张量 [vx, vy, vz, wx, wy, wz]
                   对于 TB3：vx 是线速度 v（机器人局部坐标系 x 方向），vy=0，wz 是角速度 w
        max_v: 最大线速度（用于归一化）
        max_w: 最大角速度（用于归一化）
    
    Returns:
        base_vel: (N, 2) 归一化的测量基础速度 [v_norm, w_norm]
                  v_norm: 归一化的测量线速度 ∈ [-1, 1]
                  w_norm: 归一化的测量角速度 ∈ [-1, 1]
    """
    # 提取测量的线速度 v 和角速度 w
    # 对于 TB3：vx 就是线速度 v（机器人局部坐标系 x 方向是前进方向）
    # wz 就是角速度 w（绕 z 轴旋转）
    v = velocities[:, 0].cpu().numpy()  # 线速度
    w = velocities[:, 5].cpu().numpy()  # 角速度
    
    # 归一化
    v_norm = np.clip(v / max_v, -1.0, 1.0)
    w_norm = np.clip(w / max_w, -1.0, 1.0)
    
    base_vel = np.stack([v_norm, w_norm], axis=1).astype(np.float32)
    return base_vel


def compute_lidar_ranges(robot_positions, robot_orientations, env_origins, 
                         lidar_num_rays, lidar_max_range, scene_manager):
    """
    计算LiDAR距离（射线检测）
    
    Args:
        robot_positions: (num_envs, 3) 机器人位置
        robot_orientations: (num_envs, 4) 机器人朝向（四元数）
        env_origins: (num_envs, 3) 环境原点
        lidar_num_rays: LiDAR射线数量
        lidar_max_range: LiDAR最大测量范围
        scene_manager: 场景管理器（用于边界碰撞检测）
    
    Returns:
        lidar_ranges: (num_envs, lidar_num_rays) LiDAR距离测量值
    """
    if not ISAAC_SIM_AVAILABLE:
        raise ImportError("Isaac Sim modules not available. Cannot compute LiDAR ranges.")
    
    num_envs = robot_positions.shape[0]
    lidar_ranges = np.ones((num_envs, lidar_num_rays), dtype=np.float32) * lidar_max_range
    
    physx_interface = get_physx_scene_query_interface()
    lidar_height = 0.1  # LiDAR安装高度
    
    for env_id in range(num_envs):
        robot_pos = robot_positions[env_id]
        robot_rot = robot_orientations[env_id]
        env_origin = env_origins[env_id]
        
        robot_pos_local = np.array([
            robot_pos[0] - env_origin[0],
            robot_pos[1] - env_origin[1]
        ])
        
        # 提取偏航角
        w, x, y, z = robot_rot
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        
        # 生成LiDAR射线角度
        angles = np.linspace(0, 2 * np.pi, lidar_num_rays, endpoint=False)
        
        for ray_idx, angle in enumerate(angles):
            # 局部射线方向
            ray_dir_local = np.array([np.cos(angle), np.sin(angle), 0.0])
            
            # 转换到世界坐标系
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            ray_dir_world = np.array([
                ray_dir_local[0] * cos_yaw - ray_dir_local[1] * sin_yaw,
                ray_dir_local[0] * sin_yaw + ray_dir_local[1] * cos_yaw,
                0.0
            ])
            
            # 射线起点和方向
            ray_origin = Gf.Vec3f(float(robot_pos[0]), float(robot_pos[1]), float(robot_pos[2] + lidar_height))
            ray_dir_norm = np.linalg.norm(ray_dir_world)
            ray_dir_normalized = ray_dir_world / ray_dir_norm if ray_dir_norm > 1e-6 else np.array([1.0, 0.0, 0.0])
            ray_dir = Gf.Vec3f(*ray_dir_normalized)
            
            # 射线检测
            hit = physx_interface.raycast_closest(ray_origin, ray_dir, float(lidar_max_range), False)
            if hit:
                distance = hit.get('distance', lidar_max_range)
                lidar_ranges[env_id, ray_idx] = min(float(distance), lidar_max_range)
            else:
                # 检查边界碰撞
                ray_end_local = robot_pos_local + ray_dir_local[:2] * lidar_max_range
                if scene_manager.check_boundary_collision(env_id, ray_end_local):
                    env_half = scene_manager.env_size * 0.5
                    dist_to_boundary = lidar_max_range
                    
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
    
    return lidar_ranges


def check_obstacle_collision(lidar_ranges, dcol=0.2, dcrit=0.5):
    """
    检查障碍物碰撞（胶囊形模型）
    
    Args:
        lidar_ranges: (num_envs, lidar_num_rays) LiDAR测距数据
        dcol: 碰撞阈值（m），d < dcol 表示已发生碰撞
        dcrit: 临界阈值（m），dcol < dcrit，d < dcrit 表示有碰撞风险
    
    Returns:
        collision_flag: (num_envs,) 是否已碰撞（d < dcol）
        critical_flag: (num_envs,) 是否有碰撞风险（d < dcrit）
        min_dist: (num_envs,) 最近障碍物距离（胶囊形模型修正后）
    """
    num_envs = lidar_ranges.shape[0]
    # 初始化标志
    collision_flag = np.zeros(num_envs, dtype=bool)
    critical_flag = np.zeros(num_envs, dtype=bool)
    
    # 计算全局最近距离
    min_dist = np.min(lidar_ranges, axis=1)  
    
    # 胶囊形模型修正：前进方向距离修正
    forward_rays = lidar_ranges[:, :int(lidar_ranges.shape[1]/4)]  # 前90°射线
    forward_min_dist = np.min(forward_rays, axis=1)
    min_dist = np.minimum(min_dist, forward_min_dist * 1.2)  # 长度修正
    
    # 正确区分：dcol=碰撞，dcrit=临界风险
    collision_flag = min_dist < dcol       # 真正碰撞（距离极近）
    critical_flag = min_dist < dcrit       # 临界风险（需要减速/避障）
    
    return collision_flag, critical_flag, min_dist


def get_obs(robots, env_origins, goal_pos, lidar_num_rays, lidar_max_range, scene_manager, 
            max_v, max_w, action_history, device=None):
    """
    获取观察向量（完整流程）
    
    Args:
        robots: ArticulationView 机器人视图
        env_origins: (num_envs, 3) 环境原点
        goal_pos: (num_envs, 3) 目标位置
        lidar_num_rays: LiDAR射线数量
        lidar_max_range: LiDAR最大测量范围
        scene_manager: 场景管理器
        max_v: 最大线速度
        max_w: 最大角速度
        action_history: (num_envs, 4) 动作历史
        device: torch设备（可选，用于GPU加速）
    
    Returns:
        obs: (num_envs, 44) 观察向量
    """
    # 延迟导入避免循环依赖
    from sim.scenes import yaw_from_quat_wxyz
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pos, rot = robots.get_world_poses()
    yaw = yaw_from_quat_wxyz(rot)
    
    # 局部坐标计算
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
    
    # 3. 当前测量速度（用于其他用途，不返回）
    # current_measured_v = robot_velocities_np[:, 0]
    # current_measured_w = robot_velocities_np[:, 5]
    
    # 4. 用户意图（自车坐标系）：ux=前进方向，uy=左转方向
    # 张量移到GPU
    robot_pos_torch = torch.from_numpy(pos).float().to(device)
    robot_rot_torch = torch.from_numpy(rot).float().to(device)
    goal_pos_torch = torch.from_numpy(goal_pos).float().to(device)
    env_origins_torch = torch.from_numpy(env_origins).float().to(device)
    _, user_intent_ego, _ = compute_user_intent_torch(
        robot_pos_torch, torch.tensor(yaw), goal_pos_torch, env_origins_torch, normalize=True
    )
    user_intent_np = user_intent_ego.cpu().numpy().astype(np.float32)  # 转回CPU；这是模型实际输入
    
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
    return obs
