"""
RL Stage 1 测试脚本 - 带观察模块
4个empty场景，每个TB3机器人朝随机目标行走，独立reset
加入观察模块，保持40Hz观察频率，每隔1秒输出env0的观察
"""
import os
import sys
import time
import numpy as np
import torch
from omegaconf import OmegaConf

from omni.isaac.kit import SimulationApp

# Ensure project root is on PYTHONPATH so that local modules (e.g., sim) can be imported
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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


def load_config(cfg_path="cfg/task/WaffleDrive.yaml"):
    """从配置文件加载参数"""
    # 将相对路径解析为项目根目录下的路径；若未找到则尝试常见位置
    candidates = []
    if os.path.isabs(cfg_path):
        candidates.append(cfg_path)
    else:
        candidates.extend([
            os.path.join(PROJECT_ROOT, cfg_path),      # 项目根目录
            os.path.abspath(cfg_path),                 # 运行时工作目录
            os.path.join(os.getcwd(), cfg_path),       # 显式拼当前工作目录
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
            # raycast_closest 需要归一化的方向向量，不是终点
            ray_dir_norm = np.linalg.norm(ray_dir_world)
            if ray_dir_norm > 1e-6:
                ray_dir_normalized = ray_dir_world / ray_dir_norm
            else:
                # 如果方向向量为零，使用默认方向（向前）
                ray_dir_normalized = np.array([1.0, 0.0, 0.0])
            
            ray_dir = Gf.Vec3f(
                float(ray_dir_normalized[0]),
                float(ray_dir_normalized[1]),
                float(ray_dir_normalized[2])
            )
            
            # 执行 ray cast
            # raycast_closest(origin, dir, distance, bothSides=False)
            hit = physx_interface.raycast_closest(
                ray_origin, 
                ray_dir, 
                float(lidar_max_range),
                bothSides=False
            )
            
            if hit:
                # hit 是一个字典，包含 'distance' 键
                distance = hit.get('distance', lidar_max_range)
                lidar_ranges[env_id, ray_idx] = min(float(distance), lidar_max_range)
            else:
                # 检查是否与边界墙碰撞（使用场景管理器）
                # 计算射线在局部坐标系中的终点
                ray_end_local = robot_pos_local + ray_dir_local[:2] * lidar_max_range
                
                # 检查是否超出边界
                if scene_manager.check_boundary_collision(env_id, ray_end_local):
                    # 计算到边界的距离
                    env_half = scene_manager.env_size * 0.5
                    # 计算射线与边界的交点
                    dist_to_boundary = lidar_max_range
                    
                    # 检查四个边界
                    if ray_dir_local[0] > 0:  # 向右
                        t = (env_half - robot_pos_local[0]) / ray_dir_local[0] if ray_dir_local[0] > 1e-6 else lidar_max_range
                        if 0 < t < dist_to_boundary:
                            dist_to_boundary = t
                    elif ray_dir_local[0] < 0:  # 向左
                        t = (-env_half - robot_pos_local[0]) / ray_dir_local[0] if ray_dir_local[0] < -1e-6 else lidar_max_range
                        if 0 < t < dist_to_boundary:
                            dist_to_boundary = t
                    
                    if ray_dir_local[1] > 0:  # 向前
                        t = (env_half - robot_pos_local[1]) / ray_dir_local[1] if ray_dir_local[1] > 1e-6 else lidar_max_range
                        if 0 < t < dist_to_boundary:
                            dist_to_boundary = t
                    elif ray_dir_local[1] < 0:  # 向后
                        t = (-env_half - robot_pos_local[1]) / ray_dir_local[1] if ray_dir_local[1] < -1e-6 else lidar_max_range
                        if 0 < t < dist_to_boundary:
                            dist_to_boundary = t
                    
                    lidar_ranges[env_id, ray_idx] = min(dist_to_boundary, lidar_max_range)
                else:
                    # 没有碰撞，使用最大范围
                    lidar_ranges[env_id, ray_idx] = lidar_max_range
    
    return lidar_ranges


def main():
    # ==================== 加载配置 ====================
    cfg = load_config()
    
    # ==================== 从配置读取环境参数 ====================
    num_envs = int(cfg.env.numEnvs)
    env_size = float(cfg.env.scene.env_size)
    # 环境间隔计算：env_spacing = env_size + env_gap（与原始代码保持一致）
    env_gap = 2.0  # 环境之间的间隙（固定值，与原始代码一致）
    env_spacing = env_size + env_gap  # 总间隔 = 6.0 + 2.0 = 8.0
    reset_dist = float(cfg.env.resetDist)  # 使用配置中的 resetDist
    wall_thickness = float(cfg.env.scene.wall_thickness)
    wall_height = float(cfg.env.scene.wall_height)
    show_visual_walls = bool(cfg.env.scene.show_visual_walls)
    
    # 超时时间（秒），与原始代码保持一致
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
    
    # ==================== 初始化随机数生成器 ====================
    rng = np.random.default_rng(42)
    np.random.seed(42)
    
    # ==================== 1. 创建 World 和物理环境 ====================
    world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=render_dt)
    PhysicsContext().substeps = 8
    world.scene.add_default_ground_plane()
    stage = world.scene.stage
    
    # ==================== 2. 计算环境原点（使用配置的 envSpacing）====================
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
    
    # ==================== 5. 配置机器人物理属性（使用 sim 模块）====================
    apply_massapi_all_tb3()
    for _ in range(10):
        world.step(render=True)
    
    # ==================== 6. 配置轮子关节（使用 sim 模块）====================
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
    
    # ==================== 8. 创建场景管理器（所有环境都是empty，使用 sim 模块）====================
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
    
    # 创建场景障碍物（使用 SceneManager）
    scene_manager.create_scene_obstacles(
        scene_types=scene_types,
        show_visual_walls=show_visual_walls_list
    )
    
    # ==================== 9. 初始化目标位置和标记 ====================
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
    
    # ==================== 10. 初始重置所有环境（使用 SceneManager）====================
    # 初始化 episode 开始时间
    episode_start_time = np.zeros((num_envs,), dtype=np.float32)
    
    # 初始化动作历史（用于观察）：[v_t-1, w_t-1, v_t-2, w_t-2]
    # Action History 存储的是历史时刻的测量速度，不是命令动作
    action_history = np.zeros((num_envs, 4), dtype=np.float32)
    
    # 初始化上一时刻的测量速度（用于下次观察时更新 action_history）
    prev_measured_v = np.zeros((num_envs,), dtype=np.float32)
    prev_measured_w = np.zeros((num_envs,), dtype=np.float32)
    
    for i in range(num_envs):
        # 获取机器人初始位置和朝向（使用 SceneManager）
        spawn_pos, spawn_yaw = scene_manager.get_robot_spawn_config(i, rng=rng)
        spawn_rot = quat_wxyz_from_yaw(np.array([spawn_yaw], dtype=np.float32))
        
        # 设置机器人位置
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
        
        # 初始化 episode 开始时间
        episode_start_time[i] = 0.0
    
    # ==================== 11. 主循环 ====================
    # 简单的控制参数（测试用，后续可移到配置）
    k_v = 0.8
    k_yaw = 3.5
    v_max_cmd = 0.26  # 实际命令速度限制
    
    # 仿真时间跟踪
    current_time = 0.0
    
    # 观察输出控制（每隔1秒输出一次）
    last_obs_print = time.time()
    
    # 调试输出控制（每0.5秒输出一次）
    last_debug_print = time.time()
    
    while simulation_app.is_running():
        # 更新仿真时间
        current_time += physics_dt
        
        # 获取机器人状态
        pos, rot = robots.get_world_poses()
        yaw = yaw_from_quat_wxyz(rot)
        
        # 计算局部位置
        x_local = (pos[:, 0] - env_origins[:, 0]).astype(np.float32)
        y_local = (pos[:, 1] - env_origins[:, 1]).astype(np.float32)
        
        # 计算目标偏移（相对于环境原点）
        goal_offsets = goal_pos[:, :2] - env_origins[:, :2]
        gx_local = goal_offsets[:, 0].astype(np.float32)
        gy_local = goal_offsets[:, 1].astype(np.float32)
        
        # 计算到目标的距离和方向
        ex = gx_local - x_local
        ey = gy_local - y_local
        dist = np.sqrt(ex * ex + ey * ey).astype(np.float32)
        
        goal_heading = np.arctan2(ey, ex)
        yaw_err = wrap_to_pi(goal_heading - yaw).astype(np.float32)
        
        # 简单的速度控制
        v_cmd = np.clip(k_v * dist, 0.0, v_max_cmd).astype(np.float32)
        w_cmd = np.clip(k_yaw * yaw_err, -max_w, max_w).astype(np.float32)
        
        # ==================== 计算观察（40Hz，每步都计算）====================
        # 1. 计算 LiDAR 范围（使用真实的 ray casting）
        lidar_ranges = compute_lidar_ranges(
            robot_positions=pos,
            robot_orientations=rot,
            env_origins=env_origins,
            lidar_num_rays=lidar_num_rays,
            lidar_max_range=lidar_max_range,
            scene_manager=scene_manager
        )
        
        # 2. 获取测量的基础速度（归一化）
        # Base Velocity 是测量的当前线速度和角速度 (v, w)
        # robots.get_velocities() 返回 numpy 数组 (N, 6) [vx, vy, vz, wx, wy, wz]
        # 对于 TB3：vx 是线速度 v，vy=0，wz 是角速度 w
        robot_velocities_np = robots.get_velocities()
        robot_velocities_torch = torch.from_numpy(robot_velocities_np).float()
        base_vel_np = get_base_velocity_from_tensor(
            robot_velocities_torch,
            max_v=max_v,
            max_w=max_w
        )
        
        # 获取当前时刻的测量速度（物理值，未归一化）
        # 用于更新 action_history（存储历史测量速度）
        current_measured_v = robot_velocities_np[:, 0]  # 线速度 v
        current_measured_w = robot_velocities_np[:, 5]  # 角速度 w
        
        # 3. 计算用户意图向量
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
        user_intent_np = user_intent_env.numpy().astype(np.float32)
        
        # 4. 组装观察向量
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
        
        # 转换为轮速（使用 DifferentialController）
        targets = np.zeros((num_envs, robots.num_dof), dtype=np.float32)
        for i in range(num_envs):
            action = diff_ctrl.forward(command=np.array([float(v_cmd[i]), float(w_cmd[i])], dtype=np.float32))
            targets[i, left_idx] = float(action.joint_velocities[0])
            targets[i, right_idx] = float(action.joint_velocities[1])
        
        robots.set_joint_velocity_targets(targets)
        world.step(render=True)
        
        # 更新动作历史（存储历史时刻的测量速度，不是命令动作）
        # Action History 存储的是 t-1 和 t-2 时刻的测量速度
        # 在计算观察时，当前时刻的测量速度已经获取，现在更新历史
        action_history[:, 2:] = action_history[:, :2].copy()  # 移动历史：t-2 = 旧的 t-1
        action_history[:, 0] = prev_measured_v.copy()  # v_t-1：上一时刻的测量线速度
        action_history[:, 1] = prev_measured_w.copy()  # w_t-1：上一时刻的测量角速度
        
        # 更新上一时刻的测量速度（用于下次观察）
        prev_measured_v = current_measured_v.copy()
        prev_measured_w = current_measured_w.copy()
        
        # ==================== 检查重置条件（对齐 igym_waffle_env.py 的逻辑）====================
        # 1. 超时（使用时间而不是步数，与原始代码保持一致）
        reset_timeout = (current_time - episode_start_time) >= TIMEOUT_SECONDS
        
        # 2. 到达目标（距离 < resetDist，使用配置中的 resetDist）
        reached = dist <= reset_dist
        
        # 3. 边界碰撞（使用 SceneManager）
        collision = np.zeros((num_envs,), dtype=bool)
        for i in range(num_envs):
            if scene_manager.check_boundary_collision(i, np.array([x_local[i], y_local[i]])):
                collision[i] = True
        
        # 设置重置标志（任一条件满足即重置）
        to_reset = reset_timeout | reached | collision
        
        if np.any(to_reset):
            ids = np.nonzero(to_reset)[0]
            
            # 停止所有机器人
            robots.set_joint_velocity_targets(np.zeros((num_envs, robots.num_dof), dtype=np.float32))
            v_zero = np.zeros((num_envs, 6), dtype=np.float32)
            robots.set_velocities(v_zero)
            robots.set_joint_velocities(np.zeros((num_envs, robots.num_dof), dtype=np.float32))
            
            # 重置每个环境（使用 SceneManager）
            for i in ids:
                # 重置场景障碍物（使用 SceneManager）
                scene_manager.reset_scene_obstacles(np.array([i]), rng=rng)
                
                # 重置机器人位置（使用 SceneManager）
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
                
                # 重置目标位置（使用 SceneManager）
                goal_pos[i] = scene_manager.get_goal_config(i, rng=rng)
                markers[i].set_world_pose(position=goal_pos[i].tolist(), orientation=[1, 0, 0, 0])
                
                # 重置 episode 开始时间
                episode_start_time[i] = current_time
                
                # 重置动作历史
                action_history[i] = 0.0
                # 重置上一时刻的测量速度
                prev_measured_v[i] = 0.0
                prev_measured_w[i] = 0.0
            
            # 稳定步骤
            for _ in range(8):
                robots.set_joint_velocity_targets(np.zeros((num_envs, robots.num_dof), dtype=np.float32))
                robots.set_velocities(v_zero)
                robots.set_joint_velocities(np.zeros((num_envs, robots.num_dof), dtype=np.float32))
                world.step(render=True)
        
        # ==================== 输出观察（每隔1秒输出一次，仅环境0）====================
        now = time.time()
        if now - last_obs_print >= 1.0:
            # 观察向量结构：[0:36] LiDAR, [36:38] User Input, [38:40] Base Velocity, [40:44] Action History
            obs_env0 = obs[0]
            
            # 除了雷达之外的其他观察（8维）
            non_lidar_obs = obs_env0[36:44]  # User Input[2] + Base Velocity[2] + Action History[4]
            
            # 雷达前2个观察值
            lidar_first2 = obs_env0[0:2]
            
            print("\n" + "=" * 80)
            print(f"[E0 Observation] t={current_time:.2f}s")
            print("-" * 80)
            print(f"Non-LiDAR Obs (8D):")
            print(f"  User Input [ux, uy]:              [{non_lidar_obs[0]:+.4f}, {non_lidar_obs[1]:+.4f}]")
            print(f"  Base Vel   [v, w]:                [{non_lidar_obs[2]:+.4f}, {non_lidar_obs[3]:+.4f}]")
            print(f"  Action Hist [v_t-1, w_t-1, v_t-2, w_t-2]:")
            print(f"                [{non_lidar_obs[4]:+.4f}, {non_lidar_obs[5]:+.4f}, {non_lidar_obs[6]:+.4f}, {non_lidar_obs[7]:+.4f}]")
            print(f"LiDAR (first 2 of 36):")
            print(f"  [lidar[0], lidar[1]]:   [{lidar_first2[0]:+.4f}, {lidar_first2[1]:+.4f}]")
            print("=" * 80 + "\n")
            
            last_obs_print = now
        
        # 调试输出（每0.5秒打印一次，仅环境0）
        if now - last_debug_print > 0.5:
            elapsed_time = current_time - episode_start_time[0]
            print(
                f"[E0] dist={float(dist[0]):.3f} yaw_err={float(yaw_err[0]):+.3f} "
                f"v={float(v_cmd[0]):.3f} w={float(w_cmd[0]):+.3f} "
                f"time={float(elapsed_time):.1f}/{TIMEOUT_SECONDS:.1f}s "
                f"reset(timeout,goal,collision)=({bool(reset_timeout[0])},{bool(reached[0])},{bool(collision[0])})"
            )
            last_debug_print = now
        
        time.sleep(0.001)
    
    simulation_app.close()


if __name__ == "__main__":
    main()
