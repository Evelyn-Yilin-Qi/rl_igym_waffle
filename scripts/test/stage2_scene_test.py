"""
Stage2 场景加载测试脚本
纯场景测试：加载4种不同场景，TB3小车简单向前走
用于验证Stage2场景配置是否正确
"""
import os
import sys
import time
import numpy as np
from omni.isaac.kit import SimulationApp

# 配置项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 初始化Isaac Sim
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.physics_context import PhysicsContext
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.objects import VisualSphere
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
from omegaconf import OmegaConf

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
    compute_lidar_ranges,
    check_obstacle_collision,
)

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

# ==================== 主函数 ====================
def main():
    # 加载配置
    env_cfg = load_config("cfg/WaffleDrive.yaml")
    
    # 配置参数
    num_envs = 32  # 16个环境，每个场景各4个（EMPTY, CYLINDER, DOOR, BOX各4个）
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
    
    # 碰撞模型参数
    DCOL = 0.2  # 碰撞阈值（20cm）
    DCRIT = 0.5  # 临界阈值（50cm）
    
    # 简单控制参数：只向前走
    simple_v = 0.3  # 固定线速度
    simple_w = 0.0  # 固定角速度（直走）
    
    # 随机数初始化
    seed = 0
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    
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
    for i, scene_type in enumerate(scene_types):
        print(f"环境 {i}: {scene_type}")
    
    # 对于非EMPTY场景，应该显示外墙和障碍物
    # 如果传入None，create_scene_obstacles会自动根据场景类型决定
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
    
    # 稳定步骤
    for _ in range(8):
        robots.set_joint_velocity_targets(np.zeros((num_envs, robots.num_dof), dtype=np.float32))
        robots.set_velocities(np.zeros((num_envs, 6), dtype=np.float32))
        robots.set_joint_velocities(np.zeros((num_envs, robots.num_dof), dtype=np.float32))
        world.step(render=True)
    
    # 主循环
    current_time = 0.0
    step_count = 0
    
    print(f"\n=== 开始测试 ===")
    print(f"控制模式：固定线速度 {simple_v} m/s，角速度 {simple_w} rad/s（直走）")
    
    while simulation_app.is_running():
        current_time += physics_dt
        step_count += 1
        
        # 获取机器人状态
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
        
        # ==================== LiDAR数据采集（用于碰撞检测） ====================
        lidar_ranges = compute_lidar_ranges(
            robot_positions=pos,
            robot_orientations=rot,
            env_origins=env_origins,
            lidar_num_rays=lidar_num_rays,
            lidar_max_range=lidar_max_range,
            scene_manager=scene_manager
        )
        
        # ==================== 障碍物碰撞检测 ====================
        obstacle_collision, critical_flag, min_dist = check_obstacle_collision(lidar_ranges, DCOL, DCRIT)
        
        # ==================== 简单控制：固定向前走 ====================
        v_cmd = np.ones(num_envs, dtype=np.float32) * simple_v
        w_cmd = np.ones(num_envs, dtype=np.float32) * simple_w
        
        # 碰撞时强制停止
        v_cmd[obstacle_collision] = 0.0
        w_cmd[obstacle_collision] = 0.0
        
        # ==================== 动作下发 ====================
        targets = np.zeros((num_envs, robots.num_dof), dtype=np.float32)
        for i in range(num_envs):
            action = diff_ctrl.forward(command=np.array([float(v_cmd[i]), float(w_cmd[i])], dtype=np.float32))
            targets[i, left_idx] = float(action.joint_velocities[0])
            targets[i, right_idx] = float(action.joint_velocities[1])
        
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
        
        # ==================== 打印状态信息 ====================
        if step_count % 100 == 0:  # 每100步打印一次
            print(f"\n=== Step {step_count} ===")
            for i in range(num_envs):
                print(f"环境 {i} ({scene_types[i]}): "
                      f"距离={dist[i]:.2f}m, "
                      f"碰撞={obstacle_collision[i]}, "
                      f"边界={boundary_collision[i]}, "
                      f"超时={reset_timeout[i]}, "
                      f"到达={reached[i]}, "
                      f"重置={to_reset[i]}")
        
        # ==================== 机器人重置 ====================
        if np.any(to_reset):
            reset_ids = np.nonzero(to_reset)[0]
            
            print(f"\n[重置] 环境 {reset_ids.tolist()} 触发重置")
            for rid in reset_ids:
                print(f"  - 环境 {rid} ({scene_types[rid]}): "
                      f"障碍物碰撞={obstacle_collision[rid]}, "
                      f"边界碰撞={boundary_collision[rid]}, "
                      f"超时={reset_timeout[rid]}, "
                      f"到达目标={reached[rid]}")
            
            # ==================== 处理根节点速度 ====================
            current_base_vels = robots.get_velocities()
            current_base_vels[reset_ids] = 0.0
            robots.set_velocities(current_base_vels)

            # ==================== 处理关节实时速度 ====================
            current_joint_vels = robots.get_joint_velocities()
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
                
                # 重置目标位置
                goal_pos[i] = scene_manager.get_goal_config(i, rng=rng)
                markers[i].set_world_pose(position=goal_pos[i].tolist(), orientation=[1, 0, 0, 0])
                
                # 重置计时
                episode_start_time[i] = current_time
            
            # 稳定步骤
            for _ in range(8):
                robots.set_joint_velocity_targets(np.zeros((num_envs, robots.num_dof), dtype=np.float32))
                robots.set_velocities(np.zeros((num_envs, 6), dtype=np.float32))
                robots.set_joint_velocities(np.zeros((num_envs, robots.num_dof), dtype=np.float32))
                world.step(render=True)
        
        time.sleep(0.001)
    
    simulation_app.close()

if __name__ == "__main__":
    main()
