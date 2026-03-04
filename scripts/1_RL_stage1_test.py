"""
RL Stage 1 测试脚本 - 启动文件
4个empty场景，每个TB3机器人朝随机目标行走，独立reset
最大化复用已有模块和配置，不堆积参数
"""
import time
import numpy as np
from omegaconf import OmegaConf

from omni.isaac.kit import SimulationApp

# 初始化 Isaac Sim
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.physics_context import PhysicsContext
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.objects import VisualSphere
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController

# 导入 sim 模块
from sim.robot import (
    TB3_USD, WHEEL_RADIUS, WHEEL_BASE,
    apply_massapi_all_tb3, configure_wheel_joints
)
from sim.scenes import (
    SCENE_EMPTY,
    SceneManager, yaw_from_quat_wxyz, quat_wxyz_from_yaw, wrap_to_pi
)


def load_config(cfg_path="cfg/task/WaffleDrive.yaml"):
    """从配置文件加载参数"""
    cfg = OmegaConf.load(cfg_path)
    return cfg


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
    physics_dt = float(cfg.sim.dt)
    render_dt = 1.0 / 60.0
    
    # ==================== 从配置读取机器人限制 ====================
    max_v = float(cfg.env.robot_limits.max_v)
    max_w = float(cfg.env.robot_limits.max_w)
    
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
    
    last_print = time.time()
    
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
        
        # 转换为轮速（使用 DifferentialController）
        targets = np.zeros((num_envs, robots.num_dof), dtype=np.float32)
        for i in range(num_envs):
            action = diff_ctrl.forward(command=np.array([float(v_cmd[i]), float(w_cmd[i])], dtype=np.float32))
            targets[i, left_idx] = float(action.joint_velocities[0])
            targets[i, right_idx] = float(action.joint_velocities[1])
        
        robots.set_joint_velocity_targets(targets)
        world.step(render=True)
        
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
            
            # 稳定步骤
            for _ in range(8):
                robots.set_joint_velocity_targets(np.zeros((num_envs, robots.num_dof), dtype=np.float32))
                robots.set_velocities(v_zero)
                robots.set_joint_velocities(np.zeros((num_envs, robots.num_dof), dtype=np.float32))
                world.step(render=True)
        
        # 调试输出（每0.5秒打印一次，仅环境0）
        now = time.time()
        if now - last_print > 0.5:
            elapsed_time = current_time - episode_start_time[0]
            print(
                f"[E0] dist={float(dist[0]):.3f} yaw_err={float(yaw_err[0]):+.3f} "
                f"v={float(v_cmd[0]):.3f} w={float(w_cmd[0]):+.3f} "
                f"time={float(elapsed_time):.1f}/{TIMEOUT_SECONDS:.1f}s "
                f"reset(timeout,goal,collision)=({bool(reset_timeout[0])},{bool(reached[0])},{bool(collision[0])})"
            )
            last_print = now
        
        time.sleep(0.001)
    
    simulation_app.close()


if __name__ == "__main__":
    main()
