"""
TB3 4环境测试脚本 - 模块化版本
基于 sim 模块和配置文件，实现与 10_env&TB3_reset_move_Gpt.py 相同的效果
"""
import os
import sys
import time
import numpy as np
from omegaconf import OmegaConf

from omni.isaac.kit import SimulationApp

# Ensure project root is on PYTHONPATH so local modules (sim, envs) can be imported
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

# 导入 sim 模块
from sim.robot import (
    TB3_USD, WHEEL_RADIUS, WHEEL_BASE,
    apply_massapi_all_tb3, configure_wheel_joints
)
from sim.scenes import (
    SCENE_EMPTY, SCENE_BOX, SCENE_CYLINDER, SCENE_DOOR,
    SceneManager, yaw_from_quat_wxyz, quat_wxyz_from_yaw, wrap_to_pi
)
from sim.robot.tb3_config import ROBOT_HALF_LENGTH_X, ROBOT_HALF_WIDTH_Y, ROBOT_COLLISION_RADIUS


def load_config(cfg_path="cfg/task/WaffleDrive.yaml"):
    """从配置文件加载参数，支持相对/绝对路径多重回退"""
    candidates = []
    if os.path.isabs(cfg_path):
        candidates.append(cfg_path)
    else:
        candidates.extend([
            os.path.join(PROJECT_ROOT, cfg_path),  # 项目根目录
            os.path.abspath(cfg_path),             # 运行目录
            os.path.join(os.getcwd(), cfg_path),   # 显式 cwd
        ])
    for path in candidates:
        if os.path.exists(path):
            return OmegaConf.load(path)
    raise FileNotFoundError(f"Config file not found. Tried: {candidates}")


def main():
    # 加载配置
    cfg = load_config()
    
    # 从配置读取参数
    num_envs = int(cfg.env.numEnvs)
    env_size = float(cfg.env.scene.env_size)
    # 环境间隔计算：env_spacing = env_size + env_gap（与原始代码保持一致）
    env_gap = 2.0  # 环境之间的间隙（固定值，与原始代码一致）
    env_spacing = env_size + env_gap  # 总间隔 = 6.0 + 2.0 = 8.0
    wall_thickness = float(cfg.env.scene.wall_thickness)
    wall_height = float(cfg.env.scene.wall_height)
    
    # 物理参数
    physics_dt = float(cfg.sim.dt)
    render_dt = 1.0 / 60.0
    
    # 控制参数（从原代码中提取，可以后续移到配置）
    k_v = 0.8
    k_yaw = 3.5
    v_max = 0.26
    w_max_drive = 1.6
    w_max_turn = 1.2
    reach_thresh = 0.18
    TIMEOUT_SECONDS = 60.0
    
    # 状态机参数
    ENTER_TURN = 55.0 * np.pi / 180.0
    EXIT_TURN = 25.0 * np.pi / 180.0
    ENTER_HOLD = 3
    EXIT_HOLD = 10
    
    # 初始化随机数生成器
    rng = np.random.default_rng(42)
    np.random.seed(42)
    
    # ==================== 1. 创建 World 和物理环境 ====================
    world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=render_dt)
    PhysicsContext().substeps = 8
    world.scene.add_default_ground_plane()
    stage = world.scene.stage
    
    # ==================== 2. 计算环境原点 ====================
    env_gap = env_spacing - env_size
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
        max_linear_speed=0.26,
        max_angular_speed=1.6,
    )
    
    # ==================== 8. 创建场景管理器 ====================
    scene_types = [SCENE_EMPTY, SCENE_BOX, SCENE_CYLINDER, SCENE_DOOR]
    show_visual_walls = [False, True, True, True]  # 从配置读取或根据场景类型决定
    
    scene_manager = SceneManager(
        num_envs=num_envs,
        env_size=env_size,
        env_origins=env_origins,
        stage=stage
    )
    scene_manager.wall_thickness = wall_thickness
    scene_manager.wall_height = wall_height
    
    # 创建场景障碍物
    scene_manager.create_scene_obstacles(
        scene_types=scene_types,
        show_visual_walls=show_visual_walls
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
    
    # ==================== 10. 初始化状态变量 ====================
    episode_start_time = np.zeros((num_envs,), dtype=np.float32)
    loops = np.zeros((num_envs,), dtype=np.int64)
    COOLDOWN = int(0.7 / physics_dt)
    cooldown_steps = np.zeros((num_envs,), dtype=np.int64)
    SETTLE_STEPS = 8
    
    V_DRIVE_MIN = 0.07
    warmup_steps = int(0.1 / physics_dt)
    ramp_steps = int(1.0 / physics_dt)
    local_step = np.zeros((num_envs,), dtype=np.int64)
    
    # 卡住检测参数
    STUCK_STEPS = int(1.2 / physics_dt)
    DRIVE_CMD_EPS = 2.0
    DRIVE_V_EPS = 0.03
    DRIVE_DIST_EPS = 0.003
    TURN_CMD_EPS = 2.0
    TURN_WZ_EPS = 0.03
    TURN_YAW_EPS = 0.01
    
    no_prog_cnt = np.zeros((num_envs,), dtype=np.int64)
    last_dist = np.full((num_envs,), np.inf, dtype=np.float32)
    last_abs_yaw = np.full((num_envs,), np.inf, dtype=np.float32)
    
    enter_cnt = np.zeros((num_envs,), dtype=np.int64)
    exit_cnt = np.zeros((num_envs,), dtype=np.int64)
    turn_state = np.zeros((num_envs,), dtype=bool)
    
    did_reset = np.zeros((num_envs,), dtype=bool)
    
    # ==================== 11. 初始重置 ====================
    for i in range(num_envs):
        # 重置场景障碍物
        scene_manager.reset_scene_obstacles([i], rng=rng)
        
        # 获取机器人初始位置和朝向
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
        
        # 更新目标位置
        goal_pos[i] = scene_manager.get_goal_config(i, rng=rng)
        markers[i].set_world_pose(position=goal_pos[i].tolist(), orientation=[1, 0, 0, 0])
        
        # 初始化状态
        episode_start_time[i] = 0.0
        cooldown_steps[i] = COOLDOWN
        turn_state[i] = False
        enter_cnt[i] = 0
        exit_cnt[i] = 0
        no_prog_cnt[i] = 0
        last_dist[i] = np.inf
        last_abs_yaw[i] = np.inf
    
    # ==================== 12. 主循环 ====================
    current_time = 0.0
    last_print = time.time()
    
    while simulation_app.is_running():
        did_reset[:] = False
        
        current_time += physics_dt
        local_step += 1
        
        cooldown_mask = cooldown_steps > 0
        cooldown_steps[cooldown_mask] -= 1
        
        alpha = np.clip((local_step - warmup_steps) / float(max(1, ramp_steps)), 0.0, 1.0).astype(np.float32)
        
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
        abs_yaw = np.abs(yaw_err)
        
        # 转向状态机（N帧确认）
        want_enter = abs_yaw >= ENTER_TURN
        want_exit = abs_yaw <= EXIT_TURN
        enter_cnt = np.where(want_enter, enter_cnt + 1, 0).astype(np.int64)
        exit_cnt = np.where(want_exit, exit_cnt + 1, 0).astype(np.int64)
        enter_now = (~turn_state) & (enter_cnt >= ENTER_HOLD)
        exit_now = (turn_state) & (exit_cnt >= EXIT_HOLD)
        turn_state[enter_now] = True
        turn_state[exit_now] = False
        
        # 计算速度命令
        v_nom = np.clip(k_v * dist, 0.0, v_max).astype(np.float32)
        turn_slow = np.clip(1.0 - (abs_yaw / np.pi), 0.2, 1.0).astype(np.float32)
        v_drive = (alpha * v_nom * turn_slow).astype(np.float32)
        
        v_use = np.where(turn_state, 0.0, v_drive).astype(np.float32)
        
        w_cap = np.where(turn_state, w_max_turn, w_max_drive).astype(np.float32)
        w_cmd = (alpha * np.clip(k_yaw * yaw_err, -w_cap, w_cap)).astype(np.float32)
        
        v_use = np.where(cooldown_mask, 0.0, v_use).astype(np.float32)
        w_cmd = np.where(cooldown_mask, 0.0, w_cmd).astype(np.float32)
        
        # 驱动模式防单轮卡死约束
        drive_mask = (~turn_state) & (~cooldown_mask)
        w_limit_no_stall = (2.0 * np.maximum(v_use - V_DRIVE_MIN, 0.0) / WHEEL_BASE).astype(np.float32)
        w_cmd = np.where(drive_mask, np.clip(w_cmd, -w_limit_no_stall, w_limit_no_stall), w_cmd).astype(np.float32)
        w_cmd = np.where(drive_mask & (v_use < V_DRIVE_MIN), 0.0, w_cmd).astype(np.float32)
        
        # 转换为轮速
        targets = np.zeros((num_envs, robots.num_dof), dtype=np.float32)
        for i in range(num_envs):
            if cooldown_steps[i] > 0:
                wl = 0.0
                wr = 0.0
            else:
                action = diff_ctrl.forward(command=np.array([float(v_use[i]), float(w_cmd[i])], dtype=np.float32))
                wl = float(action.joint_velocities[0])
                wr = float(action.joint_velocities[1])
            
            targets[i, left_idx] = wl
            targets[i, right_idx] = wr
        
        robots.set_joint_velocity_targets(targets)
        world.step(render=True)
        
        # 步后终止条件检查
        pos2, _ = robots.get_world_poses()
        x_local2 = (pos2[:, 0] - env_origins[:, 0]).astype(np.float32)
        y_local2 = (pos2[:, 1] - env_origins[:, 1]).astype(np.float32)
        ex2 = gx_local - x_local2
        ey2 = gy_local - y_local2
        dist2 = np.sqrt(ex2 * ex2 + ey2 * ey2).astype(np.float32)
        
        reached = dist2 <= reach_thresh
        collision = np.zeros((num_envs,), dtype=bool)
        timeout = np.zeros((num_envs,), dtype=bool)
        
        for i in range(num_envs):
            if scene_manager.check_boundary_collision(i, np.array([x_local2[i], y_local2[i]])):
                collision[i] = True
            elif scene_manager.check_collision_with_obstacles(i, np.array([x_local2[i], y_local2[i]])):
                collision[i] = True
            
            if current_time - episode_start_time[i] >= TIMEOUT_SECONDS:
                timeout[i] = True
        
        base_v = robots.get_velocities()
        base_wz = base_v[:, 5].astype(np.float32)
        
        cmd_mag = np.maximum(np.abs(targets[:, left_idx]), np.abs(targets[:, right_idx])).astype(np.float32)
        
        # 进度指标
        dist_improve = (last_dist - dist2).astype(np.float32)
        yaw_improve = (last_abs_yaw - abs_yaw).astype(np.float32)
        
        last_dist = dist2.copy()
        last_abs_yaw = abs_yaw.copy()
        
        # 卡住检测
        drive_stuck = (
            (~turn_state)
            & (~cooldown_mask)
            & (cmd_mag > DRIVE_CMD_EPS)
            & (v_use > DRIVE_V_EPS)
            & (dist_improve < DRIVE_DIST_EPS)
            & (~reached)
            & (~collision)
            & (~timeout)
        )
        
        turn_stuck = (
            (turn_state)
            & (~cooldown_mask)
            & (cmd_mag > TURN_CMD_EPS)
            & (yaw_improve < TURN_YAW_EPS)
            & (np.abs(base_wz) < TURN_WZ_EPS)
            & (~reached)
            & (~collision)
            & (~timeout)
        )
        
        stuck_now = drive_stuck | turn_stuck
        no_prog_cnt = np.where(stuck_now, no_prog_cnt + 1, 0).astype(np.int64)
        stuck_reset = no_prog_cnt >= STUCK_STEPS
        
        # 重置逻辑
        to_reset = reached | collision | timeout | stuck_reset
        if np.any(to_reset):
            ids = np.nonzero(to_reset)[0]
            
            robots.set_joint_velocity_targets(np.zeros((num_envs, robots.num_dof), dtype=np.float32))
            v_zero = np.zeros((num_envs, 6), dtype=np.float32)
            robots.set_velocities(v_zero)
            robots.set_joint_velocities(np.zeros((num_envs, robots.num_dof), dtype=np.float32))
            
            for i in ids:
                # 重置场景障碍物
                scene_manager.reset_scene_obstacles([i], rng=rng)
                
                # 重置机器人位置
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
                
                # 重置目标位置
                goal_pos[i] = scene_manager.get_goal_config(i, rng=rng)
                markers[i].set_world_pose(position=goal_pos[i].tolist(), orientation=[1, 0, 0, 0])
                
                # 重置状态
                episode_start_time[i] = current_time
                local_step[i] = 0
                loops[i] += 1
                cooldown_steps[i] = COOLDOWN
                turn_state[i] = False
                enter_cnt[i] = 0
                exit_cnt[i] = 0
                no_prog_cnt[i] = 0
                last_dist[i] = np.inf
                last_abs_yaw[i] = np.inf
                
                did_reset[i] = True
            
            # 稳定步骤
            for _ in range(SETTLE_STEPS):
                robots.set_joint_velocity_targets(np.zeros((num_envs, robots.num_dof), dtype=np.float32))
                robots.set_velocities(v_zero)
                robots.set_joint_velocities(np.zeros((num_envs, robots.num_dof), dtype=np.float32))
                world.step(render=True)
        
        # 调试输出（仅环境0）
        now = time.time()
        if now - last_print > 0.35:
            dof_vel = robots.get_joint_velocities()
            l_act = float(dof_vel[0, left_idx])
            r_act = float(dof_vel[0, right_idx])
            
            base_v0 = robots.get_velocities()
            vx0, vy0, vz0, wx0, wy0, wz0 = [float(x) for x in base_v0[0]]
            
            if did_reset[0]:
                l_cmd = 0.0
                r_cmd = 0.0
            else:
                l_cmd = float(targets[0, left_idx])
                r_cmd = float(targets[0, right_idx])
            
            errL = l_act - l_cmd
            errR = r_act - r_cmd
            
            print(
                f"[E0] scene={scene_types[0]} loop={int(loops[0])} cd={int(cooldown_steps[0])} "
                f"dist={float(dist2[0]):.3f} yaw_err={float(yaw_err[0]):+.3f} abs={float(abs_yaw[0]):.3f} "
                f"turn={bool(turn_state[0])} alpha={float(alpha[0]):.2f} "
                f"v={float(v_use[0]):.3f} w={float(w_cmd[0]):+.3f} "
                f"cmd(L,R)=({l_cmd:+.2f},{r_cmd:+.2f}) act(L,R)=({l_act:+.2f},{r_act:+.2f}) "
                f"err(L,R)=({errL:+.2f},{errR:+.2f}) "
                f"base(vx,vy,wz)=({vx0:+.3f},{vy0:+.3f},{wz0:+.3f}) "
                f"no_prog_cnt={int(no_prog_cnt[0])} "
                f"reset(goal,coll,timeout,stuck)=({bool(reached[0])},{bool(collision[0])},{bool(timeout[0])},{bool(stuck_reset[0])})"
            )
            last_print = now
        
        time.sleep(0.001)
    
    simulation_app.close()


if __name__ == "__main__":
    main()
