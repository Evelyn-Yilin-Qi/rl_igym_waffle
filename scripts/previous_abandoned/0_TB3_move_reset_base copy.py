# scripts/4_tb3_cpu_4env_random_fixed_com_v2.py
# 目标：
#   - CPU pipeline（稳定 + UI 可见）
#   - 4 个 env（2x2 四角排列，互不影响、各自坐标系=各自 env 原点）
#   - 每个 env：小车初始位置=env 原点，朝向随机（yaw 随机）
#   - 每个 env：目标点=以 env 原点为圆心，任意方向，距离随机 [1.0, 1.5] m
#   - 小车自动朝目标走，到达阈值后：只 reset 自己那台车（回 env 原点 + 随机 yaw），同时刷新自己的目标点
#   - 目标点可视化：更小、红色
#   - 【优化】进一步优化质心(COM)参数，解决全速时的前倾问题
#
# 运行：
#   isaac_python scripts/4_tb3_cpu_4env_random_fixed_com_v2.py

import time
import numpy as np

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.objects import VisualSphere
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.physics_context import PhysicsContext
from omni.isaac.core.prims import RigidPrimView

TB3_USD = "/workspace/rl_igym_waffle/assets/turtlebot3_waffle_pi/waffle_pi.usd"


def yaw_from_quat_wxyz(q):
    # q: (N,4) [w,x,y,z] -> yaw
    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(t3, t4)


def quat_wxyz_from_yaw(yaw):
    # yaw: (N,) -> (N,4) wxyz
    half = 0.5 * yaw
    q = np.zeros((yaw.shape[0], 4), dtype=np.float32)
    q[:, 0] = np.cos(half)
    q[:, 3] = np.sin(half)
    return q


def wrap_to_pi(a):
    return np.arctan2(np.sin(a), np.cos(a))


def sample_goal_offsets(n, r_min=1.0, r_max=1.5):
    # 返回 (n,2): 任意方向，距离 [r_min, r_max]
    ang = np.random.uniform(-np.pi, np.pi, size=(n,)).astype(np.float32)
    r = (r_min + (r_max - r_min) * np.random.rand(n)).astype(np.float32)
    dx = r * np.cos(ang)
    dy = r * np.sin(ang)
    return np.stack([dx, dy], axis=1)


def main():
    np.random.seed(42)

    # 更稳的物理步进（抑制点头/接触抖动）
    physics_dt = 1.0 / 240.0
    render_dt = 1.0 / 60.0
    world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=render_dt)
    phys_context = PhysicsContext()
    phys_context.substeps = 8
    world.scene.add_default_ground_plane()

    # 4 env：2x2 四角
    num_envs = 4
    env_spacing = 4.0
    env_origins = np.zeros((num_envs, 3), dtype=np.float32)
    for i in range(num_envs):
        ix = i % 2
        iy = i // 2
        env_origins[i, 0] = ix * env_spacing
        env_origins[i, 1] = iy * env_spacing
        env_origins[i, 2] = 0.0

    # 加载 4 台 TB3
    for i in range(num_envs):
        tb3_root = f"/World/envs/env_{i}/TB3"
        add_reference_to_stage(usd_path=TB3_USD, prim_path=tb3_root)

    # 让引用展开
    for _ in range(180):
        world.step(render=True)

    # 绑定 articulation view
    # 若你资产结构不同，把 base_footprint 改成 base_link
    robots = ArticulationView(
        prim_paths_expr="/World/envs/env_.*/TB3/a__namespace_base_footprint",
        name="tb3_view",
        reset_xform_properties=False
    )
    world.scene.add(robots)

    # 【新增】绑定 base_link 的 RigidPrimView，用于设置质量和质心
    # 参考 tb3_4env_com_stable.py 的做法
    base_view = RigidPrimView(
        prim_paths_expr="/World/envs/env_.*/TB3/a__namespace_base_link",
        name="base_view",
        reset_xform_properties=False,
    )
    world.scene.add(base_view)

    world.reset()

    if robots.count != num_envs:
        print(f"[WARN] robots.count={robots.count}, expected={num_envs}. 可能 prim_paths_expr 没匹配上。")

    # DOF mapping
    dof_names = robots.dof_names
    print("\n========== DOF NAMES ==========")
    for i, n in enumerate(dof_names):
        print(f"{i:02d}: {n}")
    print("================================\n")

    left_idx = None
    right_idx = None
    for i, n in enumerate(dof_names):
        s = n.lower()
        if left_idx is None and "wheel_left_joint" in s:
            left_idx = i
        if right_idx is None and "wheel_right_joint" in s:
            right_idx = i
    if left_idx is None or right_idx is None:
        raise RuntimeError("没找到 wheel_left_joint / wheel_right_joint，请按上面 DOF NAMES 改匹配")

    # 轮子 velocity drive：kp=0，只调 kd，并限制 max_effort（抑制点头 + 保持可转弯）
    # 提高 kd 防止轮子打滑（轮子转但车不走）
    wheel_kp = 0.0
    wheel_kd = 260.0  # 提高阻尼，确保轮子能有效传递力矩到地面
    wheel_max_effort = 8.0

    kps, kds = robots.get_gains()
    kps[:, left_idx] = wheel_kp
    kps[:, right_idx] = wheel_kp
    kds[:, left_idx] = wheel_kd
    kds[:, right_idx] = wheel_kd
    robots.set_gains(kps=kps, kds=kds)

    try:
        max_eff = robots.get_max_efforts()
        if max_eff is not None:
            max_eff[:, left_idx] = wheel_max_effort
            max_eff[:, right_idx] = wheel_max_effort
            robots.set_max_efforts(max_eff)
    except Exception as e:
        print("[WARN] set_max_efforts failed:", e)

    # 【优化】进一步优化 COM 下压和后移（解决全速时的前倾问题）
    # 相比 v1 版本（com_z=-0.08, com_x=-0.06），进一步：
    # - 质心向下偏移增加到 10cm（降低重心）
    # - 质心向后偏移增加到 8cm（防止前倾）
    base_mass = 2.2  # 质量（kg），保持与 v1 一致
    com_z = -0.10    # 质心向下偏移 10cm（v1: -0.08，进一步降低重心）
    com_x = -0.08    # 质心向后偏移 8cm（v1: -0.06，进一步防止前倾）
    
    # 确保数据是 CPU numpy array（CPU pipeline 要求）
    masses = np.ones(num_envs, dtype=np.float32) * base_mass
    coms = np.zeros((num_envs, 3), dtype=np.float32)
    coms[:, 0] = com_x  # X 轴：向后偏移
    coms[:, 2] = com_z  # Z 轴：向下偏移
    
    try:
        base_view.set_masses(masses)
        base_view.set_coms(coms)
        print(f"[INFO] 已设置底盘质量={base_mass}kg, COM_x={com_x}m, COM_z={com_z}m")
        print(f"[INFO] 相比 v1: COM_x 从 -0.06 增加到 -0.08, COM_z 从 -0.08 增加到 -0.10")
    except Exception as e:
        print(f"[WARN] 设置质量和质心失败: {e}")
        print("[WARN] 可能 base_link 路径不匹配，请检查 USD 资产结构")

    # 初始位姿：env 原点 + 随机 yaw（抬高一点，减少落地冲击）
    spawn_pos = env_origins.copy()
    spawn_pos[:, 2] = 0.0350

    spawn_yaw = np.random.uniform(-np.pi, np.pi, size=(num_envs,)).astype(np.float32)
    spawn_rot = quat_wxyz_from_yaw(spawn_yaw)

    robots.set_world_poses(spawn_pos, spawn_rot)
    robots.set_velocities(np.zeros((num_envs, 6), dtype=np.float32))
    robots.set_joint_velocities(np.zeros((num_envs, robots.num_dof), dtype=np.float32))

    # 目标点：以 env 原点为圆心，任意方向 [1.0, 1.5]m
    goal_offsets = sample_goal_offsets(num_envs, 1.0, 1.5)  # (N,2)
    goal_pos = env_origins.copy()
    goal_pos[:, 0] = env_origins[:, 0] + goal_offsets[:, 0]
    goal_pos[:, 1] = env_origins[:, 1] + goal_offsets[:, 1]
    goal_pos[:, 2] = 0.03

    # 红色小 marker
    markers = []
    for i in range(num_envs):
        m = world.scene.add(
            VisualSphere(
                prim_path=f"/World/envs/env_{i}/GoalMarker",
                name=f"goal_marker_{i}",
                position=goal_pos[i].tolist(),
                radius=0.02,
                color=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            )
        )
        markers.append(m)

    # 控制参数（兼顾转弯与丝滑）
    WHEEL_RADIUS = 0.033
    WHEEL_BASE = 0.287

    forward_v = 0.26  # 速度（TB3最大速度）
    k_yaw = 2.0  # 转向强度（温和）
    max_w = 1.6  # 角速度限幅（不甩头）

    reach_thresh = 0.18  # 到达判定阈值（从 0.12m 增加到 0.18m，更容易触发 reset）

    # 快速启动（缩短启动时间，提高响应速度）
    warmup_steps = int(0.1 / physics_dt)  # 0.1s 预热（从 1.0s 缩短）
    ramp_steps = int(1.0 / physics_dt)  # 1.0s 加速（从 4.0s 缩短）
    local_step = np.zeros((num_envs,), dtype=np.int64)

    loops = np.zeros((num_envs,), dtype=np.int64)
    last_print = time.time()

    while simulation_app.is_running():
        local_step += 1

        v_cmd_raw = forward_v
        # ramp v
        alpha = np.clip((local_step - warmup_steps) / float(max(1, ramp_steps)), 0.0, 1.0).astype(np.float32)
        v_cmd = alpha * v_cmd_raw

        # 读位姿（当前帧）
        pos, rot = robots.get_world_poses()
        yaw = yaw_from_quat_wxyz(rot)

        # 独立坐标系：每个 env 都以 env_origin 为原点算误差
        x_local = (pos[:, 0] - env_origins[:, 0]).astype(np.float32)
        y_local = (pos[:, 1] - env_origins[:, 1]).astype(np.float32)

        gx_local = goal_offsets[:, 0].astype(np.float32)
        gy_local = goal_offsets[:, 1].astype(np.float32)

        ex = gx_local - x_local
        ey = gy_local - y_local
        dist = np.sqrt(ex * ex + ey * ey).astype(np.float32)

        goal_heading = np.arctan2(ey, ex)
        yaw_err = wrap_to_pi(goal_heading - yaw)

        # 不使用距离和角度误差减速，保持恒定速度
        v_use = v_cmd

        # 【关键修改】根据当前速度动态限制角速度，确保两个轮子都向前转
        # 数学约束：left_w > 0 => v_use > w_cmd * WHEEL_BASE/2
        # 因此：|w_cmd| < 2 * v_use / WHEEL_BASE
        # 这样可以确保小车在转向的同时也能前进，形成曲线轨迹而不是原地转向
        max_w_dynamic = 2.0 * v_use / WHEEL_BASE
        max_w_effective = np.minimum(max_w, max_w_dynamic).astype(np.float32)

        # 角速度命令（使用动态限制）
        w_cmd = np.clip(k_yaw * yaw_err, -max_w_effective, max_w_effective).astype(np.float32)

        # 差速 -> 左右轮角速度
        left_w = (v_use - w_cmd * (WHEEL_BASE / 2.0)) / WHEEL_RADIUS
        right_w = (v_use + w_cmd * (WHEEL_BASE / 2.0)) / WHEEL_RADIUS

        # 【安全保护】防止轮子速度过大或反向导致卡死
        # 确保两个轮子都向前转，且速度在合理范围内
        max_wheel_vel = 15.0  # 最大轮子角速度（rad/s），防止极端情况
        min_wheel_vel = 0.0   # 最小轮子角速度，确保不反向
        
        # 如果计算出的轮子速度异常，强制限制
        left_w = np.clip(left_w, min_wheel_vel, max_wheel_vel).astype(np.float32)
        right_w = np.clip(right_w, min_wheel_vel, max_wheel_vel).astype(np.float32)
        
        # 额外检查：如果两个轮子速度差异过大，可能是计算错误，进行修正
        wheel_diff = np.abs(left_w - right_w)
        wheel_sum = np.abs(left_w) + np.abs(right_w)
        # 如果差异超过总和的一定比例，说明可能有问题，降低角速度
        problematic = wheel_diff > 0.5 * wheel_sum
        if np.any(problematic):
            # 对于有问题的环境，降低角速度重新计算
            w_cmd[problematic] = w_cmd[problematic] * 0.5
            left_w[problematic] = (v_use[problematic] - w_cmd[problematic] * (WHEEL_BASE / 2.0)) / WHEEL_RADIUS
            right_w[problematic] = (v_use[problematic] + w_cmd[problematic] * (WHEEL_BASE / 2.0)) / WHEEL_RADIUS
            left_w[problematic] = np.clip(left_w[problematic], min_wheel_vel, max_wheel_vel).astype(np.float32)
            right_w[problematic] = np.clip(right_w[problematic], min_wheel_vel, max_wheel_vel).astype(np.float32)

        targets = np.zeros((num_envs, robots.num_dof), dtype=np.float32)
        targets[:, left_idx] = left_w
        targets[:, right_idx] = right_w
        robots.set_joint_velocity_targets(targets)

        world.step(render=True)

        # step 后再读一次（用新状态判定到达）
        pos2, rot2 = robots.get_world_poses()
        x_local2 = (pos2[:, 0] - env_origins[:, 0]).astype(np.float32)
        y_local2 = (pos2[:, 1] - env_origins[:, 1]).astype(np.float32)
        ex2 = gx_local - x_local2
        ey2 = gy_local - y_local2
        dist2 = np.sqrt(ex2 * ex2 + ey2 * ey2).astype(np.float32)

        reached = dist2 <= reach_thresh
        if np.any(reached):
            ids = np.nonzero(reached)[0]

            # 先把系统能量清掉（线/角速度 + 关节速度）
            robots.set_joint_velocity_targets(np.zeros((num_envs, robots.num_dof), dtype=np.float32))
            v_zero = np.zeros((num_envs, 6), dtype=np.float32)
            robots.set_velocities(v_zero)

            # 这些 env 随机新 yaw
            new_yaw = np.random.uniform(-np.pi, np.pi, size=(len(ids),)).astype(np.float32)
            spawn_rot_new = quat_wxyz_from_yaw(new_yaw)

            # 全量写回 pose（版本更稳，避免 indices 差异）
            pos_new = np.array(pos2, copy=True)
            rot_new = np.array(rot2, copy=True)
            pos_new[ids, :] = spawn_pos[ids, :]
            rot_new[ids, :] = spawn_rot_new

            robots.set_world_poses(pos_new, rot_new)
            robots.set_velocities(v_zero)
            robots.set_joint_velocities(np.zeros((num_envs, robots.num_dof), dtype=np.float32))

            # 刷新目标
            goal_offsets[ids, :] = sample_goal_offsets(len(ids), 1.0, 1.5)

            # 更新 marker
            for j in ids:
                goal_pos[j, 0] = env_origins[j, 0] + goal_offsets[j, 0]
                goal_pos[j, 1] = env_origins[j, 1] + goal_offsets[j, 1]
                goal_pos[j, 2] = 0.03
                markers[j].set_world_pose(position=goal_pos[j].tolist(), orientation=[1, 0, 0, 0])

            local_step[ids] = 0
            loops[ids] += 1

        now = time.time()
        if now - last_print > 0.6:
            print(
                f"env0 dist={dist2[0]:.2f} loops={loops[0]} | "
                f"env1 dist={dist2[1]:.2f} loops={loops[1]} | "
                f"env2 dist={dist2[2]:.2f} loops={loops[2]} | "
                f"env3 dist={dist2[3]:.2f} loops={loops[3]} | "
                f"COM_x={com_x:.3f}m COM_z={com_z:.3f}m mass={base_mass:.1f}kg"
            )
            last_print = now

        time.sleep(0.001)

    simulation_app.close()


if __name__ == "__main__":
    main()
