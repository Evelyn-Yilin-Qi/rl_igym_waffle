# scripts/tb3_cpu_4env_random_yaw_random_goal_smooth.py
# 改动目标：同时压“点头”和“转弯困难”
# 关键改动：
#   1) 轮子 max_effort 提升（转弯更有力）
#   2) 轮子 kd 提升 + 更长 ramp（减少点头/速度过冲）
#   3) 角速度限幅降低 + yaw 控制减小（不甩头、更顺）
#   4) 近目标强减速 + “转向优先”抑制（不顶着目标抖）
#   5) physics_dt 提高到 1/120（接触更稳，代价是更慢）
#
# 运行：
#   isaac_python scripts/tb3_cpu_4env_random_yaw_random_goal_smooth.py

import time
import numpy as np

from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.objects import VisualSphere
from omni.isaac.core.utils.stage import add_reference_to_stage


TB3_USD = "/workspace/rl_igym_waffle/assets/turtlebot3_waffle_pi/waffle_pi.usd"


def yaw_from_quat_wxyz(q):
    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(t3, t4)


def quat_wxyz_from_yaw(yaw):
    half = 0.5 * yaw
    q = np.zeros((yaw.shape[0], 4), dtype=np.float32)
    q[:, 0] = np.cos(half)
    q[:, 3] = np.sin(half)
    return q


def wrap_to_pi(a):
    return np.arctan2(np.sin(a), np.cos(a))


def sample_goal_offsets(n, r_min=1.0, r_max=1.5):
    ang = np.random.uniform(-np.pi, np.pi, size=(n,)).astype(np.float32)
    r = (r_min + (r_max - r_min) * np.random.rand(n)).astype(np.float32)
    dx = r * np.cos(ang)
    dy = r * np.sin(ang)
    return np.stack([dx, dy], axis=1)


def main():
    np.random.seed(42)

    # ====== 更稳的物理步进：physics 120Hz, render 60Hz ======
    physics_dt = 1.0 / 120.0
    render_dt = 1.0 / 60.0
    world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=render_dt)
    world.scene.add_default_ground_plane()

    # ====== 4 env layout: 2x2 四角 ======
    num_envs = 4
    env_spacing = 4.0
    env_origins = np.zeros((num_envs, 3), dtype=np.float32)
    for i in range(num_envs):
        ix = i % 2
        iy = i // 2
        env_origins[i, 0] = ix * env_spacing
        env_origins[i, 1] = iy * env_spacing
        env_origins[i, 2] = 0.0

    # ====== load TB3 into each env ======
    for i in range(num_envs):
        tb3_root = f"/World/envs/env_{i}/TB3"
        add_reference_to_stage(usd_path=TB3_USD, prim_path=tb3_root)

    for _ in range(120):
        world.step(render=True)

    # ====== bind articulation view ======
    robots = ArticulationView(
        prim_paths_expr="/World/envs/env_.*/TB3/a__namespace_base_footprint",
        name="tb3_view",
        reset_xform_properties=False
    )
    world.scene.add(robots)
    world.reset()

    # ====== DOF mapping ======
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

    # ====== gains + max_efforts（同时解决：点头&转弯） ======
    # 点头：提高 kd（阻尼）
    # 转弯困难：提高 max_effort（扭矩上限）
    wheel_kp = 0.0
    wheel_kd = 140.0
    wheel_max_effort = 10.0

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

    # ====== spawn pose: env 原点 + 随机 yaw ======
    spawn_pos = env_origins.copy()
    spawn_pos[:, 2] = 0.12

    spawn_yaw = np.random.uniform(-np.pi, np.pi, size=(num_envs,)).astype(np.float32)
    spawn_rot = quat_wxyz_from_yaw(spawn_yaw)

    robots.set_world_poses(spawn_pos, spawn_rot)
    robots.set_velocities(np.zeros((num_envs, 6), dtype=np.float32))
    robots.set_joint_velocities(np.zeros((num_envs, robots.num_dof), dtype=np.float32))

    # ====== goals ======
    goal_offsets = sample_goal_offsets(num_envs, 1.0, 1.5)
    goal_pos = env_origins.copy()
    goal_pos[:, 0] = env_origins[:, 0] + goal_offsets[:, 0]
    goal_pos[:, 1] = env_origins[:, 1] + goal_offsets[:, 1]
    goal_pos[:, 2] = 0.03

    # smaller red markers
    markers = []
    for i in range(num_envs):
        m = world.scene.add(
            VisualSphere(
                prim_path=f"/World/envs/env_{i}/GoalMarker",
                name=f"goal_marker_{i}",
                position=goal_pos[i].tolist(),
                radius=0.03,
                color=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            )
        )
        markers.append(m)

    # ====== controller params ======
    WHEEL_RADIUS = 0.033
    WHEEL_BASE = 0.287

    forward_v = 0.10     # 提速（你要更快就加，但不建议超过 0.15 先）
    k_yaw = 2.0          # 降低（不甩头）
    max_w = 1.6          # 降低（更顺）

    reach_thresh = 0.12

    warmup_steps = int(0.5 / physics_dt)  # 0.5s
    ramp_steps = int(2.0 / physics_dt)    # 2s（更平滑）
    local_step = np.zeros((num_envs,), dtype=np.int64)

    loops = np.zeros((num_envs,), dtype=np.int64)
    last_print = time.time()

    while simulation_app.is_running():
        local_step += 1

        alpha = np.clip((local_step - warmup_steps) / float(max(1, ramp_steps)), 0.0, 1.0).astype(np.float32)
        v_cmd = alpha * forward_v

        pos, rot = robots.get_world_poses()
        yaw = yaw_from_quat_wxyz(rot)

        x_local = (pos[:, 0] - env_origins[:, 0]).astype(np.float32)
        y_local = (pos[:, 1] - env_origins[:, 1]).astype(np.float32)

        gx_local = goal_offsets[:, 0].astype(np.float32)
        gy_local = goal_offsets[:, 1].astype(np.float32)

        ex = gx_local - x_local
        ey = gy_local - y_local
        dist = np.sqrt(ex * ex + ey * ey).astype(np.float32)

        goal_heading = np.arctan2(ey, ex)
        yaw_err = wrap_to_pi(goal_heading - yaw)

        # 角速度命令（更小、更平滑）
        w_cmd = np.clip(k_yaw * yaw_err, -max_w, max_w).astype(np.float32)

        # 近目标强减速（0.05~1.0）
        slow = np.clip(dist / 0.8, 0.05, 1.0).astype(np.float32)
        v_use = v_cmd * slow

        # 转向优先：yaw_err 大就压 v（避免顶着目标抖/拐不动）
        v_use = v_use * (1.0 - np.clip(np.abs(yaw_err) / 1.2, 0.0, 0.7)).astype(np.float32)

        left_w = (v_use - w_cmd * (WHEEL_BASE / 2.0)) / WHEEL_RADIUS
        right_w = (v_use + w_cmd * (WHEEL_BASE / 2.0)) / WHEEL_RADIUS

        targets = np.zeros((num_envs, robots.num_dof), dtype=np.float32)
        targets[:, left_idx] = left_w
        targets[:, right_idx] = right_w
        robots.set_joint_velocity_targets(targets)

        world.step(render=True)

        reached = dist <= reach_thresh
        if np.any(reached):
            ids = np.nonzero(reached)[0]

            robots.set_joint_velocity_targets(np.zeros((num_envs, robots.num_dof), dtype=np.float32))
            v_zero = np.zeros((num_envs, 6), dtype=np.float32)
            robots.set_velocities(v_zero)

            new_yaw = np.random.uniform(-np.pi, np.pi, size=(len(ids),)).astype(np.float32)
            spawn_rot_new = quat_wxyz_from_yaw(new_yaw)

            pos_new = np.array(pos, copy=True)
            rot_new = np.array(rot, copy=True)
            pos_new[ids, :] = spawn_pos[ids, :]
            rot_new[ids, :] = spawn_rot_new

            robots.set_world_poses(pos_new, rot_new)
            robots.set_velocities(v_zero)
            robots.set_joint_velocities(np.zeros((num_envs, robots.num_dof), dtype=np.float32))

            # resample goals
            goal_offsets[ids, :] = sample_goal_offsets(len(ids), 1.0, 1.5)

            # update markers
            for j in ids:
                goal_pos[j, 0] = env_origins[j, 0] + goal_offsets[j, 0]
                goal_pos[j, 1] = env_origins[j, 1] + goal_offsets[j, 1]
                goal_pos[j, 2] = 0.03
                markers[j].set_world_pose(position=goal_pos[j].tolist(), orientation=[1, 0, 0, 0])

            local_step[ids] = 0
            loops[ids] += 1

        now = time.time()
        if now - last_print > 0.5:
            print(
                f"env0 dist={dist[0]:.2f} loops={loops[0]} | "
                f"env1 dist={dist[1]:.2f} loops={loops[1]} | "
                f"env2 dist={dist[2]:.2f} loops={loops[2]} | "
                f"env3 dist={dist[3]:.2f} loops={loops[3]}"
            )
            last_print = now

        time.sleep(0.001)

    simulation_app.close()


if __name__ == "__main__":
    main()