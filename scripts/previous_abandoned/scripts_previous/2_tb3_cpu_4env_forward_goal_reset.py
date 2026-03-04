# scripts/tb3_cpu_4env_forward_goal_reset.py
# 目标：
#   - CPU pipeline（稳定 + UI 可见）
#   - 4 个 env（2x2 四角排列，互不影响）
#   - 每个 env：goal 在正前方 +x，距离随机 [1.0, 2.0] 米
#   - 到达 goal 后：只 reset 自己那台车回起点，并重新采样自己的 goal，循环
#   - 可视化：每个 env 一个目标点 marker（每次 reset 同步刷新）
#
# 运行：
#   isaac_python scripts/tb3_cpu_4env_forward_goal_reset.py

import time
import numpy as np

from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.objects import VisualSphere
from omni.isaac.core.utils.stage import add_reference_to_stage


TB3_USD = "/workspace/rl_igym_waffle/assets/turtlebot3_waffle_pi/waffle_pi.usd"


def main():
    np.random.seed(42)

    # CPU world
    dt = 1.0 / 60.0
    world = World(stage_units_in_meters=1.0, physics_dt=dt, rendering_dt=dt)
    world.scene.add_default_ground_plane()

    # ====== 4 env spawn layout: 2x2 四角 ======
    num_envs = 4
    env_spacing = 4.0
    env_origins = np.zeros((num_envs, 3), dtype=np.float32)
    for i in range(num_envs):
        ix = i % 2
        iy = i // 2
        env_origins[i, 0] = ix * env_spacing
        env_origins[i, 1] = iy * env_spacing
        env_origins[i, 2] = 0.0

    # ====== load 4 TB3 ======
    for i in range(num_envs):
        tb3_root = f"/World/envs/env_{i}/TB3"
        add_reference_to_stage(usd_path=TB3_USD, prim_path=tb3_root)

    # 让引用展开
    for _ in range(60):
        world.step(render=True)

    # ====== bind articulation view ======
    # 如果你资产结构不同，把 base_footprint 改成 base_link
    robots = ArticulationView(
        prim_paths_expr="/World/envs/env_.*/TB3/a__namespace_base_footprint",
        name="tb3_view",
        reset_xform_properties=False
    )
    world.scene.add(robots)
    world.reset()

    if robots.count != num_envs:
        print(f"[WARN] robots.count={robots.count}, expected={num_envs}. 可能 prim_paths_expr 没匹配上。")

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

    # ====== gains + max_efforts（压前倾的保守参数） ======
    wheel_kp = 0.0
    wheel_kd = 80.0
    wheel_max_effort = 4.0

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

    # ====== spawn pose for each env ======
    spawn_pos = env_origins.copy()
    spawn_pos[:, 2] = 0.12  # reset 抬高一点，减少落地冲击

    spawn_rot = np.zeros((num_envs, 4), dtype=np.float32)  # wxyz
    spawn_rot[:, 0] = 1.0

    robots.set_world_poses(spawn_pos, spawn_rot)
    robots.set_velocities(np.zeros((num_envs, 6), dtype=np.float32))
    robots.set_joint_velocities(np.zeros((num_envs, robots.num_dof), dtype=np.float32))

    # ====== per-env goal distance in [1,2] & marker ======
    goal_dist = 1.0 + np.random.rand(num_envs).astype(np.float32) * 1.0  # [1,2]
    start_x = spawn_pos[:, 0].copy()

    goal_pos = env_origins.copy()
    goal_pos[:, 0] = env_origins[:, 0] + goal_dist
    goal_pos[:, 2] = 0.05

    markers = []
    for i in range(num_envs):
        m = world.scene.add(
            VisualSphere(
                prim_path=f"/World/envs/env_{i}/GoalMarker",
                name=f"goal_marker_{i}",
                position=goal_pos[i].tolist(),
                radius=0.07,
            )
        )
        markers.append(m)

    # ====== control params ======
    WHEEL_RADIUS = 0.033
    forward_v = 0.02
    wheel_w_max = forward_v / WHEEL_RADIUS

    warmup_steps = int(1.0 / dt)  # 1s
    ramp_steps = int(4.0 / dt)    # 4s
    local_step = np.zeros((num_envs,), dtype=np.int64)

    loops = np.zeros((num_envs,), dtype=np.int64)
    last_print = time.time()

    while simulation_app.is_running():
        local_step += 1

        alpha = np.clip((local_step - warmup_steps) / float(max(1, ramp_steps)), 0.0, 1.0).astype(np.float32)
        wheel_w_cmd = alpha * wheel_w_max

        targets = np.zeros((num_envs, robots.num_dof), dtype=np.float32)
        targets[:, left_idx] = wheel_w_cmd
        targets[:, right_idx] = wheel_w_cmd
        robots.set_joint_velocity_targets(targets)

        world.step(render=True)

        pos, rot = robots.get_world_poses()
        dx = (pos[:, 0] - start_x).astype(np.float32)

        reached = dx >= goal_dist
        if np.any(reached):
            ids = np.nonzero(reached)[0]

            # 清目标/速度（全量写更稳，避开 indices 版本坑）
            robots.set_joint_velocity_targets(np.zeros((num_envs, robots.num_dof), dtype=np.float32))
            v_zero = np.zeros((num_envs, 6), dtype=np.float32)
            robots.set_velocities(v_zero)

            # 回起点（全量写）
            pos_new = np.array(pos, copy=True)
            rot_new = np.array(rot, copy=True)
            pos_new[ids, :] = spawn_pos[ids, :]
            rot_new[ids, :] = spawn_rot[ids, :]
            robots.set_world_poses(pos_new, rot_new)
            robots.set_velocities(v_zero)
            robots.set_joint_velocities(np.zeros((num_envs, robots.num_dof), dtype=np.float32))

            # 更新这些 env 的 goal
            start_x[ids] = spawn_pos[ids, 0]
            goal_dist[ids] = 1.0 + np.random.rand(len(ids)).astype(np.float32) * 1.0
            local_step[ids] = 0
            loops[ids] += 1

            # 刷新 marker 位置（只更新触发 reset 的 env）
            for j in ids:
                goal_pos[j, 0] = env_origins[j, 0] + goal_dist[j]
                goal_pos[j, 1] = env_origins[j, 1]
                goal_pos[j, 2] = 0.05
                markers[j].set_world_pose(position=goal_pos[j].tolist(), orientation=[1, 0, 0, 0])

        now = time.time()
        if now - last_print > 0.5:
            p = pos
            print(
                f"env0 dx={dx[0]:.2f}/{goal_dist[0]:.2f} loops={loops[0]} | "
                f"env1 dx={dx[1]:.2f}/{goal_dist[1]:.2f} loops={loops[1]} | "
                f"env2 dx={dx[2]:.2f}/{goal_dist[2]:.2f} loops={loops[2]} | "
                f"env3 dx={dx[3]:.2f}/{goal_dist[3]:.2f} loops={loops[3]}"
            )
            last_print = now

        time.sleep(0.001)

    simulation_app.close()


if __name__ == "__main__":
    main()