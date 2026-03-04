# scripts/tb3_4env_com_stable_v2.py

import time
import numpy as np

from omni.isaac.kit import SimulationApp

TB3_USD = "/workspace/rl_igym_waffle/assets/turtlebot3_waffle_pi/waffle_pi.usd"


def quat_from_yaw(yaw):
    half = 0.5 * yaw
    q = np.zeros((len(yaw), 4), dtype=np.float32)  # wxyz
    q[:, 0] = np.cos(half)
    q[:, 3] = np.sin(half)
    return q


def yaw_from_quat(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(t3, t4)


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def sample_goals(n):
    ang = np.random.uniform(-np.pi, np.pi, n).astype(np.float32)
    r = np.random.uniform(1.0, 1.5, n).astype(np.float32)
    dx = r * np.cos(ang)
    dy = r * np.sin(ang)
    return np.stack([dx, dy], axis=1).astype(np.float32)


def main():
    print("[BOOT] starting SimulationApp ...")
    simulation_app = SimulationApp({"headless": False})

    try:
        # 关键：让 Kit/渲染/扩展先跑起来，不然 is_running 可能一开始就是 False
        for _ in range(30):
            simulation_app.update()

        print("[BOOT] Kit updated, building world ...")

        from omni.isaac.core import World
        from omni.isaac.core.articulations import ArticulationView
        from omni.isaac.core.objects import VisualSphere
        from omni.isaac.core.utils.stage import add_reference_to_stage
        from omni.isaac.core.prims import RigidPrimView

        np.random.seed(0)

        physics_dt = 1.0 / 240.0
        world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=1.0 / 60.0)
        world.scene.add_default_ground_plane()

        num_envs = 4
        spacing = 4.0

        env_origins = np.zeros((num_envs, 3), dtype=np.float32)
        for i in range(num_envs):
            env_origins[i, 0] = (i % 2) * spacing
            env_origins[i, 1] = (i // 2) * spacing
            env_origins[i, 2] = 0.0

        print("[BOOT] adding TB3 USD references ...")
        for i in range(num_envs):
            add_reference_to_stage(TB3_USD, f"/World/envs/env_{i}/TB3")

        # 等引用展开
        for _ in range(120):
            world.step(render=True)

        print("[BOOT] creating views ...")
        robots = ArticulationView(
            prim_paths_expr="/World/envs/env_.*/TB3/a__namespace_base_footprint",
            name="tb3_view",
            reset_xform_properties=False,
        )
        world.scene.add(robots)

        base_view = RigidPrimView(
            prim_paths_expr="/World/envs/env_.*/TB3/a__namespace_base_link",
            name="base_view",
            reset_xform_properties=False,
        )
        world.scene.add(base_view)

        world.reset()

        print("[INFO] robots.count =", robots.count)
        if robots.count != num_envs:
            print("[WARN] prim_paths_expr 没匹配到 4 台车，先检查 TB3 资产的 prim 层级/名字")

        dof_names = robots.dof_names
        print("\n========== DOF NAMES ==========")
        for i, n in enumerate(dof_names):
            print(f"{i:02d}: {n}")
        print("================================\n")

        left = None
        right = None
        for i, n in enumerate(dof_names):
            s = n.lower()
            if left is None and "wheel_left_joint" in s:
                left = i
            if right is None and "wheel_right_joint" in s:
                right = i
        if left is None or right is None:
            raise RuntimeError("没找到 wheel_left_joint / wheel_right_joint，请按 DOF NAMES 改匹配规则")

        # velocity drive tuning：kp=0，只调 kd；effort 限制能量
        wheel_kp = 0.0
        wheel_kd = 260.0
        wheel_max_effort = 8.0

        kps, kds = robots.get_gains()
        kps[:, left] = wheel_kp
        kps[:, right] = wheel_kp
        kds[:, left] = wheel_kd
        kds[:, right] = wheel_kd
        robots.set_gains(kps=kps, kds=kds)

        max_eff = robots.get_max_efforts()
        if max_eff is not None:
            max_eff[:, left] = wheel_max_effort
            max_eff[:, right] = wheel_max_effort
            robots.set_max_efforts(max_eff)

        # COM 下压（改善点头）
        base_mass = 5.0
        com_z = -0.06
        base_view.set_masses(np.ones(num_envs, dtype=np.float32) * base_mass)
        coms = np.zeros((num_envs, 3), dtype=np.float32)
        coms[:, 2] = com_z
        base_view.set_coms(coms)

        # spawn：各自 env 原点 + 随机 yaw
        spawn = env_origins.copy()
        spawn[:, 2] = 0.12
        spawn_yaw = np.random.uniform(-np.pi, np.pi, num_envs).astype(np.float32)
        robots.set_world_poses(spawn, quat_from_yaw(spawn_yaw))
        robots.set_velocities(np.zeros((num_envs, 6), dtype=np.float32))
        robots.set_joint_velocities(np.zeros((num_envs, robots.num_dof), dtype=np.float32))

        # goals
        goals_local = sample_goals(num_envs)
        goals_world = env_origins.copy()
        goals_world[:, 0] += goals_local[:, 0]
        goals_world[:, 1] += goals_local[:, 1]
        goals_world[:, 2] = 0.03

        markers = []
        for i in range(num_envs):
            m = world.scene.add(
                VisualSphere(
                    prim_path=f"/World/envs/env_{i}/goal",
                    name=f"goal_{i}",
                    position=goals_world[i].tolist(),
                    radius=0.02,
                    color=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                )
            )
            markers.append(m)

        # controller
        WHEEL_R = 0.033
        WHEEL_BASE = 0.287

        v_max = 0.08
        k_yaw = 2.0
        max_w = 1.4

        reach = 0.12

        warmup_steps = int(0.5 / physics_dt)
        ramp_steps = int(4.0 / physics_dt)
        step_counter = np.zeros(num_envs, dtype=np.int64)

        print("[RUN] enter main loop ...")
        last_log = time.time()

        while simulation_app.is_running():
            step_counter += 1
            alpha = np.clip((step_counter - warmup_steps) / float(max(1, ramp_steps)), 0.0, 1.0).astype(np.float32)
            v_cmd = v_max * alpha

            pos, rot = robots.get_world_poses()
            yaw = yaw_from_quat(rot)

            x = (pos[:, 0] - env_origins[:, 0]).astype(np.float32)
            y = (pos[:, 1] - env_origins[:, 1]).astype(np.float32)

            dx = (goals_local[:, 0] - x).astype(np.float32)
            dy = (goals_local[:, 1] - y).astype(np.float32)

            dist = np.sqrt(dx * dx + dy * dy).astype(np.float32)
            heading = np.arctan2(dy, dx)
            err = wrap(heading - yaw).astype(np.float32)

            w = np.clip(k_yaw * err, -max_w, max_w).astype(np.float32)

            slow = np.clip(dist / 1.0, 0.1, 1.0).astype(np.float32)
            v = v_cmd * slow * (1.0 - np.clip(np.abs(err) / 1.2, 0.0, 0.8).astype(np.float32))

            wl = (v - w * (WHEEL_BASE / 2.0)) / WHEEL_R
            wr = (v + w * (WHEEL_BASE / 2.0)) / WHEEL_R

            targets = np.zeros((num_envs, robots.num_dof), dtype=np.float32)
            targets[:, left] = wl
            targets[:, right] = wr
            robots.set_joint_velocity_targets(targets)

            world.step(render=True)

            reached = dist < reach
            if np.any(reached):
                ids = np.where(reached)[0]

                robots.set_joint_velocity_targets(np.zeros_like(targets))
                robots.set_velocities(np.zeros((num_envs, 6), dtype=np.float32))
                robots.set_joint_velocities(np.zeros((num_envs, robots.num_dof), dtype=np.float32))

                # reset those envs: back to origin + new yaw
                pos2, rot2 = robots.get_world_poses()
                pos_new = np.array(pos2, copy=True)
                rot_new = np.array(rot2, copy=True)

                new_yaw = np.random.uniform(-np.pi, np.pi, len(ids)).astype(np.float32)
                pos_new[ids, :] = spawn[ids, :]
                rot_new[ids, :] = quat_from_yaw(new_yaw)

                robots.set_world_poses(pos_new, rot_new)
                robots.set_velocities(np.zeros((num_envs, 6), dtype=np.float32))
                robots.set_joint_velocities(np.zeros((num_envs, robots.num_dof), dtype=np.float32))

                # refresh goals for those envs
                goals_local[ids] = sample_goals(len(ids))
                for i in ids:
                    goals_world[i, 0] = env_origins[i, 0] + goals_local[i, 0]
                    goals_world[i, 1] = env_origins[i, 1] + goals_local[i, 1]
                    goals_world[i, 2] = 0.03
                    markers[i].set_world_pose(position=goals_world[i].tolist(), orientation=[1, 0, 0, 0])

                step_counter[ids] = 0

            now = time.time()
            if now - last_log > 0.8:
                print(
                    f"dist: [{dist[0]:.2f}, {dist[1]:.2f}, {dist[2]:.2f}, {dist[3]:.2f}] "
                    f"COMz={com_z:.3f} kd={wheel_kd:.1f} eff={wheel_max_effort:.1f}"
                )
                last_log = now

            time.sleep(0.001)

    except Exception as e:
        print("[FATAL] exception:", repr(e))
        import traceback
        traceback.print_exc()
    finally:
        try:
            simulation_app.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()