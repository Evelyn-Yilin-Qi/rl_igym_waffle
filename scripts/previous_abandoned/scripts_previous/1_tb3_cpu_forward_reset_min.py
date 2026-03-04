# scripts/tb3_cpu_forward_reset_gains.py
# 运行：
#   isaac_python scripts/tb3_cpu_forward_reset_gains.py

import time
import numpy as np

from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.stage import add_reference_to_stage


TB3_USD = "/workspace/rl_igym_waffle/assets/turtlebot3_waffle_pi/waffle_pi.usd"


def main():
    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0)
    world.scene.add_default_ground_plane()

    add_reference_to_stage(usd_path=TB3_USD, prim_path="/World/TB3")

    for _ in range(30):
        world.step(render=True)

    # 如果这里 dof_names 为空：把 base_footprint 换成 base_link
    robots = ArticulationView(
        prim_paths_expr="/World/TB3/a__namespace_base_footprint",
        name="tb3_view",
        reset_xform_properties=False
    )
    world.scene.add(robots)
    world.reset()

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
        raise RuntimeError("没找到 wheel_left_joint / wheel_right_joint，请按 DOF NAMES 改匹配")

    # ===== 强制设置 wheel 的 gains + max_efforts =====
    # 2022.2.1 确认存在：get_gains/set_gains, get_max_efforts/set_max_efforts
    # 注意：get_gains 返回 (kps, kds)，形状通常是 (num_envs, num_dof)
    try:
        kps, kds = robots.get_gains()
        print("[DBG] gains shape:", kps.shape, kds.shape)

        # wheel velocity drive：kp=0，kd 给阻尼
        kps[:, left_idx] = 0.0
        kps[:, right_idx] = 0.0
        kds[:, left_idx] = 50.0
        kds[:, right_idx] = 50.0
        robots.set_gains(kps=kps, kds=kds)

        kps2, kds2 = robots.get_gains()
        print("[DBG] wheel gains after set:",
              float(kps2[0, left_idx]), float(kds2[0, left_idx]),
              float(kps2[0, right_idx]), float(kds2[0, right_idx]))
    except Exception as e:
        print("[ERR] set_gains failed:", e)
        return

    try:
        max_eff = robots.get_max_efforts()
        if max_eff is None:
            print("[WARN] get_max_efforts returned None (some assets hide this). Continue anyway.")
        else:
            print("[DBG] max_efforts shape:", max_eff.shape)
            # effort 给小一点先稳，走不动再加
            max_eff[:, left_idx] = 6.0
            max_eff[:, right_idx] = 6.0
            robots.set_max_efforts(max_eff)

            max_eff2 = robots.get_max_efforts()
            print("[DBG] wheel max_efforts after set:",
                  float(max_eff2[0, left_idx]), float(max_eff2[0, right_idx]))
    except Exception as e:
        print("[WARN] set_max_efforts failed:", e)

    # 起点
    start_pos, start_rot = robots.get_world_poses()
    start_pos = np.array(start_pos, copy=True)
    start_rot = np.array(start_rot, copy=True)

    # 控制参数（先小）
    WHEEL_RADIUS = 0.033
    forward_v = 0.03
    wheel_w = forward_v / WHEEL_RADIUS
    target_dist = 1.0

    warmup_steps = 60
    ramp_steps = 120
    step = 0

    last_print = time.time()

    while simulation_app.is_running():
        step += 1

        if step < warmup_steps:
            alpha = 0.0
        else:
            alpha = min(1.0, (step - warmup_steps) / float(max(1, ramp_steps)))
        w_cmd = alpha * wheel_w

        targets = np.zeros((1, robots.num_dof), dtype=np.float32)
        targets[0, left_idx] = w_cmd
        targets[0, right_idx] = w_cmd
        robots.set_joint_velocity_targets(targets)

        world.step(render=True)

        pos, rot = robots.get_world_poses()
        dx = float(pos[0, 0] - start_pos[0, 0])

        now = time.time()
        if now - last_print > 0.5:
            print(f"dx={dx:.3f}  pos=({pos[0,0]:+.3f},{pos[0,1]:+.3f},{pos[0,2]:+.3f})  w={w_cmd:.2f}")
            last_print = now

        if dx >= target_dist:
            robots.set_joint_velocity_targets(np.zeros((1, robots.num_dof), dtype=np.float32))
            robots.set_velocities(np.zeros((1, 6), dtype=np.float32))
            robots.set_world_poses(start_pos, start_rot)
            robots.set_velocities(np.zeros((1, 6), dtype=np.float32))
            step = 0
            print("[RESET] back to start")

        time.sleep(0.001)

    simulation_app.close()


if __name__ == "__main__":
    main()