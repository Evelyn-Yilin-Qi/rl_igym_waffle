import numpy as np
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.physics_context import PhysicsContext
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.stage import add_reference_to_stage

import omni.usd
from pxr import UsdPhysics, Gf

# 从 sim 模块导入配置
from sim.robot.tb3_config import TB3_USD, WHEEL_RADIUS, WHEEL_BASE, BASE_MASS, COM_X, COM_Z


TB3_ROOT = "/World/TB3"
BASE_FOOTPRINT_PRIM = "/World/TB3/a__namespace_base_footprint"
BASE_LINK_PRIM = "/World/TB3/a__namespace_base_link"
IMU_LINK_PRIM = "/World/TB3/a__namespace_imu_link"


def apply_mass_properties():
    stage = omni.usd.get_context().get_stage()

    base_prim = stage.GetPrimAtPath(BASE_LINK_PRIM)
    if base_prim.IsValid():
        base_mass_api = UsdPhysics.MassAPI.Apply(base_prim)
        base_mass_api.CreateMassAttr(float(BASE_MASS))
        base_mass_api.CreateCenterOfMassAttr(Gf.Vec3f(float(COM_X), 0.0, float(COM_Z)))
        base_mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(0.02, 0.02, 0.02))
        print(f"[INFO] MassAPI base_link: mass={BASE_MASS}, COM=({COM_X},0,{COM_Z})")
    else:
        print(f"[WARN] base_link prim not found: {BASE_LINK_PRIM}")

    imu_prim = stage.GetPrimAtPath(IMU_LINK_PRIM)
    if imu_prim.IsValid():
        imu_mass_api = UsdPhysics.MassAPI.Apply(imu_prim)
        imu_mass_api.CreateMassAttr(0.01)
        imu_mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
        imu_mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-5, 1e-5, 1e-5))
        print("[INFO] MassAPI imu_link fixed: mass=0.01, inertia=(1e-5,1e-5,1e-5)")
    else:
        print(f"[INFO] imu_link prim not found (skip): {IMU_LINK_PRIM}")


def find_wheel_dof_indices(dof_names):
    left_idx = None
    right_idx = None
    for i, n in enumerate(dof_names):
        s = n.lower()
        if left_idx is None and "wheel_left_joint" in s:
            left_idx = i
        if right_idx is None and "wheel_right_joint" in s:
            right_idx = i
    return left_idx, right_idx


def main():
    physics_dt = 1.0 / 240.0
    render_dt = 1.0 / 60.0

    world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=render_dt)

    physics_ctx = PhysicsContext()
    physics_ctx.substeps = 8

    world.scene.add_default_ground_plane()

    add_reference_to_stage(usd_path=TB3_USD, prim_path=TB3_ROOT)

    for _ in range(120):
        world.step(render=True)

    robots = ArticulationView(
        prim_paths_expr=BASE_FOOTPRINT_PRIM,
        name="tb3_view",
        reset_xform_properties=False,
    )
    world.scene.add(robots)

    world.reset()
    robots.initialize()

    for _ in range(5):
        world.step(render=True)

    apply_mass_properties()

    for _ in range(10):
        world.step(render=True)

    dof_names = robots.dof_names
    left_idx, right_idx = find_wheel_dof_indices(dof_names)

    if left_idx is None or right_idx is None:
        print("[ERROR] Could not find wheel joints!")
        print(f"[ERROR] DOF names: {dof_names}")
        simulation_app.close()
        return

    print(f"[INFO] Found wheel joints: left_idx={left_idx}, right_idx={right_idx}")

    kps, kds = robots.get_gains()
    kps[:, left_idx] = 0.0
    kps[:, right_idx] = 0.0
    kds[:, left_idx] = 260.0
    kds[:, right_idx] = 260.0
    robots.set_gains(kps=kps, kds=kds)

    robots.set_world_poses(
        positions=np.array([[0.0, 0.0, 0.035]], dtype=np.float32),
        orientations=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    )

    robots.set_velocities(velocities=np.zeros((1, 6), dtype=np.float32))
    robots.set_joint_velocities(velocities=np.zeros((1, robots.num_dof), dtype=np.float32))
    robots.set_joint_velocity_targets(np.zeros((1, robots.num_dof), dtype=np.float32))

    for _ in range(240):
        world.step(render=True)
        robots.set_joint_velocity_targets(np.zeros((1, robots.num_dof), dtype=np.float32))

    angular_velocity = 1.0
    left_wheel_vel = -angular_velocity * WHEEL_BASE / (2.0 * WHEEL_RADIUS)
    right_wheel_vel = angular_velocity * WHEEL_BASE / (2.0 * WHEEL_RADIUS)

    print(f"[INFO] Starting rotation: omega={angular_velocity:.2f} rad/s (CCW)")
    print(f"[INFO] Wheel velocities: left={left_wheel_vel:.3f} rad/s, right={right_wheel_vel:.3f} rad/s")

    step_count = 0

    try:
        while simulation_app.is_running():
            targets = np.zeros((1, robots.num_dof), dtype=np.float32)
            targets[0, left_idx] = float(left_wheel_vel)
            targets[0, right_idx] = float(right_wheel_vel)
            robots.set_joint_velocity_targets(targets)

            world.step(render=True)
            step_count += 1

            if step_count % 240 == 0:
                pos, _ = robots.get_world_poses()
                z = float(pos[0, 2])
                warn = ""
                if z < 0.030 or z > 0.040:
                    warn = f" [WARN: Z={z:.4f}]"
                print(
                    f"[Status] Step {step_count} | "
                    f"Pos=({pos[0,0]:.3f},{pos[0,1]:.3f},{z:.4f}) | "
                    f"omega={angular_velocity:.2f}{warn}"
                )

    except KeyboardInterrupt:
        pass

    finally:
        robots.set_joint_velocity_targets(np.zeros((1, robots.num_dof), dtype=np.float32))
        simulation_app.close()


if __name__ == "__main__":
    main()