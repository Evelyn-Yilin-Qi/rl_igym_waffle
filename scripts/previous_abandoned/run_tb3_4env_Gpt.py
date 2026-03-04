"""
Main entry point for TB3 multi-environment simulation.

Only change vs your current version:
- After TB3Simulator finishes spawning all TB3s, apply USD MassAPI in one shot to every TB3 base_link + imu_link in the whole stage.
- Step a few frames so MassAPI values propagate into PhysX.

Everything else stays the same.
"""

import time
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.physics_context import PhysicsContext
from core.simulator import TB3Simulator

import omni.usd
from pxr import UsdPhysics, Gf


def apply_tb3_massapi_for_all_tb3_links(base_mass: float, com_x: float, com_z: float):
    """
    Traverse the whole stage and apply sane mass/COM/inertia to:
      - a__namespace_base_link
      - a__namespace_imu_link

    This avoids:
      - negative mass / invalid inertia warnings
      - unreliable view buffer writes (set_masses/set_coms) in multi-env setups
    """
    stage = omni.usd.get_context().get_stage()
    base_count = 0
    imu_count = 0

    for prim in stage.Traverse():
        if not prim.IsValid():
            continue

        name = prim.GetName()

        if name == "a__namespace_base_link":
            api = UsdPhysics.MassAPI.Apply(prim)
            api.CreateMassAttr(float(base_mass))
            api.CreateCenterOfMassAttr(Gf.Vec3f(float(com_x), 0.0, float(com_z)))
            # Stability-first inertia (diagonal). Adjust later if you want realism.
            api.CreateDiagonalInertiaAttr(Gf.Vec3f(0.02, 0.02, 0.02))
            base_count += 1

        elif name == "a__namespace_imu_link":
            # Critical: ensure strictly positive mass + sane inertia
            api = UsdPhysics.MassAPI.Apply(prim)
            api.CreateMassAttr(0.01)
            api.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
            api.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-5, 1e-5, 1e-5))
            imu_count += 1

    print(f"[INFO] MassAPI applied: base_link={base_count}, imu_link={imu_count}")


def main():
    physics_dt = 1.0 / 240.0
    render_dt = 1.0 / 60.0

    world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=render_dt)
    PhysicsContext().substeps = 8
    world.scene.add_default_ground_plane()

    from core.config import SCENE_EMPTY, SCENE_BOX, SCENE_CYLINDER, SCENE_DOOR
    scene_config = {
        SCENE_EMPTY: 4,
        SCENE_BOX: 4,
        SCENE_CYLINDER: 4,
        SCENE_DOOR: 4,
    }

    simulator = TB3Simulator(world, scene_config=scene_config, physics_dt=physics_dt, render_dt=render_dt)

    # ---- Only new block starts here ----
    # Apply mass/COM/inertia after TB3Simulator has spawned robots into the stage.
    from core.config import BASE_MASS, COM_X, COM_Z

    # Let the stage settle a bit (ensures prims exist and PhysX buffers are initialized)
    for _ in range(10):
        simulator.step()

    apply_tb3_massapi_for_all_tb3_links(BASE_MASS, COM_X, COM_Z)

    # Step a few frames so MassAPI propagates into physics
    for _ in range(20):
        simulator.step()
    # ---- Only new block ends here ----

    last_print = time.time()

    while simulation_app.is_running():
        simulator.step()

        now = time.time()
        if now - last_print > 0.6:
            print(simulator.get_status_string())
            last_print = now

        time.sleep(0.001)

    simulation_app.close()


if __name__ == "__main__":
    main()