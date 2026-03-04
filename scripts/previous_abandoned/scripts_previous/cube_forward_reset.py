import time
import torch

from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid


def main():

    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1.0 / 60.0,
        rendering_dt=1.0 / 60.0,
    )

    world.scene.add_default_ground_plane()

    cube = world.scene.add(
        DynamicCuboid(
            prim_path="/World/TestCube",
            name="cube",
            position=[0.0, 0.0, 0.5],
            scale=[0.2, 0.2, 0.2],
        )
    )

    world.reset()

    start_pos = cube.get_world_pose()[0]

    velocity = 0.5
    target_distance = 1.0

    while simulation_app.is_running():

        cube.set_linear_velocity(torch.tensor([velocity, 0.0, 0.0]))
        cube.set_angular_velocity(torch.zeros(3))

        world.step(render=True)

        pos = cube.get_world_pose()[0]
        dx = pos[0] - start_pos[0]

        print("dx:", float(dx))

        if dx >= target_distance:
            cube.set_linear_velocity(torch.zeros(3))
            cube.set_angular_velocity(torch.zeros(3))
            cube.set_world_pose(start_pos, [1, 0, 0, 0])
            print("RESET DONE")

        time.sleep(0.01)

    simulation_app.close()


if __name__ == "__main__":
    main()