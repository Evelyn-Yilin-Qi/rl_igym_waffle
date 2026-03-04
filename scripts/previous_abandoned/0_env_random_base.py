import time
import numpy as np

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.physics_context import PhysicsContext

from pxr import Gf, UsdGeom, UsdPhysics


SCENE_EMPTY = "empty"
SCENE_BOX = "box"
SCENE_CYLINDER = "cylinder"
SCENE_DOOR = "door"


def _create_cube(stage, prim_path, size_xyz, center_xyz, visible=True, color=(0.6, 0.6, 0.6), collision=True):
    cube = UsdGeom.Cube.Define(stage, prim_path)
    cube.CreateSizeAttr(1.0)

    prim = cube.GetPrim()
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    t_op = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    s_op = xformable.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)

    t_op.Set(Gf.Vec3d(float(center_xyz[0]), float(center_xyz[1]), float(center_xyz[2])))
    s_op.Set(Gf.Vec3d(float(size_xyz[0]), float(size_xyz[1]), float(size_xyz[2])))

    imageable = UsdGeom.Imageable(prim)
    if visible:
        imageable.MakeVisible()
    else:
        imageable.MakeInvisible()

    gprim = UsdGeom.Gprim(prim)
    gprim.CreateDisplayColorAttr([Gf.Vec3f(*color)])

    if collision:
        UsdPhysics.CollisionAPI.Apply(prim)

    return prim, t_op, s_op


def _create_cylinder(stage, prim_path, radius, height, center_xyz, visible=True, color=(0.5, 0.7, 0.9), collision=True):
    cyl = UsdGeom.Cylinder.Define(stage, prim_path)
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))

    prim = cyl.GetPrim()
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    t_op = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)

    t_op.Set(Gf.Vec3d(float(center_xyz[0]), float(center_xyz[1]), float(center_xyz[2])))

    imageable = UsdGeom.Imageable(prim)
    if visible:
        imageable.MakeVisible()
    else:
        imageable.MakeInvisible()

    gprim = UsdGeom.Gprim(prim)
    gprim.CreateDisplayColorAttr([Gf.Vec3f(*color)])

    if collision:
        UsdPhysics.CollisionAPI.Apply(prim)

    return prim, t_op, cyl


def _set_visibility(prim, visible):
    imageable = UsdGeom.Imageable(prim)
    if visible:
        imageable.MakeVisible()
    else:
        imageable.MakeInvisible()


def _hide_prim_below_ground(t_op, z=-10.0):
    p = t_op.Get()
    t_op.Set(Gf.Vec3d(float(p[0]), float(p[1]), float(z)))


def reset_env(env_id, scene_type, env_origin, handles, rng):
    # Hide all scenario-specific objects first.
    _hide_prim_below_ground(handles["box"]["t_op"])
    for c in handles["cyls"]:
        _hide_prim_below_ground(c["t_op"])
    _hide_prim_below_ground(handles["door_left"]["t_op"])
    _hide_prim_below_ground(handles["door_right"]["t_op"])

    if scene_type == SCENE_EMPTY:
        return

    if scene_type == SCENE_BOX:
        # Random box dimensions (both length and width between 1~3m)
        l = rng.uniform(1.0, 3.0)
        b = rng.uniform(1.0, 3.0)
        h = 0.6
        
        # Box center fixed at env local origin (0, 0)
        handles["box"]["s_op"].Set(Gf.Vec3d(float(l), float(b), float(h)))
        handles["box"]["t_op"].Set(Gf.Vec3d(
            float(env_origin[0]),  # local x = 0
            float(env_origin[1]),  # local y = 0
            h * 0.5
        ))
        return

    if scene_type == SCENE_CYLINDER:
        # Four cylinders fixed at local coordinates (±1, ±1)
        h = 0.7
        fixed_positions = [
            (1.0, 1.0),   # (+1, +1)
            (-1.0, 1.0),  # (-1, +1)
            (1.0, -1.0),  # (+1, -1)
            (-1.0, -1.0), # (-1, -1)
        ]
        for i, c in enumerate(handles["cyls"]):
            # Random radius between 0.1 and 0.3 for each reset
            rad = rng.uniform(0.1, 0.3)
            ox, oy = fixed_positions[i]
            c["geom"].CreateRadiusAttr(float(rad))
            c["geom"].CreateHeightAttr(float(h))
            c["t_op"].Set(Gf.Vec3d(float(env_origin[0] + ox), float(env_origin[1] + oy), h * 0.5))
        return

    if scene_type == SCENE_DOOR:
        # Door with random width and position offset (can be shifted up or down)
        door_w = rng.uniform(0.9, 1.75)
        wall_t = 0.12  # Slightly thicker wall for door scene
        wall_h = 1.0
        half = 3.0  # env size / 2
        
        # Random door offset in y direction (up/down): ensure door stays within bounds
        # Door center can be between -2.0 and +2.0 (leaving margin for door width)
        max_offset = 2.0 - door_w * 0.5  # Ensure door doesn't exceed boundaries
        door_offset_y = rng.uniform(-max_offset, max_offset)
        
        # Vertical partition at x=0 in local frame, door gap is at y = door_offset_y
        # Calculate two wall segments: upper and lower
        
        # Door edges in local y coordinates
        door_top = door_offset_y + door_w * 0.5
        door_bottom = door_offset_y - door_w * 0.5
        
        # Upper segment: from top (y=3) down to door top
        upper_seg_len = 3.0 - door_top
        upper_center_local_y = (3.0 + door_top) * 0.5
        
        # Lower segment: from door bottom down to bottom (y=-3)
        lower_seg_len = door_bottom - (-3.0)
        lower_center_local_y = (door_bottom + (-3.0)) * 0.5
        
        # Ensure segments have minimum length
        min_seg_len = 0.1
        if upper_seg_len < min_seg_len:
            upper_seg_len = min_seg_len
            upper_center_local_y = 2.5  # Adjust center if needed
        if lower_seg_len < min_seg_len:
            lower_seg_len = min_seg_len
            lower_center_local_y = -2.5  # Adjust center if needed

        handles["door_left"]["s_op"].Set(Gf.Vec3d(float(wall_t), float(upper_seg_len), float(wall_h)))
        handles["door_right"]["s_op"].Set(Gf.Vec3d(float(wall_t), float(lower_seg_len), float(wall_h)))

        handles["door_left"]["t_op"].Set(Gf.Vec3d(float(env_origin[0]), float(env_origin[1] + upper_center_local_y), wall_h * 0.5))
        handles["door_right"]["t_op"].Set(Gf.Vec3d(float(env_origin[0]), float(env_origin[1] + lower_center_local_y), wall_h * 0.5))


def main():
    rng = np.random.default_rng(42)

    physics_dt = 1.0 / 240.0
    render_dt = 1.0 / 60.0

    world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=render_dt)
    PhysicsContext().substeps = 8
    world.scene.add_default_ground_plane()

    stage = world.scene.stage

    num_envs = 4
    env_size = 6.0
    env_gap = 2.0  # Increased from 1.0m to 2.0m for better isolation
    env_spacing = env_size + env_gap  # 8m
    wall_t = 0.08
    wall_h = 1.2

    scene_types = [SCENE_EMPTY, SCENE_BOX, SCENE_CYLINDER, SCENE_DOOR]
    show_visual_walls = [False, True, True, True]  # Box scene now has visible walls

    env_origins = np.zeros((num_envs, 3), dtype=np.float32)
    for i in range(num_envs):
        ix = i % 2
        iy = i // 2
        env_origins[i, 0] = ix * env_spacing
        env_origins[i, 1] = iy * env_spacing

    handles = []

    for i in range(num_envs):
        root = f"/World/envs/env_{i}"
        origin = env_origins[i]
        half = env_size * 0.5

        walls = []
        wall_defs = [
            (f"{root}/BoundLeft", (wall_t, env_size + 2 * wall_t, wall_h), (origin[0] - half - wall_t * 0.5, origin[1], wall_h * 0.5)),
            (f"{root}/BoundRight", (wall_t, env_size + 2 * wall_t, wall_h), (origin[0] + half + wall_t * 0.5, origin[1], wall_h * 0.5)),
            (f"{root}/BoundFront", (env_size + 2 * wall_t, wall_t, wall_h), (origin[0], origin[1] - half - wall_t * 0.5, wall_h * 0.5)),
            (f"{root}/BoundBack", (env_size + 2 * wall_t, wall_t, wall_h), (origin[0], origin[1] + half + wall_t * 0.5, wall_h * 0.5)),
        ]

        for p, size, center in wall_defs:
            prim, t_op, s_op = _create_cube(
                stage,
                p,
                size_xyz=size,
                center_xyz=center,
                visible=show_visual_walls[i],
                color=(0.6, 0.6, 0.6),
                collision=True,
            )
            walls.append({"prim": prim, "t_op": t_op, "s_op": s_op})

        # Scenario objects (all pre-created; reset updates pose/size only)
        box_prim, box_t, box_s = _create_cube(
            stage,
            f"{root}/ScenarioBox",
            size_xyz=(1.0, 1.0, 0.6),
            center_xyz=(origin[0], origin[1], -10.0),
            visible=True,
            color=(0.85, 0.85, 0.85),
            collision=True,
        )

        cyls = []
        for k in range(4):
            c_prim, c_t, c_geom = _create_cylinder(
                stage,
                f"{root}/ScenarioCyl_{k}",
                radius=0.18,
                height=0.7,
                center_xyz=(origin[0], origin[1], -10.0),
                visible=True,
                color=(0.7, 0.7, 0.7),
                collision=True,
            )
            cyls.append({"prim": c_prim, "t_op": c_t, "geom": c_geom})

        door_l_prim, door_l_t, door_l_s = _create_cube(
            stage,
            f"{root}/DoorSegLeft",
            size_xyz=(0.12, 2.0, 1.0),  # Thicker wall for door scene
            center_xyz=(origin[0], origin[1], -10.0),
            visible=True,
            color=(0.7, 0.7, 0.7),
            collision=True,
        )

        door_r_prim, door_r_t, door_r_s = _create_cube(
            stage,
            f"{root}/DoorSegRight",
            size_xyz=(0.12, 2.0, 1.0),  # Thicker wall for door scene
            center_xyz=(origin[0], origin[1], -10.0),
            visible=True,
            color=(0.7, 0.7, 0.7),
            collision=True,
        )

        # Respect scene-level visibility for scenario objects too.
        # Note: Box scene's box should be visible even if walls are not
        scene_visible = show_visual_walls[i]
        box_visible = True if scene_types[i] == SCENE_BOX else scene_visible
        _set_visibility(box_prim, box_visible)
        for c in cyls:
            _set_visibility(c["prim"], scene_visible)
        _set_visibility(door_l_prim, scene_visible)
        _set_visibility(door_r_prim, scene_visible)

        handles.append(
            {
                "walls": walls,
                "box": {"prim": box_prim, "t_op": box_t, "s_op": box_s},
                "cyls": cyls,
                "door_left": {"prim": door_l_prim, "t_op": door_l_t, "s_op": door_l_s},
                "door_right": {"prim": door_r_prim, "t_op": door_r_t, "s_op": door_r_s},
            }
        )

    world.reset()

    elapsed = np.zeros((num_envs,), dtype=np.float32)
    # All scenes reset randomly between 2-5 seconds
    next_reset = rng.uniform(2.0, 5.0, size=(num_envs,)).astype(np.float32)

    for i in range(num_envs):
        reset_env(i, scene_types[i], env_origins[i], handles[i], rng)

    print("[INFO] 4 scenarios initialized")
    print("[INFO] env size=6x6m, env gap=2m, env spacing=8m")
    print("[INFO] scenes: env0=empty, env1=box, env2=cylinder, env3=door")

    last_print = time.time()

    while simulation_app.is_running():
        elapsed += physics_dt
        to_reset = np.nonzero(elapsed >= next_reset)[0]

        if len(to_reset) > 0:
            for i in to_reset:
                reset_env(i, scene_types[i], env_origins[i], handles[i], rng)
                elapsed[i] = 0.0
                # All scenes reset randomly between 2-5 seconds
                next_reset[i] = np.float32(rng.uniform(2.0, 5.0))

        world.step(render=True)

        now = time.time()
        if now - last_print > 1.0:
            print(
                f"[STATUS] E0:{next_reset[0]-elapsed[0]:.1f}s | "
                f"E1:{next_reset[1]-elapsed[1]:.1f}s | "
                f"E2:{next_reset[2]-elapsed[2]:.1f}s | "
                f"E3:{next_reset[3]-elapsed[3]:.1f}s"
            )
            last_print = now

        time.sleep(0.001)

    simulation_app.close()


if __name__ == "__main__":
    main()