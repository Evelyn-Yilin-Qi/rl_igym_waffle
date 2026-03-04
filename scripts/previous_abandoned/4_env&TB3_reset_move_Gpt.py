import time
import numpy as np

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.physics_context import PhysicsContext
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.objects import VisualSphere
from omni.isaac.core.utils.stage import add_reference_to_stage

import omni.usd
from pxr import Gf, UsdGeom, UsdPhysics


SCENE_EMPTY = "empty"
SCENE_BOX = "box"
SCENE_CYLINDER = "cylinder"
SCENE_DOOR = "door"

TB3_USD = "/workspace/rl_igym_waffle/assets/turtlebot3_waffle_pi/waffle_pi.usd"

WHEEL_RADIUS = 0.033
WHEEL_BASE = 0.287

ENV_SIZE = 6.0
ENV_HALF = 3.0

ROBOT_HALF_LENGTH_X = 0.145
ROBOT_HALF_WIDTH_Y = 0.155
ROBOT_COLLISION_RADIUS = max(ROBOT_HALF_LENGTH_X, ROBOT_HALF_WIDTH_Y)
TIMEOUT_SECONDS = 60.0


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


def apply_massapi_all_tb3(base_mass=2.2, com_x=-0.08, com_z=-0.10):
    stage = omni.usd.get_context().get_stage()
    base_cnt = 0
    imu_cnt = 0

    for prim in stage.Traverse():
        if not prim.IsValid():
            continue

        name = prim.GetName()

        if name == "a__namespace_base_link":
            api = UsdPhysics.MassAPI.Apply(prim)
            api.CreateMassAttr(float(base_mass))
            api.CreateCenterOfMassAttr(Gf.Vec3f(float(com_x), 0.0, float(com_z)))
            api.CreateDiagonalInertiaAttr(Gf.Vec3f(0.02, 0.02, 0.02))
            base_cnt += 1

        elif name == "a__namespace_imu_link":
            api = UsdPhysics.MassAPI.Apply(prim)
            api.CreateMassAttr(0.01)
            api.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
            api.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-5, 1e-5, 1e-5))
            imu_cnt += 1

    print(f"[INFO] MassAPI applied: base_link={base_cnt}, imu_link={imu_cnt}")


def check_collision_with_obstacles(robot_pos_local, scene_type, handles, env_origin):
    x, y = robot_pos_local[0], robot_pos_local[1]

    if scene_type == SCENE_BOX:
        box_t = handles["box"]["t_op"]
        box_s = handles["box"]["s_op"]
        box_pos = box_t.Get()
        box_size = box_s.Get()

        box_center_world = np.array([box_pos[0], box_pos[1]])
        robot_pos_world = np.array([env_origin[0] + x, env_origin[1] + y])

        box_half_x = box_size[0] * 0.5
        box_half_y = box_size[1] * 0.5

        box_min_x = box_center_world[0] - box_half_x
        box_max_x = box_center_world[0] + box_half_x
        box_min_y = box_center_world[1] - box_half_y
        box_max_y = box_center_world[1] + box_half_y

        robot_min_x = robot_pos_world[0] - ROBOT_HALF_LENGTH_X
        robot_max_x = robot_pos_world[0] + ROBOT_HALF_LENGTH_X
        robot_min_y = robot_pos_world[1] - ROBOT_HALF_WIDTH_Y
        robot_max_y = robot_pos_world[1] + ROBOT_HALF_WIDTH_Y

        if (
            robot_min_x < box_max_x
            and robot_max_x > box_min_x
            and robot_min_y < box_max_y
            and robot_max_y > box_min_y
        ):
            return True

    elif scene_type == SCENE_CYLINDER:
        for cyl in handles["cyls"]:
            cyl_t = cyl["t_op"]
            cyl_geom = cyl["geom"]
            cyl_pos = cyl_t.Get()
            cyl_radius = cyl_geom.GetRadiusAttr().Get()

            cyl_center_world = np.array([cyl_pos[0], cyl_pos[1]])
            robot_pos_world = np.array([env_origin[0] + x, env_origin[1] + y])

            dist_to_cyl = np.linalg.norm(robot_pos_world - cyl_center_world)
            if dist_to_cyl < (cyl_radius + ROBOT_COLLISION_RADIUS):
                return True

    elif scene_type == SCENE_DOOR:
        door_l_t = handles["door_left"]["t_op"]
        door_r_t = handles["door_right"]["t_op"]
        door_l_s = handles["door_left"]["s_op"]
        door_r_s = handles["door_right"]["s_op"]

        door_l_pos = door_l_t.Get()
        door_r_pos = door_r_t.Get()
        door_l_size = door_l_s.Get()
        door_r_size = door_r_s.Get()

        robot_pos_world = np.array([env_origin[0] + x, env_origin[1] + y])

        wall_l_center = np.array([door_l_pos[0], door_l_pos[1]])
        wall_l_half_len = door_l_size[1] * 0.5
        wall_l_thickness = door_l_size[0] * 0.5

        dist_to_wall_l_x = abs(robot_pos_world[0] - wall_l_center[0])
        dist_to_wall_l_y = abs(robot_pos_world[1] - wall_l_center[1])
        if dist_to_wall_l_x < (wall_l_thickness + ROBOT_HALF_LENGTH_X):
            if dist_to_wall_l_y < (wall_l_half_len + ROBOT_HALF_WIDTH_Y):
                return True

        wall_r_center = np.array([door_r_pos[0], door_r_pos[1]])
        wall_r_half_len = door_r_size[1] * 0.5
        wall_r_thickness = door_r_size[0] * 0.5

        dist_to_wall_r_x = abs(robot_pos_world[0] - wall_r_center[0])
        dist_to_wall_r_y = abs(robot_pos_world[1] - wall_r_center[1])
        if dist_to_wall_r_x < (wall_r_thickness + ROBOT_HALF_LENGTH_X):
            if dist_to_wall_r_y < (wall_r_half_len + ROBOT_HALF_WIDTH_Y):
                return True

    return False


def check_boundary_collision(robot_pos_local):
    x, y = robot_pos_local[0], robot_pos_local[1]
    if abs(x) > (ENV_HALF - ROBOT_HALF_LENGTH_X) or abs(y) > (ENV_HALF - ROBOT_HALF_WIDTH_Y):
        return True
    return False


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


def reset_env(env_id, scene_type, env_origin, handles, rng, robots=None, goal_offsets=None, markers=None, goal_pos=None):
    _hide_prim_below_ground(handles["box"]["t_op"])
    for c in handles["cyls"]:
        _hide_prim_below_ground(c["t_op"])
    _hide_prim_below_ground(handles["door_left"]["t_op"])
    _hide_prim_below_ground(handles["door_right"]["t_op"])

    if scene_type == SCENE_BOX:
        l = rng.uniform(1.0, 3.0)
        b = rng.uniform(1.0, 3.0)
        h = 0.6
        handles["box"]["s_op"].Set(Gf.Vec3d(float(l), float(b), float(h)))
        handles["box"]["t_op"].Set(Gf.Vec3d(float(env_origin[0]), float(env_origin[1]), h * 0.5))

    elif scene_type == SCENE_CYLINDER:
        h = 0.7
        fixed_positions = [(1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)]
        for i, c in enumerate(handles["cyls"]):
            rad = rng.uniform(0.1, 0.3)
            ox, oy = fixed_positions[i]
            c["geom"].CreateRadiusAttr(float(rad))
            c["geom"].CreateHeightAttr(float(h))
            c["t_op"].Set(Gf.Vec3d(float(env_origin[0] + ox), float(env_origin[1] + oy), h * 0.5))

    elif scene_type == SCENE_DOOR:
        door_w = rng.uniform(0.9, 1.75)
        wall_t = 0.12
        wall_h = 1.0
        max_offset = 2.0 - door_w * 0.5
        door_offset_y = rng.uniform(-max_offset, max_offset)
        door_top = door_offset_y + door_w * 0.5
        door_bottom = door_offset_y - door_w * 0.5
        upper_seg_len = 3.0 - door_top
        upper_center_local_y = (3.0 + door_top) * 0.5
        lower_seg_len = door_bottom - (-3.0)
        lower_center_local_y = (door_bottom + (-3.0)) * 0.5

        min_seg_len = 0.1
        if upper_seg_len < min_seg_len:
            upper_seg_len = min_seg_len
            upper_center_local_y = 2.5
        if lower_seg_len < min_seg_len:
            lower_seg_len = min_seg_len
            lower_center_local_y = -2.5

        handles["door_left"]["s_op"].Set(Gf.Vec3d(float(wall_t), float(upper_seg_len), float(wall_h)))
        handles["door_right"]["s_op"].Set(Gf.Vec3d(float(wall_t), float(lower_seg_len), float(wall_h)))
        handles["door_left"]["t_op"].Set(
            Gf.Vec3d(float(env_origin[0]), float(env_origin[1] + upper_center_local_y), wall_h * 0.5)
        )
        handles["door_right"]["t_op"].Set(
            Gf.Vec3d(float(env_origin[0]), float(env_origin[1] + lower_center_local_y), wall_h * 0.5)
        )

    if robots is not None:
        if scene_type == SCENE_EMPTY or scene_type == SCENE_CYLINDER:
            local_x = 0.0
            local_y = 0.0
            spawn_yaw = rng.uniform(-np.pi, np.pi)
        else:
            local_x = 2.5
            local_y = 2.5
            dx = -2.5 - local_x
            dy = -2.5 - local_y
            spawn_yaw = np.arctan2(dy, dx)

        spawn_pos = np.array([env_origin[0] + local_x, env_origin[1] + local_y, 0.0350], dtype=np.float32)
        spawn_rot = quat_wxyz_from_yaw(np.array([spawn_yaw], dtype=np.float32))

        idx = np.array([env_id], dtype=np.int32)

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

    if goal_offsets is not None and goal_pos is not None:
        if scene_type == SCENE_EMPTY or scene_type == SCENE_CYLINDER:
            goal_offsets[env_id, :] = sample_goal_offsets(1, 1.0, 1.5)[0]
            goal_pos[env_id, 0] = env_origin[0] + goal_offsets[env_id, 0]
            goal_pos[env_id, 1] = env_origin[1] + goal_offsets[env_id, 1]
            goal_pos[env_id, 2] = 0.03
        else:
            goal_offsets[env_id, 0] = -2.5
            goal_offsets[env_id, 1] = -2.5
            goal_pos[env_id, 0] = env_origin[0] - 2.5
            goal_pos[env_id, 1] = env_origin[1] - 2.5
            goal_pos[env_id, 2] = 0.03

        if markers is not None:
            markers[env_id].set_world_pose(position=goal_pos[env_id].tolist(), orientation=[1, 0, 0, 0])


def main():
    rng = np.random.default_rng(42)
    np.random.seed(42)

    physics_dt = 1.0 / 240.0
    render_dt = 1.0 / 60.0

    world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=render_dt)
    PhysicsContext().substeps = 8
    world.scene.add_default_ground_plane()

    stage = world.scene.stage

    num_envs = 4
    env_size = 6.0
    env_gap = 2.0
    env_spacing = env_size + env_gap
    wall_t = 0.08
    wall_h = 1.2

    scene_types = [SCENE_EMPTY, SCENE_BOX, SCENE_CYLINDER, SCENE_DOOR]
    show_visual_walls = [False, True, True, True]

    env_origins = np.zeros((num_envs, 3), dtype=np.float32)
    for i in range(num_envs):
        ix = i % 2
        iy = i // 2
        env_origins[i, 0] = ix * env_spacing
        env_origins[i, 1] = iy * env_spacing

    for i in range(num_envs):
        tb3_root = f"/World/envs/env_{i}/TB3"
        add_reference_to_stage(usd_path=TB3_USD, prim_path=tb3_root)

    for _ in range(180):
        world.step(render=True)

    robots = ArticulationView(
        prim_paths_expr="/World/envs/env_.*/TB3/a__namespace_base_footprint",
        name="tb3_view",
        reset_xform_properties=False,
    )
    world.scene.add(robots)

    world.reset()
    robots.initialize()

    apply_massapi_all_tb3(base_mass=2.2, com_x=-0.08, com_z=-0.10)
    for _ in range(10):
        world.step(render=True)

    dof_names = robots.dof_names
    left_idx = None
    right_idx = None
    for i, n in enumerate(dof_names):
        s = n.lower()
        if left_idx is None and "wheel_left_joint" in s:
            left_idx = i
        if right_idx is None and "wheel_right_joint" in s:
            right_idx = i

    if left_idx is None or right_idx is None:
        print("[WARN] Could not find wheel joints.")
        simulation_app.close()
        return

    wheel_kp = 0.0
    wheel_kd = 260.0
    wheel_max_effort = 30.0

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
        print(f"[WARN] set_max_efforts failed: {e}")

    handles = []

    for i in range(num_envs):
        root = f"/World/envs/env_{i}"
        origin = env_origins[i]
        half = env_size * 0.5

        walls = []
        wall_defs = [
            (f"{root}/BoundLeft", (wall_t, env_size + 2 * wall_t, wall_h),
             (origin[0] - half - wall_t * 0.5, origin[1], wall_h * 0.5)),
            (f"{root}/BoundRight", (wall_t, env_size + 2 * wall_t, wall_h),
             (origin[0] + half + wall_t * 0.5, origin[1], wall_h * 0.5)),
            (f"{root}/BoundFront", (env_size + 2 * wall_t, wall_t, wall_h),
             (origin[0], origin[1] - half - wall_t * 0.5, wall_h * 0.5)),
            (f"{root}/BoundBack", (env_size + 2 * wall_t, wall_t, wall_h),
             (origin[0], origin[1] + half + wall_t * 0.5, wall_h * 0.5)),
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
            size_xyz=(0.12, 2.0, 1.0),
            center_xyz=(origin[0], origin[1], -10.0),
            visible=True,
            color=(0.7, 0.7, 0.7),
            collision=True,
        )

        door_r_prim, door_r_t, door_r_s = _create_cube(
            stage,
            f"{root}/DoorSegRight",
            size_xyz=(0.12, 2.0, 1.0),
            center_xyz=(origin[0], origin[1], -10.0),
            visible=True,
            color=(0.7, 0.7, 0.7),
            collision=True,
        )

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

    goal_offsets = np.zeros((num_envs, 2), dtype=np.float32)
    goal_pos = np.zeros((num_envs, 3), dtype=np.float32)
    markers = []
    for i in range(num_envs):
        if scene_types[i] == SCENE_EMPTY or scene_types[i] == SCENE_CYLINDER:
            goal_offsets[i, :] = sample_goal_offsets(1, 1.0, 1.5)[0]
            goal_pos[i, 0] = env_origins[i, 0] + goal_offsets[i, 0]
            goal_pos[i, 1] = env_origins[i, 1] + goal_offsets[i, 1]
        else:
            goal_offsets[i, 0] = -2.5
            goal_offsets[i, 1] = -2.5
            goal_pos[i, 0] = env_origins[i, 0] - 2.5
            goal_pos[i, 1] = env_origins[i, 1] - 2.5
        goal_pos[i, 2] = 0.03

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

    episode_start_time = np.zeros((num_envs,), dtype=np.float32)
    for i in range(num_envs):
        reset_env(i, scene_types[i], env_origins[i], handles[i], rng, robots, goal_offsets, markers, goal_pos)
        episode_start_time[i] = 0.0

    v_max = 0.26
    w_max = 2.2
    k_yaw = 3.5
    k_v = 0.8
    reach_thresh = 0.18

    warmup_steps = int(0.1 / physics_dt)
    ramp_steps = int(1.0 / physics_dt)
    local_step = np.zeros((num_envs,), dtype=np.int64)

    current_time = 0.0
    loops = np.zeros((num_envs,), dtype=np.int64)
    last_print = time.time()

    max_wheel_vel = 15.0

    COOLDOWN = int(0.5 / physics_dt)
    cooldown_steps = np.zeros((num_envs,), dtype=np.int64)

    # A) hysteresis thresholds for turn_in_place
    ENTER_TURN = 70.0 * np.pi / 180.0
    EXIT_TURN = 40.0 * np.pi / 180.0
    turn_state = np.zeros((num_envs,), dtype=bool)

    # C) only enforce "no reverse" when forward is stable
    V_NO_REVERSE_MIN = 0.08

    while simulation_app.is_running():
        local_step += 1
        current_time += physics_dt

        cooldown_mask = cooldown_steps > 0
        cooldown_steps[cooldown_mask] -= 1

        alpha = np.clip((local_step - warmup_steps) / float(max(1, ramp_steps)), 0.0, 1.0).astype(np.float32)

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
        yaw_err = wrap_to_pi(goal_heading - yaw).astype(np.float32)
        abs_yaw = np.abs(yaw_err)

        # A) hysteresis update
        enter_mask = (~turn_state) & (abs_yaw > ENTER_TURN)
        exit_mask = (turn_state) & (abs_yaw < EXIT_TURN)
        turn_state[enter_mask] = True
        turn_state[exit_mask] = False

        # B) smooth transition between "turn in place" and "drive"
        # blend=1 -> pure turn in place (v=0), blend=0 -> pure drive
        blend = np.clip((abs_yaw - EXIT_TURN) / (ENTER_TURN - EXIT_TURN), 0.0, 1.0).astype(np.float32)
        blend = np.where(turn_state, 1.0, blend).astype(np.float32)

        v_nom = np.clip(k_v * dist, 0.0, v_max).astype(np.float32)
        turn_slow = np.clip(1.0 - (abs_yaw / np.pi), 0.2, 1.0).astype(np.float32)

        # v ramps up with alpha, and fades out as blend->1
        v_use = (alpha * v_nom * turn_slow * (1.0 - blend)).astype(np.float32)

        # w is allowed more during turn; in drive mode we damp it to reduce abrupt wheel target jumps
        w_raw = np.clip(k_yaw * yaw_err, -w_max, w_max).astype(np.float32)
        w_drive_scale = 0.45
        w_scale = (blend * 1.0 + (1.0 - blend) * w_drive_scale).astype(np.float32)
        w_cmd = (alpha * w_raw * w_scale).astype(np.float32)

        # hard stop during cooldown
        v_use = np.where(cooldown_mask, 0.0, v_use).astype(np.float32)
        w_cmd = np.where(cooldown_mask, 0.0, w_cmd).astype(np.float32)

        # C) only enforce "no reverse" when forward is stable (avoid choking w at tiny v during transition)
        forward_stable = (v_use >= V_NO_REVERSE_MIN) & (~cooldown_mask) & (blend < 0.5)
        w_no_reverse = (2.0 * v_use / WHEEL_BASE) * 0.95
        w_cmd = np.where(forward_stable, np.clip(w_cmd, -w_no_reverse, w_no_reverse), w_cmd).astype(np.float32)

        # feasible w to avoid wheel saturation
        w_feasible = (max_wheel_vel * WHEEL_RADIUS - np.abs(v_use)) / (WHEEL_BASE * 0.5)
        w_feasible = np.clip(w_feasible, 0.0, w_max).astype(np.float32)
        w_cmd = np.clip(w_cmd, -w_feasible, w_feasible).astype(np.float32)

        left_w = (v_use - w_cmd * (WHEEL_BASE / 2.0)) / WHEEL_RADIUS
        right_w = (v_use + w_cmd * (WHEEL_BASE / 2.0)) / WHEEL_RADIUS

        wheel_abs_max = np.maximum(np.abs(left_w), np.abs(right_w))
        scale = np.where(wheel_abs_max > max_wheel_vel, max_wheel_vel / wheel_abs_max, 1.0).astype(np.float32)
        left_w = (left_w * scale).astype(np.float32)
        right_w = (right_w * scale).astype(np.float32)

        targets = np.zeros((num_envs, robots.num_dof), dtype=np.float32)
        targets[:, left_idx] = left_w
        targets[:, right_idx] = right_w
        robots.set_joint_velocity_targets(targets)

        world.step(render=True)

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
            if check_boundary_collision(np.array([x_local2[i], y_local2[i]])):
                collision[i] = True
            elif check_collision_with_obstacles(
                np.array([x_local2[i], y_local2[i]]),
                scene_types[i],
                handles[i],
                env_origins[i],
            ):
                collision[i] = True

            if current_time - episode_start_time[i] >= TIMEOUT_SECONDS:
                timeout[i] = True

        to_reset = reached | collision | timeout
        if np.any(to_reset):
            ids = np.nonzero(to_reset)[0]

            robots.set_joint_velocity_targets(np.zeros((num_envs, robots.num_dof), dtype=np.float32))
            v_zero = np.zeros((num_envs, 6), dtype=np.float32)
            robots.set_velocities(v_zero)

            for i in ids:
                reset_env(i, scene_types[i], env_origins[i], handles[i], rng, robots, goal_offsets, markers, goal_pos)
                episode_start_time[i] = current_time
                local_step[i] = 0
                loops[i] += 1
                cooldown_steps[i] = COOLDOWN
                turn_state[i] = False

            robots.set_velocities(v_zero)
            robots.set_joint_velocities(np.zeros((num_envs, robots.num_dof), dtype=np.float32))
            robots.set_joint_velocity_targets(np.zeros((num_envs, robots.num_dof), dtype=np.float32))

        # Debug output: env0 only
        now = time.time()
        if now - last_print > 0.35:
            # dof_vel = robots.get_joint_velocities()
            # l_act = float(dof_vel[0, left_idx])
            # r_act = float(dof_vel[0, right_idx])
            # print(
            #     f"[E0] scene={scene_types[0]} loop={int(loops[0])} cd={int(cooldown_steps[0])} "
            #     f"dist={float(dist2[0]):.3f} yaw_err={float(yaw_err[0]):+.3f} abs={float(abs_yaw[0]):.3f} "
            #     f"blend={float(blend[0]):.2f} turn={bool(turn_state[0])} alpha={float(alpha[0]):.2f} "
            #     f"v={float(v_use[0]):.3f} w={float(w_cmd[0]):+.3f} "
            #     f"cmd(L,R)=({float(left_w[0]):+.2f},{float(right_w[0]):+.2f}) "
            #     f"act(L,R)=({l_act:+.2f},{r_act:+.2f}) "
            #     f"reset(goal,coll,timeout)=({bool(reached[0])},{bool(collision[0])},{bool(timeout[0])})"
            # )
            # last_print = now

            dof_vel = robots.get_joint_velocities()
            l_act = float(dof_vel[0, left_idx])
            r_act = float(dof_vel[0, right_idx])

            # --- 新增 1：base 6D 速度（env0） ---
            base_v = robots.get_velocities()          # shape: (num_envs, 6)
            vx, vy, vz, wx, wy, wz = [float(x) for x in base_v[0]]

            # --- 新增 2：目标-实际误差（env0） ---
            l_cmd = float(left_w[0])
            r_cmd = float(right_w[0])
            errL = l_act - l_cmd
            errR = r_act - r_cmd

            print(
                f"[E0] scene={scene_types[0]} loop={int(loops[0])} cd={int(cooldown_steps[0])} "
                f"dist={float(dist2[0]):.3f} yaw_err={float(yaw_err[0]):+.3f} abs={float(abs_yaw[0]):.3f} "
                f"blend={float(blend[0]):.2f} turn={bool(turn_state[0])} alpha={float(alpha[0]):.2f} "
                f"v={float(v_use[0]):.3f} w={float(w_cmd[0]):+.3f} "
                f"cmd(L,R)=({l_cmd:+.2f},{r_cmd:+.2f}) act(L,R)=({l_act:+.2f},{r_act:+.2f}) "
                f"err(L,R)=({errL:+.2f},{errR:+.2f}) "
                f"base(vx,vy,wz)=({vx:+.3f},{vy:+.3f},{wz:+.3f}) "
                f"reset(goal,coll,timeout)=({bool(reached[0])},{bool(collision[0])},{bool(timeout[0])})"
            )
            last_print = now




        time.sleep(0.01)

    simulation_app.close()


if __name__ == "__main__":
    main()