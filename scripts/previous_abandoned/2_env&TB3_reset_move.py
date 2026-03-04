import time
import numpy as np

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.physics_context import PhysicsContext
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.objects import VisualSphere
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.prims import RigidPrimView

from pxr import Gf, UsdGeom, UsdPhysics


SCENE_EMPTY = "empty"
SCENE_BOX = "box"
SCENE_CYLINDER = "cylinder"
SCENE_DOOR = "door"

TB3_USD = "/workspace/rl_igym_waffle/assets/turtlebot3_waffle_pi/waffle_pi.usd"

# Physical parameters
WHEEL_RADIUS = 0.033
WHEEL_BASE = 0.287
ENV_SIZE = 6.0
ENV_HALF = 3.0
# Robot dimensions: 281 x 306 x 141mm (length x width x height)
# Use rectangular AABB collision detection based on actual dimensions
# Tightened values with small safety margin (actual: 0.1405m x 0.153m)
ROBOT_HALF_LENGTH_X = 0.145  # 281mm / 2 = 0.1405m + 0.0045m safety margin
ROBOT_HALF_WIDTH_Y = 0.155   # 306mm / 2 = 0.153m + 0.002m safety margin
# Keep backward compatibility for circular obstacles
ROBOT_COLLISION_RADIUS = max(ROBOT_HALF_LENGTH_X, ROBOT_HALF_WIDTH_Y)  # 0.155m for cylinders
COLLISION_THRESHOLD = ROBOT_COLLISION_RADIUS
TIMEOUT_SECONDS = 60.0


def yaw_from_quat_wxyz(q):
    """Convert quaternion (w, x, y, z) to yaw angle"""
    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(t3, t4)


def quat_wxyz_from_yaw(yaw):
    """Convert yaw angle to quaternion (w, x, y, z)"""
    half = 0.5 * yaw
    q = np.zeros((yaw.shape[0], 4), dtype=np.float32)
    q[:, 0] = np.cos(half)  # w
    q[:, 3] = np.sin(half)  # z
    return q


def wrap_to_pi(a):
    """Wrap angle to [-pi, pi]"""
    return np.arctan2(np.sin(a), np.cos(a))


def sample_goal_offsets(n, r_min=1.0, r_max=1.5):
    """Sample random goal offsets: (n,2) array of (dx, dy)"""
    ang = np.random.uniform(-np.pi, np.pi, size=(n,)).astype(np.float32)
    r = (r_min + (r_max - r_min) * np.random.rand(n)).astype(np.float32)
    dx = r * np.cos(ang)
    dy = r * np.sin(ang)
    return np.stack([dx, dy], axis=1)


def check_collision_with_obstacles(robot_pos_local, scene_type, handles, env_origin, rng):
    """Check if robot collides with obstacles (box, cylinders, door walls)"""
    x, y = robot_pos_local[0], robot_pos_local[1]
    
    if scene_type == SCENE_BOX:
        # Precise AABB collision detection for box
        box_t = handles["box"]["t_op"]
        box_s = handles["box"]["s_op"]
        box_pos = box_t.Get()
        box_size = box_s.Get()
        
        # Box center in world coordinates
        box_center_world = np.array([box_pos[0], box_pos[1]])
        # Robot position in world coordinates
        robot_pos_world = np.array([env_origin[0] + x, env_origin[1] + y])
        
        # Box half dimensions
        box_half_x = box_size[0] * 0.5
        box_half_y = box_size[1] * 0.5
        
        # Box AABB boundaries (in world coordinates)
        box_min_x = box_center_world[0] - box_half_x
        box_max_x = box_center_world[0] + box_half_x
        box_min_y = box_center_world[1] - box_half_y
        box_max_y = box_center_world[1] + box_half_y
        
        # Robot AABB boundaries (considering robot rectangular dimensions)
        robot_min_x = robot_pos_world[0] - ROBOT_HALF_LENGTH_X
        robot_max_x = robot_pos_world[0] + ROBOT_HALF_LENGTH_X
        robot_min_y = robot_pos_world[1] - ROBOT_HALF_WIDTH_Y
        robot_max_y = robot_pos_world[1] + ROBOT_HALF_WIDTH_Y
        
        # AABB collision check: check if the two AABBs overlap
        if (robot_min_x < box_max_x and robot_max_x > box_min_x and
            robot_min_y < box_max_y and robot_max_y > box_min_y):
            return True
    
    elif scene_type == SCENE_CYLINDER:
        # Check collision with cylinders (at fixed positions ±1, ±1)
        for cyl in handles["cyls"]:
            cyl_t = cyl["t_op"]
            cyl_geom = cyl["geom"]
            cyl_pos = cyl_t.Get()
            cyl_radius = cyl_geom.GetRadiusAttr().Get()
            
            # Cylinder center in world coordinates
            cyl_center_world = np.array([cyl_pos[0], cyl_pos[1]])
            # Robot position in world coordinates
            robot_pos_world = np.array([env_origin[0] + x, env_origin[1] + y])
            
            # Distance from robot to cylinder center
            dist_to_cyl = np.linalg.norm(robot_pos_world - cyl_center_world)
            
            # Check if robot is too close to cylinder (circle-circle collision)
            if dist_to_cyl < (cyl_radius + ROBOT_COLLISION_RADIUS):
                return True
    
    elif scene_type == SCENE_DOOR:
        # Check collision with door walls
        door_l_t = handles["door_left"]["t_op"]
        door_r_t = handles["door_right"]["t_op"]
        door_l_s = handles["door_left"]["s_op"]
        door_r_s = handles["door_right"]["s_op"]
        
        door_l_pos = door_l_t.Get()
        door_r_pos = door_r_t.Get()
        door_l_size = door_l_s.Get()
        door_r_size = door_r_s.Get()
        
        # Door walls are vertical (along y-axis), centered at x=0 in local frame
        # Check if robot is too close to either wall segment
        robot_pos_world = np.array([env_origin[0] + x, env_origin[1] + y])
        
        # Left wall (upper segment)
        wall_l_center = np.array([door_l_pos[0], door_l_pos[1]])
        wall_l_half_len = door_l_size[1] * 0.5
        wall_l_thickness = door_l_size[0] * 0.5
        
        # Check distance to left wall (AABB collision with wall segment)
        # Wall is vertical, so check x-distance (thickness) and y-distance (length) separately
        dist_to_wall_l_x = abs(robot_pos_world[0] - wall_l_center[0])
        dist_to_wall_l_y = abs(robot_pos_world[1] - wall_l_center[1])
        if dist_to_wall_l_x < (wall_l_thickness + ROBOT_HALF_LENGTH_X):
            # Check if within wall's y-range (extended by robot width)
            if dist_to_wall_l_y < (wall_l_half_len + ROBOT_HALF_WIDTH_Y):
                return True
        
        # Right wall (lower segment)
        wall_r_center = np.array([door_r_pos[0], door_r_pos[1]])
        wall_r_half_len = door_r_size[1] * 0.5
        wall_r_thickness = door_r_size[0] * 0.5
        
        # Check distance to right wall (AABB collision with wall segment)
        dist_to_wall_r_x = abs(robot_pos_world[0] - wall_r_center[0])
        dist_to_wall_r_y = abs(robot_pos_world[1] - wall_r_center[1])
        if dist_to_wall_r_x < (wall_r_thickness + ROBOT_HALF_LENGTH_X):
            # Check if within wall's y-range (extended by robot width)
            if dist_to_wall_r_y < (wall_r_half_len + ROBOT_HALF_WIDTH_Y):
                return True
    
    return False


def check_boundary_collision(robot_pos_local):
    """Check if robot collides with environment boundaries using rectangular AABB"""
    x, y = robot_pos_local[0], robot_pos_local[1]
    # Check if robot is too close to boundaries (considering robot rectangular dimensions)
    # Boundary walls are at ±ENV_HALF, robot center should be at least robot half-dimension away
    # Check x-direction (length) and y-direction (width) separately
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


def reset_env(env_id, scene_type, env_origin, handles, rng, robots=None, base_view=None, goal_offsets=None, markers=None, goal_pos=None):
    """Reset environment, TB3 robot, and goal position"""
    # Hide all scenario-specific objects first.
    _hide_prim_below_ground(handles["box"]["t_op"])
    for c in handles["cyls"]:
        _hide_prim_below_ground(c["t_op"])
    _hide_prim_below_ground(handles["door_left"]["t_op"])
    _hide_prim_below_ground(handles["door_right"]["t_op"])

    # Reset scenario objects
    if scene_type == SCENE_BOX:
        l = rng.uniform(1.0, 3.0)
        b = rng.uniform(1.0, 3.0)
        h = 0.6
        handles["box"]["s_op"].Set(Gf.Vec3d(float(l), float(b), float(h)))
        handles["box"]["t_op"].Set(Gf.Vec3d(
            float(env_origin[0]),
            float(env_origin[1]),
            h * 0.5
        ))
    elif scene_type == SCENE_CYLINDER:
        h = 0.7
        fixed_positions = [
            (1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0),
        ]
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
        handles["door_left"]["t_op"].Set(Gf.Vec3d(float(env_origin[0]), float(env_origin[1] + upper_center_local_y), wall_h * 0.5))
        handles["door_right"]["t_op"].Set(Gf.Vec3d(float(env_origin[0]), float(env_origin[1] + lower_center_local_y), wall_h * 0.5))

    # Reset TB3 robot position and orientation
    if robots is not None:
        if scene_type == SCENE_EMPTY or scene_type == SCENE_CYLINDER:
            # Empty and Cylinder: spawn at local (0, 0) with random orientation
            local_x = 0.0
            local_y = 0.0
            spawn_yaw = rng.uniform(-np.pi, np.pi)
        else:  # SCENE_BOX or SCENE_DOOR
            # Box and Door: spawn at local (2.5, 2.5), facing (-2.5, -2.5)
            local_x = 2.5
            local_y = 2.5
            dx = -2.5 - local_x
            dy = -2.5 - local_y
            spawn_yaw = np.arctan2(dy, dx)
        
        spawn_pos = np.array([
            env_origin[0] + local_x,
            env_origin[1] + local_y,
            0.0350
        ], dtype=np.float32)
        
        spawn_rot = quat_wxyz_from_yaw(np.array([spawn_yaw], dtype=np.float32))
        
        robots.set_world_poses(
            positions=spawn_pos.reshape(1, 3),
            orientations=spawn_rot.reshape(1, 4),
            indices=np.array([env_id], dtype=np.int32)
        )
        
        robots.set_velocities(
            velocities=np.zeros((1, 6), dtype=np.float32),
            indices=np.array([env_id], dtype=np.int32)
        )
        robots.set_joint_velocities(
            velocities=np.zeros((1, robots.num_dof), dtype=np.float32),
            indices=np.array([env_id], dtype=np.int32)
        )

    # Reset goal position
    if goal_offsets is not None and goal_pos is not None:
        if scene_type == SCENE_EMPTY or scene_type == SCENE_CYLINDER:
            # Random goal position (1.0-1.5m from origin)
            goal_offsets[env_id, :] = sample_goal_offsets(1, 1.0, 1.5)[0]
            goal_pos[env_id, 0] = env_origin[0] + goal_offsets[env_id, 0]
            goal_pos[env_id, 1] = env_origin[1] + goal_offsets[env_id, 1]
            goal_pos[env_id, 2] = 0.03
        else:  # SCENE_BOX or SCENE_DOOR
            # Fixed goal at local (-2.5, -2.5)
            goal_offsets[env_id, 0] = -2.5
            goal_offsets[env_id, 1] = -2.5
            goal_pos[env_id, 0] = env_origin[0] - 2.5
            goal_pos[env_id, 1] = env_origin[1] - 2.5
            goal_pos[env_id, 2] = 0.03
        
        # Update marker
        if markers is not None:
            markers[env_id].set_world_pose(
                position=goal_pos[env_id].tolist(),
                orientation=[1, 0, 0, 0]
            )


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
    env_spacing = env_size + env_gap  # 8m
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

    # Load TB3 robots
    for i in range(num_envs):
        tb3_root = f"/World/envs/env_{i}/TB3"
        add_reference_to_stage(usd_path=TB3_USD, prim_path=tb3_root)

    # Let references expand
    for _ in range(180):
        world.step(render=True)

    # Create ArticulationView for robots
    robots = ArticulationView(
        prim_paths_expr="/World/envs/env_.*/TB3/a__namespace_base_footprint",
        name="tb3_view",
        reset_xform_properties=False
    )
    world.scene.add(robots)

    # Create RigidPrimView for base_link
    base_view = RigidPrimView(
        prim_paths_expr="/World/envs/env_.*/TB3/a__namespace_base_link",
        name="base_view",
        reset_xform_properties=False,
    )
    world.scene.add(base_view)

    world.reset()

    if robots.count != num_envs:
        print(f"[WARN] robots.count={robots.count}, expected={num_envs}.")

    # Configure wheel joints
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
    else:
        wheel_kp = 0.0
        wheel_kd = 260.0
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
            print(f"[WARN] set_max_efforts failed: {e}")

    # Configure base mass and COM
    base_mass = 2.2
    com_z = -0.10
    com_x = -0.08
    
    masses = np.ones(num_envs, dtype=np.float32) * base_mass
    coms = np.zeros((num_envs, 3), dtype=np.float32)
    coms[:, 0] = com_x
    coms[:, 2] = com_z
    
    try:
        base_view.set_masses(masses)
        base_view.set_coms(coms)
        print(f"[INFO] Set base mass={base_mass}kg, COM_x={com_x}m, COM_z={com_z}m")
    except Exception as e:
        print(f"[WARN] Failed to set mass and COM: {e}")

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

    # Initialize goal positions and markers
    goal_offsets = np.zeros((num_envs, 2), dtype=np.float32)
    goal_pos = np.zeros((num_envs, 3), dtype=np.float32)
    markers = []
    for i in range(num_envs):
        if scene_types[i] == SCENE_EMPTY or scene_types[i] == SCENE_CYLINDER:
            goal_offsets[i, :] = sample_goal_offsets(1, 1.0, 1.5)[0]
            goal_pos[i, 0] = env_origins[i, 0] + goal_offsets[i, 0]
            goal_pos[i, 1] = env_origins[i, 1] + goal_offsets[i, 1]
        else:  # BOX or DOOR
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

    # Initial reset of all environments
    episode_start_time = np.zeros((num_envs,), dtype=np.float32)
    for i in range(num_envs):
        reset_env(i, scene_types[i], env_origins[i], handles[i], rng, robots, base_view, goal_offsets, markers, goal_pos)
        episode_start_time[i] = 0.0  # Initialize episode start time

    # Control parameters
    forward_v = 0.26
    k_yaw = 4.0  # Further increased for stronger turning response
    max_w = 2.0  # Further increased to allow larger angular velocity for tighter turns
    reach_thresh = 0.18

    # Velocity ramp-up
    warmup_steps = int(0.1 / physics_dt)
    ramp_steps = int(1.0 / physics_dt)
    local_step = np.zeros((num_envs,), dtype=np.int64)

    # Timing for timeout
    current_time = 0.0

    loops = np.zeros((num_envs,), dtype=np.int64)
    last_print = time.time()

    print("[INFO] 4 scenarios initialized with TB3 robots")
    print("[INFO] env size=6x6m, env gap=2m, env spacing=8m")
    print("[INFO] scenes: env0=empty, env1=box, env2=cylinder, env3=door")
    print("[INFO] Reset conditions: reach goal, collision, or 60s timeout")

    while simulation_app.is_running():
        local_step += 1
        current_time += physics_dt

        # Velocity ramp-up
        v_cmd_raw = forward_v
        alpha = np.clip((local_step - warmup_steps) / float(max(1, ramp_steps)), 0.0, 1.0).astype(np.float32)
        v_cmd = alpha * v_cmd_raw

        # Get robot poses
        pos, rot = robots.get_world_poses()
        yaw = yaw_from_quat_wxyz(rot)

        # Calculate local positions
        x_local = (pos[:, 0] - env_origins[:, 0]).astype(np.float32)
        y_local = (pos[:, 1] - env_origins[:, 1]).astype(np.float32)

        # Calculate goal errors
        gx_local = goal_offsets[:, 0].astype(np.float32)
        gy_local = goal_offsets[:, 1].astype(np.float32)
        ex = gx_local - x_local
        ey = gy_local - y_local
        dist = np.sqrt(ex * ex + ey * ey).astype(np.float32)

        goal_heading = np.arctan2(ey, ex)
        yaw_err = wrap_to_pi(goal_heading - yaw)

        # Reduce speed when turning to improve maneuverability
        # Use cubic function for more aggressive reduction at large angles
        # Small angle (< 30°): almost no speed reduction
        # 90°: significant reduction, 180°: very aggressive reduction
        abs_yaw_err = np.abs(yaw_err)
        normalized_err = abs_yaw_err / np.pi  # 0 to 1
        # Use cubic function: more aggressive reduction for large angles
        # 90° (0.5): ~25% reduction, 180° (1.0): ~60% reduction
        speed_reduction = 0.6 * (normalized_err ** 3)  # Cubic: small errors -> minimal, large errors -> significant
        speed_factor = 1.0 - speed_reduction
        speed_factor = np.clip(speed_factor, 0.3, 1.0)  # Minimum 30% speed (allow more reduction for 180°)
        v_use = v_cmd * speed_factor

        # Dynamic angular velocity limit: allow larger angular velocity when speed is reduced for turning
        # Base limit ensures both wheels rotate forward: w < 2*v/WHEEL_BASE
        max_w_dynamic_base = 2.0 * v_use / WHEEL_BASE  # Physical constraint based on actual speed
        # When speed is reduced for turning, allow slightly larger angular velocity
        # Only apply multiplier when speed reduction is moderate (speed_factor > 0.5)
        # This improves turning radius while ensuring safety (no wheel reversal)
        # Use vectorized operations for multiple environments
        speed_reduction_factor = 1.0 - speed_factor  # How much speed was reduced
        # Conservative multiplier: up to 5% more when speed reduced moderately
        angular_multiplier = np.where(
            speed_factor > 0.5,
            1.0 + 0.05 * speed_reduction_factor,  # Apply multiplier when speed_factor > 0.5
            1.0  # Use base limit when speed is very low (large turn)
        )
        max_w_dynamic = (max_w_dynamic_base * angular_multiplier).astype(np.float32)
        # When close to goal, allow larger angular velocity for better maneuverability
        # This prevents circling around the goal
        close_to_goal = dist < 1.0  # Within 1m of goal
        max_w_close = np.where(
            close_to_goal,
            np.minimum(max_w, max_w_dynamic * 1.4),  # Allow 40% more angular velocity when close
            max_w_dynamic
        )
        max_w_effective = np.minimum(max_w, max_w_close).astype(np.float32)

        w_cmd = np.clip(k_yaw * yaw_err, -max_w_effective, max_w_effective).astype(np.float32)

        # Differential drive
        left_w = (v_use - w_cmd * (WHEEL_BASE / 2.0)) / WHEEL_RADIUS
        right_w = (v_use + w_cmd * (WHEEL_BASE / 2.0)) / WHEEL_RADIUS

        # Safety limits
        max_wheel_vel = 15.0
        min_wheel_vel = 0.0
        left_w = np.clip(left_w, min_wheel_vel, max_wheel_vel).astype(np.float32)
        right_w = np.clip(right_w, min_wheel_vel, max_wheel_vel).astype(np.float32)

        wheel_diff = np.abs(left_w - right_w)
        wheel_sum = np.abs(left_w) + np.abs(right_w)
        problematic = wheel_diff > 0.5 * wheel_sum
        if np.any(problematic):
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

        # Check reset conditions after step
        pos2, rot2 = robots.get_world_poses()
        x_local2 = (pos2[:, 0] - env_origins[:, 0]).astype(np.float32)
        y_local2 = (pos2[:, 1] - env_origins[:, 1]).astype(np.float32)
        ex2 = gx_local - x_local2
        ey2 = gy_local - y_local2
        dist2 = np.sqrt(ex2 * ex2 + ey2 * ey2).astype(np.float32)

        # Check reset conditions
        reached = dist2 <= reach_thresh
        collision = np.zeros((num_envs,), dtype=bool)
        timeout = np.zeros((num_envs,), dtype=bool)

        for i in range(num_envs):
            # Check boundary collision
            if check_boundary_collision(np.array([x_local2[i], y_local2[i]])):
                collision[i] = True
            # Check obstacle collision
            elif check_collision_with_obstacles(
                np.array([x_local2[i], y_local2[i]]),
                scene_types[i],
                handles[i],
                env_origins[i],
                rng
            ):
                collision[i] = True
            # Check timeout
            if current_time - episode_start_time[i] >= TIMEOUT_SECONDS:
                timeout[i] = True

        # Reset if any condition is met
        to_reset = reached | collision | timeout
        if np.any(to_reset):
            ids = np.nonzero(to_reset)[0]

            # Stop robots
            robots.set_joint_velocity_targets(np.zeros((num_envs, robots.num_dof), dtype=np.float32))
            v_zero = np.zeros((num_envs, 6), dtype=np.float32)
            robots.set_velocities(v_zero)

            # Reset environments
            for i in ids:
                reset_env(i, scene_types[i], env_origins[i], handles[i], rng, robots, base_view, goal_offsets, markers, goal_pos)
                episode_start_time[i] = current_time
                local_step[i] = 0
                loops[i] += 1

            robots.set_velocities(v_zero)
            robots.set_joint_velocities(np.zeros((num_envs, robots.num_dof), dtype=np.float32))

        now = time.time()
        if now - last_print > 0.6:
            reset_reasons = []
            for i in range(num_envs):
                if reached[i]:
                    reset_reasons.append("goal")
                elif collision[i]:
                    reset_reasons.append("collision")
                elif timeout[i]:
                    reset_reasons.append("timeout")
                else:
                    reset_reasons.append("running")
            
            print(
                f"E0: dist={dist2[0]:.2f} {reset_reasons[0]} loops={loops[0]} | "
                f"E1: dist={dist2[1]:.2f} {reset_reasons[1]} loops={loops[1]} | "
                f"E2: dist={dist2[2]:.2f} {reset_reasons[2]} loops={loops[2]} | "
                f"E3: dist={dist2[3]:.2f} {reset_reasons[3]} loops={loops[3]}"
            )
            last_print = now

        time.sleep(0.001)

    simulation_app.close()


if __name__ == "__main__":
    main()
