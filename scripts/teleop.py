import time
import numpy as np
import carb

from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

import omni.usd
from pxr import Gf, UsdGeom, UsdPhysics
from omni.isaac.core import World
from omni.isaac.core.physics_context import PhysicsContext
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.objects import VisualSphere
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController

# ===================== 全局配置 =====================
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

# 键盘控制指令
keyboard_cmd = {
    "forward": 0.0,  # W/S: 前进/后退 (-1.0 ~ 1.0)
    "rotate": 0.0    # A/D: 左转/右转 (-1.0 ~ 1.0)
}

# ===================== 工具函数 =====================
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

def reset_env(scene_type, env_origin, handles, rng, robots=None, goal_pos=None, marker=None):
    """重置单机环境"""
    # 隐藏所有障碍物
    _hide_prim_below_ground(handles["box"]["t_op"])
    for c in handles["cyls"]:
        _hide_prim_below_ground(c["t_op"])
    _hide_prim_below_ground(handles["door_left"]["t_op"])
    _hide_prim_below_ground(handles["door_right"]["t_op"])

    # 生成场景
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

    # 重置机器人位姿
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

        robots.set_world_poses(
            positions=spawn_pos.reshape(1, 3),
            orientations=spawn_rot.reshape(1, 4),
            indices=np.array([0], dtype=np.int32),
        )
        robots.set_velocities(
            velocities=np.zeros((1, 6), dtype=np.float32),
            indices=np.array([0], dtype=np.int32),
        )
        robots.set_joint_velocities(
            velocities=np.zeros((1, robots.num_dof), dtype=np.float32),
            indices=np.array([0], dtype=np.int32),
        )
        robots.set_joint_velocity_targets(
            np.zeros((1, robots.num_dof), dtype=np.float32),
            indices=np.array([0], dtype=np.int32),
        )

    # 重置目标点
    if goal_pos is not None and marker is not None:
        if scene_type == SCENE_EMPTY or scene_type == SCENE_CYLINDER:
            ang = rng.uniform(-np.pi, np.pi)
            r = rng.uniform(1.0, 1.5)
            goal_pos[0] = env_origin[0] + r * np.cos(ang)
            goal_pos[1] = env_origin[1] + r * np.sin(ang)
        else:
            goal_pos[0] = env_origin[0] - 2.5
            goal_pos[1] = env_origin[1] - 2.5
        goal_pos[2] = 0.03
        marker.set_world_pose(position=goal_pos.tolist(), orientation=[1, 0, 0, 0])

def on_keyboard_event(event, *args, **kwargs):
    """键盘事件回调"""
    global keyboard_cmd
    key_name_map = {
        carb.input.KeyboardInput.W: "W",
        carb.input.KeyboardInput.S: "S",
        carb.input.KeyboardInput.A: "A",
        carb.input.KeyboardInput.D: "D"
    }

    if event.input not in key_name_map:
        return

    # 按键按下
    if event.type == carb.input.KeyboardEventType.KEY_PRESS:
        key_name = key_name_map[event.input]
        print(f"【按键按下】{key_name}键")

        if event.input == carb.input.KeyboardInput.W:
            keyboard_cmd["forward"] = 1.0
        elif event.input == carb.input.KeyboardInput.S:
            keyboard_cmd["forward"] = -1.0
        elif event.input == carb.input.KeyboardInput.A:
            keyboard_cmd["rotate"] = 1.0
        elif event.input == carb.input.KeyboardInput.D:
            keyboard_cmd["rotate"] = -1.0

    # 按键松开
    elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
        key_name = key_name_map[event.input]
        print(f"【按键松开】{key_name}键")

        if event.input in [carb.input.KeyboardInput.W, carb.input.KeyboardInput.S]:
            keyboard_cmd["forward"] = 0.0
        elif event.input in [carb.input.KeyboardInput.A, carb.input.KeyboardInput.D]:
            keyboard_cmd["rotate"] = 0.0

    # 打印当前指令状态
    print(f"  → 控制指令: 前进/后退={keyboard_cmd['forward']}, 左转/右转={keyboard_cmd['rotate']}")

# ===================== 主函数 =====================
def main():
    # 初始化随机数
    rng = np.random.default_rng(42)
    np.random.seed(42)

    # 物理配置
    physics_dt = 1.0 / 240.0
    render_dt = 1.0 / 60.0
    world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=render_dt)
    PhysicsContext().substeps = 8
    world.scene.add_default_ground_plane()
    stage = world.scene.stage

    # 单机环境配置
    num_envs = 1
    env_size = 6.0
    env_gap = 2.0
    env_spacing = env_size + env_gap
    wall_t = 0.08
    wall_h = 1.2

    # 选择场景类型（可修改：empty/box/cylinder/door）
    scene_type = SCENE_EMPTY
    show_visual_walls = True

    # 环境原点（单机固定在0,0）
    env_origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # 加载TB3小车
    tb3_root = f"/World/envs/env_0/TB3"
    add_reference_to_stage(usd_path=TB3_USD, prim_path=tb3_root)

    # 等待资产加载
    for _ in range(180):
        world.step(render=True)

    # 创建机器人视图
    robots = ArticulationView(
        prim_paths_expr="/World/envs/env_.*/TB3/a__namespace_base_footprint",
        name="tb3_view",
        reset_xform_properties=False,
    )
    world.scene.add(robots)
    world.reset()
    robots.initialize()

    # 设置小车质量
    apply_massapi_all_tb3(base_mass=2.2, com_x=-0.08, com_z=-0.10)
    for _ in range(10):
        world.step(render=True)

    # 找到车轮关节索引
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
        print("[ERROR] Could not find wheel joints in dof_names.")
        print(dof_names)
        simulation_app.close()
        return

    # 差速控制器
    diff_ctrl = DifferentialController(
        name="tb3_diff_ctrl",
        wheel_radius=WHEEL_RADIUS,
        wheel_base=WHEEL_BASE,
        max_linear_speed=0.26,
        max_angular_speed=1.6,
    )

    # 车轮控制参数
    wheel_kp = 0.0
    wheel_kd = 120.0
    wheel_max_effort = 80.0

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

    # 创建场景障碍物
    root = f"/World/envs/env_0"
    half = env_size * 0.5

    # 创建边界墙
    walls = []
    wall_defs = [
        (f"{root}/BoundLeft", (wall_t, env_size + 2 * wall_t, wall_h),
         (env_origin[0] - half - wall_t * 0.5, env_origin[1], wall_h * 0.5)),
        (f"{root}/BoundRight", (wall_t, env_size + 2 * wall_t, wall_h),
         (env_origin[0] + half + wall_t * 0.5, env_origin[1], wall_h * 0.5)),
        (f"{root}/BoundFront", (env_size + 2 * wall_t, wall_t, wall_h),
         (env_origin[0], env_origin[1] - half - wall_t * 0.5, wall_h * 0.5)),
        (f"{root}/BoundBack", (env_size + 2 * wall_t, wall_t, wall_h),
         (env_origin[0], env_origin[1] + half + wall_t * 0.5, wall_h * 0.5)),
    ]
    for p, size, center in wall_defs:
        prim, t_op, s_op = _create_cube(
            stage, p, size_xyz=size, center_xyz=center,
            visible=show_visual_walls, color=(0.6, 0.6, 0.6), collision=True
        )
        walls.append({"prim": prim, "t_op": t_op, "s_op": s_op})

    # 创建箱子障碍物
    box_prim, box_t, box_s = _create_cube(
        stage, f"{root}/ScenarioBox",
        size_xyz=(1.0, 1.0, 0.6), center_xyz=(env_origin[0], env_origin[1], -10.0),
        visible=True, color=(0.85, 0.85, 0.85), collision=True
    )

    # 创建圆柱障碍物
    cyls = []
    for k in range(4):
        c_prim, c_t, c_geom = _create_cylinder(
            stage, f"{root}/ScenarioCyl_{k}",
            radius=0.18, height=0.7, center_xyz=(env_origin[0], env_origin[1], -10.0),
            visible=True, color=(0.7, 0.7, 0.7), collision=True
        )
        cyls.append({"prim": c_prim, "t_op": c_t, "geom": c_geom})

    # 创建门场景障碍物
    door_l_prim, door_l_t, door_l_s = _create_cube(
        stage, f"{root}/DoorSegLeft",
        size_xyz=(0.12, 2.0, 1.0), center_xyz=(env_origin[0], env_origin[1], -10.0),
        visible=True, color=(0.7, 0.7, 0.7), collision=True
    )
    door_r_prim, door_r_t, door_r_s = _create_cube(
        stage, f"{root}/DoorSegRight",
        size_xyz=(0.12, 2.0, 1.0), center_xyz=(env_origin[0], env_origin[1], -10.0),
        visible=True, color=(0.7, 0.7, 0.7), collision=True
    )

    # 设置障碍物可见性
    box_visible = True if scene_type == SCENE_BOX else show_visual_walls
    _set_visibility(box_prim, box_visible)
    for c in cyls:
        _set_visibility(c["prim"], show_visual_walls)
    _set_visibility(door_l_prim, show_visual_walls)
    _set_visibility(door_r_prim, show_visual_walls)

    # 场景句柄
    handles = {
        "walls": walls,
        "box": {"prim": box_prim, "t_op": box_t, "s_op": box_s},
        "cyls": cyls,
        "door_left": {"prim": door_l_prim, "t_op": door_l_t, "s_op": door_l_s},
        "door_right": {"prim": door_r_prim, "t_op": door_r_t, "s_op": door_r_s},
    }

    # 创建目标点标记
    goal_pos = np.zeros(3, dtype=np.float32)
    marker = world.scene.add(
        VisualSphere(
            prim_path=f"/World/envs/env_0/GoalMarker",
            name=f"goal_marker_0",
            position=goal_pos.tolist(),
            radius=0.05,
            color=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        )
    )

    # 重置环境
    reset_env(scene_type, env_origin, handles, rng, robots, goal_pos, marker)

    # 注册键盘回调
    input_interface = carb.input.acquire_input_interface()
    app_window = omni.appwindow.get_default_app_window()
    keyboard = app_window.get_keyboard()
    if keyboard is None:
        raise RuntimeError("无法获取键盘设备，无法进行控制")
    keyboard_sub = input_interface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)

    # 控制参数
    max_linear_vel = 0.26  # 最大线速度
    max_angular_vel = 1.6   # 最大角速度
    SETTLE_STEPS = 8
    episode_start_time = time.time()
    last_print = time.time()

    # 打印控制说明
    print("\n===== 控制说明 =====")
    print("W: 前进 | S: 后退 | A: 左转 | D: 右转")
    print("松开按键自动停止 | 关闭窗口退出程序")
    print(f"当前场景: {scene_type}")
    print("====================\n")

    # 主仿真循环
    while simulation_app.is_running():
        current_time = time.time() - episode_start_time

        # 获取键盘指令并映射到实际速度
        forward_cmd = keyboard_cmd["forward"] * max_linear_vel
        rotate_cmd = keyboard_cmd["rotate"] * max_angular_vel

        # 差速控制器计算车轮速度
        action = diff_ctrl.forward(command=np.array([forward_cmd, rotate_cmd], dtype=np.float32))
        wl = float(action.joint_velocities[0])
        wr = float(action.joint_velocities[1])

        # 设置车轮速度目标
        targets = np.zeros((1, robots.num_dof), dtype=np.float32)
        targets[0, left_idx] = wl
        targets[0, right_idx] = wr
        robots.set_joint_velocity_targets(targets)

        # 步进仿真
        world.step(render=True)

        # 碰撞检测
        pos, _ = robots.get_world_poses()
        x_local = pos[0, 0] - env_origin[0]
        y_local = pos[0, 1] - env_origin[1]
        boundary_collision = check_boundary_collision(np.array([x_local, y_local]))
        obstacle_collision = check_collision_with_obstacles(
            np.array([x_local, y_local]), scene_type, handles, env_origin
        )
        collision = boundary_collision or obstacle_collision

        # 超时检测
        timeout = current_time >= TIMEOUT_SECONDS

        # 需要重置的情况
        to_reset = collision or timeout
        if to_reset:
            print(f"\n[重置触发] 原因: {'碰撞' if collision else '超时'}")
            # 停止小车
            robots.set_joint_velocity_targets(np.zeros((1, robots.num_dof), dtype=np.float32))
            robots.set_velocities(np.zeros((1, 6), dtype=np.float32))
            robots.set_joint_velocities(np.zeros((1, robots.num_dof), dtype=np.float32))

            # 重置环境
            reset_env(scene_type, env_origin, handles, rng, robots, goal_pos, marker)
            episode_start_time = time.time()

            # 稳定场景
            for _ in range(SETTLE_STEPS):
                robots.set_joint_velocity_targets(np.zeros((1, robots.num_dof), dtype=np.float32))
                robots.set_velocities(np.zeros((1, 6), dtype=np.float32))
                robots.set_joint_velocities(np.zeros((1, robots.num_dof), dtype=np.float32))
                world.step(render=True)

        # 打印状态信息
        now = time.time()
        if now - last_print > 0.35:
            dof_vel = robots.get_joint_velocities()
            l_act = float(dof_vel[0, left_idx])
            r_act = float(dof_vel[0, right_idx])
            base_v = robots.get_velocities()
            vx0, vy0, wz0 = float(base_v[0, 0]), float(base_v[0, 1]), float(base_v[0, 5])

            print(
                f"[状态] 位置(X,Y)=({pos[0,0]:.3f},{pos[0,1]:.3f}) | "
                f"速度(vx,vy,wz)=({vx0:+.3f},{vy0:+.3f},{wz0:+.3f}) | "
                f"车轮(指令/实际): L=({wl:+.2f}/{l_act:+.2f}) R=({wr:+.2f}/{r_act:+.2f}) | "
                f"碰撞={collision} 超时={timeout}"
            )
            last_print = now

        time.sleep(0.001)

    # 清理资源
    input_interface.unsubscribe_from_keyboard_events(keyboard, keyboard_sub)
    simulation_app.close()

if __name__ == "__main__":
    main()