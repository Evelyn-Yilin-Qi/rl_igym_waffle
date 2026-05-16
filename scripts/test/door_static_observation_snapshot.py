"""
Door scene static observation snapshot script.

Loads a single DOOR scene env, keeps TB3 static at spawn,
sets goal to local (2.0, -1.0), then exports one PNG that shows:
1) local map geometry,
2) 36-ray LiDAR scan,
3) user intent vector (robot -> goal),
4) per-ray distance chart.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf
from omni.isaac.kit import SimulationApp


# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# Isaac Sim app
simulation_app = SimulationApp({"headless": False})


from sim.scenes import SCENE_DOOR, yaw_from_quat_wxyz, quat_wxyz_from_yaw
from process_settings.env_setup import EnvironmentSetup
from envs.observations import compute_lidar_ranges, get_base_velocity_from_tensor, assemble_observations


def load_config(cfg_path):
    """Load config file from common candidate paths."""
    candidates = []
    if os.path.isabs(cfg_path):
        candidates.append(cfg_path)
    else:
        candidates.extend([
            os.path.join(PROJECT_ROOT, cfg_path),
            os.path.abspath(cfg_path),
            os.path.join(os.getcwd(), cfg_path),
        ])

    found = None
    for path in candidates:
        if os.path.exists(path):
            found = os.path.abspath(path)
            break

    if found is None:
        raise FileNotFoundError(f"Config file not found. Tried: {candidates}")

    return OmegaConf.load(found)


def draw_rect(ax, center_xy, size_xy, color, alpha=0.35, label=None):
    """Draw axis-aligned rectangle defined by center and size."""
    cx, cy = center_xy
    sx, sy = size_xy
    x0 = cx - sx * 0.5
    y0 = cy - sy * 0.5
    rect = plt.Rectangle((x0, y0), sx, sy, facecolor=color, edgecolor="k", alpha=alpha, label=label)
    ax.add_patch(rect)


def save_observation_figures(
    map_output_path,
    lidar_output_path,
    env_size,
    robot_local_xy,
    robot_yaw,
    goal_local_xy,
    lidar_ranges,
    lidar_max_range,
    user_intent_env,
    user_intent_ego,
    door_left_center_local,
    door_left_size,
    door_right_center_local,
    door_right_size,
    extra_cyl_center_local,
    extra_cyl_radius,
):
    """Save two PNGs: map view and LiDAR-only chart."""
    # ===== Figure 1: local map =====
    fig_map, ax_map = plt.subplots(1, 1, figsize=(8, 8))

    # ===== Left: local map + lidar + intent =====
    env_half = env_size * 0.5
    ax_map.set_title("Door Scene Local Map + LiDAR + User Intent")

    # Environment boundary
    boundary = plt.Rectangle(
        (-env_half, -env_half),
        env_size,
        env_size,
        fill=False,
        edgecolor="black",
        linewidth=2.0,
        label="Env Boundary",
    )
    ax_map.add_patch(boundary)

    # Door segments (actual randomized size/position)
    draw_rect(
        ax_map,
        center_xy=door_left_center_local,
        size_xy=door_left_size,
        color="#7f8c8d",
        alpha=0.45,
        label="Door Upper Segment",
    )
    draw_rect(
        ax_map,
        center_xy=door_right_center_local,
        size_xy=door_right_size,
        color="#95a5a6",
        alpha=0.45,
        label="Door Lower Segment",
    )
    extra_cyl = plt.Circle(
        (float(extra_cyl_center_local[0]), float(extra_cyl_center_local[1])),
        float(extra_cyl_radius),
        facecolor="#c0392b",
        edgecolor="black",
        alpha=0.7,
        label="Extra Cylinder r=0.1m",
    )
    ax_map.add_patch(extra_cyl)

    # Robot and goal
    ax_map.scatter([robot_local_xy[0]], [robot_local_xy[1]], c="tab:blue", s=70, label="Robot")
    ax_map.scatter([goal_local_xy[0]], [goal_local_xy[1]], c="tab:red", s=70, marker="x", label="Goal (-2.0, -2.0)")

    # Robot heading
    heading_len = 0.35
    ax_map.arrow(
        robot_local_xy[0],
        robot_local_xy[1],
        heading_len * np.cos(robot_yaw),
        heading_len * np.sin(robot_yaw),
        width=0.01,
        head_width=0.08,
        head_length=0.08,
        color="tab:blue",
        length_includes_head=True,
    )

    # LiDAR rays
    num_rays = lidar_ranges.shape[0]
    angles = np.linspace(0.0, 2.0 * np.pi, num_rays, endpoint=False)
    for i, angle in enumerate(angles):
        ray_theta = robot_yaw + angle
        ray_len = float(lidar_ranges[i])
        x_end = robot_local_xy[0] + ray_len * np.cos(ray_theta)
        y_end = robot_local_xy[1] + ray_len * np.sin(ray_theta)
        ax_map.plot(
            [robot_local_xy[0], x_end],
            [robot_local_xy[1], y_end],
            color="tab:green",
            alpha=0.6,
            linewidth=1.0,
            label="LiDAR Rays" if i == 0 else None,
        )
        ax_map.scatter([x_end], [y_end], c="tab:green", s=8, alpha=0.7)

    # Map arrow uses env-frame direction for geometric visualization.
    # Model input intent is ego-frame and shown in text/terminal.
    intent_len = 0.6
    ax_map.arrow(
        robot_local_xy[0],
        robot_local_xy[1],
        intent_len * float(user_intent_env[0]),
        intent_len * float(user_intent_env[1]),
        width=0.01,
        head_width=0.08,
        head_length=0.08,
        color="tab:orange",
        label="User Intent (normalized)",
        length_includes_head=True,
    )
    # Put intent text on the side away from legend/right margin.
    # If robot is near the right side, place text to the left.
    intent_text_dx = -1.35 if robot_local_xy[0] > (env_half * 0.3) else 0.15
    ax_map.text(
        robot_local_xy[0] + intent_text_dx,
        robot_local_xy[1] - 0.18,
        f"intent_ego=({float(user_intent_ego[0]):.3f}, {float(user_intent_ego[1]):.3f})",
        fontsize=9,
        color="tab:orange",
        bbox=dict(facecolor="white", edgecolor="tab:orange", alpha=0.85, boxstyle="round,pad=0.2"),
    )

    ax_map.set_xlim(-env_half - 0.5, env_half + 0.5)
    ax_map.set_ylim(-env_half - 0.5, env_half + 0.5)
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.set_xlabel("Local X (m)")
    ax_map.set_ylabel("Local Y (m)")
    ax_map.grid(True, alpha=0.3)
    # Keep legend fully outside map area to avoid any overlap.
    ax_map.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=2,
        fontsize=9,
        frameon=True,
    )
    fig_map.tight_layout(rect=[0.0, 0.1, 1.0, 1.0])
    fig_map.savefig(map_output_path, dpi=180)
    plt.close(fig_map)

    # ===== Figure 2: LiDAR-only chart (legend here, not on map) =====
    fig_lidar, ax_scan = plt.subplots(1, 1, figsize=(10, 4.5))
    ax_scan.set_title("36-D LiDAR Observation")
    ray_indices = np.arange(num_rays)
    ax_scan.plot(ray_indices, lidar_ranges, marker="o", linewidth=1.5, markersize=3, color="tab:green")
    ax_scan.axhline(lidar_max_range, linestyle="--", linewidth=1.0, color="gray")
    ax_scan.set_xlabel("Ray Index")
    ax_scan.set_ylabel("Distance (m)")
    ax_scan.set_xticks(np.arange(0, num_rays, 3))
    ax_scan.set_ylim(0.0, lidar_max_range * 1.05)
    ax_scan.grid(True, alpha=0.3)

    fig_lidar.tight_layout()
    fig_lidar.savefig(lidar_output_path, dpi=180)
    plt.close(fig_lidar)


def compute_user_intent_vectors(robot_local_xy, goal_local_xy, robot_yaw):
    """
    Compute user intent with explicit body-frame definition:
      - ego x+: robot forward
      - ego y+: robot left
      - unit length
    """
    dir_env = np.asarray(goal_local_xy, dtype=np.float32) - np.asarray(robot_local_xy, dtype=np.float32)
    dist = float(np.linalg.norm(dir_env))
    if dist > 1e-6:
        env_unit = dir_env / dist
    else:
        env_unit = np.zeros(2, dtype=np.float32)

    # Ego basis in env frame
    c = float(np.cos(robot_yaw))
    s = float(np.sin(robot_yaw))
    forward_env = np.array([c, s], dtype=np.float32)      # ego x+ axis in env frame
    left_env = np.array([-s, c], dtype=np.float32)        # ego y+ axis in env frame

    # Project env direction onto ego axes => model input intent (x_forward, y_left)
    ego_unit = np.array([
        float(np.dot(env_unit, forward_env)),
        float(np.dot(env_unit, left_env)),
    ], dtype=np.float32)
    return env_unit, ego_unit


def main():
    env_cfg = load_config("cfg/WaffleDrive.yaml")

    # Deterministic random seed
    rng = np.random.default_rng(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # Single DOOR env
    num_envs = 1
    scene_types = [SCENE_DOOR]

    env_setup = EnvironmentSetup(
        env_cfg=env_cfg,
        num_envs=num_envs,
        scene_types=scene_types,
        simulation_app=simulation_app,
        rng=rng,
        show_visual_walls=True,
        reset_scene_obstacles=True,
    )
    setup = env_setup.setup_all()

    world = setup["world"]
    robots = setup["robots"]
    scene_manager = setup["scene_manager"]
    env_origins = setup["env_origins"]

    physics_dt = float(env_cfg.sim.dt)
    lidar_num_rays = int(env_cfg.env.lidar.num_rays)
    lidar_max_range = float(env_cfg.env.lidar.max_range)
    max_v = float(env_cfg.env.robot_limits.max_v)
    max_w = float(env_cfg.env.robot_limits.max_w)

    # Reset robot to scene default spawn and keep static
    spawn_pos, spawn_yaw = scene_manager.get_robot_spawn_config(0, rng=rng)
    spawn_rot = quat_wxyz_from_yaw(np.array([spawn_yaw], dtype=np.float32))
    idx = np.array([0], dtype=np.int32)
    robots.set_world_poses(
        positions=spawn_pos.reshape(1, 3),
        orientations=spawn_rot.reshape(1, 4),
        indices=idx,
    )
    robots.set_velocities(np.zeros((1, 6), dtype=np.float32), indices=idx)
    robots.set_joint_velocities(np.zeros((1, robots.num_dof), dtype=np.float32), indices=idx)
    robots.set_joint_velocity_targets(np.zeros((1, robots.num_dof), dtype=np.float32), indices=idx)

    # Goal fixed at local (-2.0, -2.0)
    goal_pos = np.zeros((1, 3), dtype=np.float32)
    goal_pos[0, 0] = env_origins[0, 0] - 2.0
    goal_pos[0, 1] = env_origins[0, 1] - 2.0
    goal_pos[0, 2] = 0.03

    # Update goal marker to fixed goal
    setup["markers"][0].set_world_pose(position=goal_pos[0].tolist(), orientation=[1, 0, 0, 0])

    # Add extra static cylinder obstacle at local (2.0, 1.0), radius=0.1m
    extra_cyl_center_local = np.array([2.0, 1.0], dtype=np.float32)
    extra_cyl_radius = 0.18
    extra_cyl_height = 0.7
    extra_cyl_center_world = (
        float(env_origins[0, 0] + extra_cyl_center_local[0]),
        float(env_origins[0, 1] + extra_cyl_center_local[1]),
        float(extra_cyl_height * 0.5),
    )
    scene_manager._create_cylinder(
        scene_manager.stage,
        "/World/envs/env_0/ExtraStaticCylinder",
        radius=extra_cyl_radius,
        height=extra_cyl_height,
        center_xyz=extra_cyl_center_world,
        visible=True,
        color=(0.75, 0.15, 0.15),
        collision=True,
    )

    # Wait a few seconds for stabilization (static scene / static robot)
    settle_steps = int(3.0 / physics_dt)
    for _ in range(settle_steps):
        robots.set_joint_velocity_targets(np.zeros((1, robots.num_dof), dtype=np.float32))
        robots.set_velocities(np.zeros((1, 6), dtype=np.float32))
        robots.set_joint_velocities(np.zeros((1, robots.num_dof), dtype=np.float32))
        world.step(render=True)

    # Collect one observation snapshot
    pos, rot = robots.get_world_poses()
    yaw = yaw_from_quat_wxyz(rot)
    yaw_arr = np.atleast_1d(np.asarray(yaw, dtype=np.float32))
    yaw0 = float(yaw_arr[0])
    lidar_ranges = compute_lidar_ranges(
        robot_positions=pos,
        robot_orientations=rot,
        env_origins=env_origins,
        lidar_num_rays=lidar_num_rays,
        lidar_max_range=lidar_max_range,
        scene_manager=scene_manager,
    )

    robot_velocities_np = robots.get_velocities()
    robot_velocities_torch = torch.from_numpy(robot_velocities_np).float()
    base_vel = get_base_velocity_from_tensor(robot_velocities_torch, max_v=max_v, max_w=max_w)
    action_history = np.zeros((1, 4), dtype=np.float32)

    robot_local_xy = (pos[0, :2] - env_origins[0, :2]).astype(np.float32)
    goal_local_xy = (goal_pos[0, :2] - env_origins[0, :2]).astype(np.float32)
    user_intent_env_np, user_intent_ego_np = compute_user_intent_vectors(
        robot_local_xy=robot_local_xy,
        goal_local_xy=goal_local_xy,
        robot_yaw=yaw0,
    )

    obs = assemble_observations(
        robot_positions=pos,
        robot_orientations=rot,
        goal_positions=goal_pos,
        env_origins=env_origins,
        lidar_ranges=lidar_ranges,
        base_vel=base_vel,
        action_history=action_history,
        max_v=max_v,
        max_w=max_w,
        lidar_max_range=lidar_max_range,
        user_input=user_intent_ego_np.reshape(1, 2),
    )

    # Extract current randomized door geometry in local coordinates
    handles = scene_manager.scene_handles[0]
    door_left_center_world = np.array(handles["door_left"]["t_op"].Get(), dtype=np.float32)
    door_left_size = np.array(handles["door_left"]["s_op"].Get(), dtype=np.float32)
    door_right_center_world = np.array(handles["door_right"]["t_op"].Get(), dtype=np.float32)
    door_right_size = np.array(handles["door_right"]["s_op"].Get(), dtype=np.float32)

    door_left_center_local = (door_left_center_world[:2] - env_origins[0, :2]).astype(np.float32)
    door_right_center_local = (door_right_center_world[:2] - env_origins[0, :2]).astype(np.float32)

    out_dir = os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    map_output_png = os.path.join(out_dir, "door_static_observation_map.png")
    lidar_output_png = os.path.join(out_dir, "door_static_observation_lidar.png")

    save_observation_figures(
        map_output_path=map_output_png,
        lidar_output_path=lidar_output_png,
        env_size=scene_manager.env_size,
        robot_local_xy=robot_local_xy,
        robot_yaw=yaw0,
        goal_local_xy=goal_local_xy,
        lidar_ranges=lidar_ranges[0],
        lidar_max_range=lidar_max_range,
        user_intent_env=user_intent_env_np,
        user_intent_ego=user_intent_ego_np,
        door_left_center_local=door_left_center_local,
        door_left_size=door_left_size[:2],
        door_right_center_local=door_right_center_local,
        door_right_size=door_right_size[:2],
        extra_cyl_center_local=extra_cyl_center_local,
        extra_cyl_radius=extra_cyl_radius,
    )

    print("\n=== Static observation snapshot done ===")
    print(f"Scene: DOOR (single env, randomized door width/offset)")
    print(f"Robot local pose: x={robot_local_xy[0]:.3f}, y={robot_local_xy[1]:.3f}, yaw={yaw0:.3f}")
    print(f"Goal local position: x={goal_local_xy[0]:.3f}, y={goal_local_xy[1]:.3f}")
    print(f"User intent (model input / ego frame X-forward Y-left): {user_intent_ego_np}")
    print(f"User intent (env frame for map draw only): {user_intent_env_np}")
    print(f"LiDAR shape: {lidar_ranges.shape}, min={float(np.min(lidar_ranges)):.3f}, max={float(np.max(lidar_ranges)):.3f}")
    print(f"Observation shape: {obs.shape}")
    print(f"Saved map PNG: {map_output_png}")
    print(f"Saved lidar PNG: {lidar_output_png}")

    print("Scene kept open after plotting. Close Isaac Sim window to exit.")
    while simulation_app.is_running():
        robots.set_joint_velocity_targets(np.zeros((1, robots.num_dof), dtype=np.float32))
        robots.set_velocities(np.zeros((1, 6), dtype=np.float32))
        robots.set_joint_velocities(np.zeros((1, robots.num_dof), dtype=np.float32))
        world.step(render=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
