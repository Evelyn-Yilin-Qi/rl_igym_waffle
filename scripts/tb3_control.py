"""
TurtleBot3 local navigation in Isaac Sim using RuleBasedController (sector rules + intent).
LiDAR ray count must match controller num_rays (default 360 here).
"""
import os
import sys
import time
from collections import deque

import numpy as np
import torch
from omegaconf import OmegaConf
from omni.isaac.kit import SimulationApp

# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=== Device (for intent torch ops) ===")
print(f"Using device: {device}")

simulation_app = SimulationApp({"headless": False})

from sim.scenes import (
    SCENE_EMPTY,
    SCENE_CYLINDER,
    SCENE_DOOR,
    SCENE_BOX,
    yaw_from_quat_wxyz,
    quat_wxyz_from_yaw,
    wrap_to_pi,
)
from sim.robot import MAX_V, MAX_W
from envs.observations import (
    get_base_velocity_from_tensor,
    compute_lidar_ranges,
    check_obstacle_collision,
)
from envs.user_intent import compute_user_intent_torch

from process_settings.env_setup import EnvironmentSetup
from rewards import RewardComposer
from sft import load_config, print_phase_status, print_reward_breakdown, RuleBasedController


def main():
    # 单环境测试：SCENE_EMPTY / SCENE_CYLINDER / SCENE_DOOR / SCENE_BOX
    test_scene_type = SCENE_BOX
    num_envs = 1
    scene_types = [test_scene_type]

    env_cfg = OmegaConf.create(
        {
            "env": {
                "numEnvs": num_envs,
                "resetDist": 0.8,
                "lidar": {"num_rays": 360, "max_range": 3.0},
                "scene": {
                    "env_size": 6.0,
                    "wall_thickness": 0.08,
                    "wall_height": 1.2,
                    "show_visual_walls": False,
                },
            },
            "sim": {"dt": 0.025},
        }
    )
    reward_cfg = load_config(PROJECT_ROOT, "config/reward_config.yaml")

    num_envs = int(env_cfg.env.numEnvs)
    reset_dist = float(env_cfg.env.resetDist)
    show_visual_walls = None
    TIMEOUT_SECONDS = 600.0
    physics_dt = float(env_cfg.sim.dt)
    max_v = float(MAX_V)
    max_w = float(MAX_W)
    lidar_num_rays = int(env_cfg.env.lidar.num_rays)
    lidar_max_range = float(env_cfg.env.lidar.max_range)
    DCOL = 0.2
    DCRIT = 0.5

    seed = 0
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    env_setup = EnvironmentSetup(
        env_cfg=env_cfg,
        num_envs=num_envs,
        scene_types=scene_types,
        simulation_app=simulation_app,
        rng=rng,
        show_visual_walls=show_visual_walls,
        reset_scene_obstacles=True,
    )
    setup = env_setup.setup_all()
    world = setup["world"]
    robots = setup["robots"]
    scene_manager = setup["scene_manager"]
    diff_ctrl = setup["diff_ctrl"]
    left_idx = setup["left_idx"]
    right_idx = setup["right_idx"]
    env_origins = setup["env_origins"]
    goal_pos = setup["goal_pos"]
    markers = setup["markers"]

    episode_start_time = np.zeros((num_envs,), dtype=np.float32)
    action_history = np.zeros((num_envs, 4), dtype=np.float32)
    prev_act = np.zeros((num_envs, 2), dtype=np.float32)
    prev_prev_act = np.zeros((num_envs, 2), dtype=np.float32)

    for i in range(num_envs):
        spawn_pos, spawn_yaw = scene_manager.get_robot_spawn_config(i, rng=rng)
        spawn_rot = quat_wxyz_from_yaw(np.array([spawn_yaw], dtype=np.float32))
        idx = np.array([i], dtype=np.int32)
        robots.set_world_poses(positions=spawn_pos.reshape(1, 3), orientations=spawn_rot.reshape(1, 4), indices=idx)
        robots.set_velocities(np.zeros((1, 6), dtype=np.float32), indices=idx)
        robots.set_joint_velocities(np.zeros((1, robots.num_dof), dtype=np.float32), indices=idx)
        robots.set_joint_velocity_targets(np.zeros((1, robots.num_dof), dtype=np.float32), indices=idx)
        episode_start_time[i] = 0.0

    reward_composer = RewardComposer(reward_cfg.rewards)
    rule_controller = RuleBasedController(
        max_v=max_v,
        max_w=max_w,
        safe_dist=1.0,
        collision_dist=DCOL,
        v_gain=0.4,
        w_gain=1.8,
        obstacle_v_gain=0.3,
        obstacle_w_gain=2.2,
        num_rays=lidar_num_rays,
    )

    for _ in range(8):
        robots.set_joint_velocity_targets(np.zeros((num_envs, robots.num_dof), dtype=np.float32))
        robots.set_velocities(np.zeros((num_envs, 6), dtype=np.float32))
        robots.set_joint_velocities(np.zeros((num_envs, robots.num_dof), dtype=np.float32))
        world.step(render=True)

    step_count = -1
    current_time = 0.0
    reward_buffer = deque(maxlen=100)
    reward_component_sums = {}
    reward_component_count = 0
    total_obstacle_collision = 0

    print("\n=== RuleBasedController (360 LiDAR sectors) ===")
    print(f"num_envs={num_envs}, scene={test_scene_type}, lidar_num_rays={lidar_num_rays}")

    while simulation_app.is_running():
        current_time += physics_dt
        step_count += 1

        pos, rot = robots.get_world_poses()
        yaw = yaw_from_quat_wxyz(rot)
        x_local = (pos[:, 0] - env_origins[:, 0]).astype(np.float32)
        y_local = (pos[:, 1] - env_origins[:, 1]).astype(np.float32)
        car_local = np.stack([x_local, y_local], axis=1)
        goal_local = goal_pos[:, :2] - env_origins[:, :2]
        gx_local = goal_local[:, 0].astype(np.float32)
        gy_local = goal_local[:, 1].astype(np.float32)
        ex = gx_local - x_local
        ey = gy_local - y_local
        goal_heading = np.arctan2(ey, ex)

        lidar_ranges = compute_lidar_ranges(
            robot_positions=pos,
            robot_orientations=rot,
            env_origins=env_origins,
            lidar_num_rays=lidar_num_rays,
            lidar_max_range=lidar_max_range,
            scene_manager=scene_manager,
        )
        robot_velocities_np = robots.get_velocities()
        robot_velocities_torch = torch.from_numpy(robot_velocities_np).float().to(device)
        base_vel_np = get_base_velocity_from_tensor(robot_velocities_torch, max_v=max_v, max_w=max_w)
        current_measured_v = robot_velocities_np[:, 0]
        current_measured_w = robot_velocities_np[:, 5]

        robot_pos_torch = torch.from_numpy(pos).float().to(device)
        goal_pos_torch = torch.from_numpy(goal_pos).float().to(device)
        env_origins_torch = torch.from_numpy(env_origins).float().to(device)
        _, user_intent_ego, _ = compute_user_intent_torch(
            robot_pos_torch,
            torch.tensor(yaw),
            goal_pos_torch,
            env_origins_torch,
            normalize=True,
        )
        user_intent_np = user_intent_ego.cpu().numpy().astype(np.float32)

        obstacle_collision, _, min_dist = check_obstacle_collision(lidar_ranges, DCOL, DCRIT)
        total_obstacle_collision += int(np.sum(obstacle_collision))

        v_cmd = np.zeros(num_envs, dtype=np.float32)
        w_cmd = np.zeros(num_envs, dtype=np.float32)
        current_act = np.zeros((num_envs, 2), dtype=np.float32)

        for i in range(num_envs):
            if obstacle_collision[i]:
                v_cmd[i] = 0.0
                w_cmd[i] = 0.0
            else:
                v, w = rule_controller.compute_velocity_commands(
                    lidar_ranges=lidar_ranges[i],
                    user_intent=user_intent_np[i],
                    base_vel=base_vel_np[i],
                )
                v_cmd[i] = v
                w_cmd[i] = w
            current_act[i, 0] = v_cmd[i] / max_v
            current_act[i, 1] = w_cmd[i] / max_w

        targets = np.zeros((num_envs, robots.num_dof), dtype=np.float32)
        for i in range(num_envs):
            action = diff_ctrl.forward(command=np.array([float(v_cmd[i]), float(w_cmd[i])], dtype=np.float32))
            targets[i, left_idx] = float(action.joint_velocities[0])
            targets[i, right_idx] = float(action.joint_velocities[1])

        action_history[:, 2:] = action_history[:, :2]
        action_history[:, 0] = v_cmd
        action_history[:, 1] = w_cmd

        robots.set_joint_velocity_targets(targets)
        world.step(render=True)

        dist = np.sqrt(ex * ex + ey * ey).astype(np.float32)
        yaw_err = wrap_to_pi(goal_heading - yaw).astype(np.float32)
        reset_timeout = (current_time - episode_start_time) >= TIMEOUT_SECONDS
        reached = dist <= reset_dist

        boundary_collision = np.zeros((num_envs,), dtype=bool)
        for i in range(num_envs):
            boundary_collision[i] = scene_manager.check_boundary_collision(i, np.array([x_local[i], y_local[i]]))
        to_reset = obstacle_collision | boundary_collision | reset_timeout | reached

        rewards, reward_info = reward_composer.compute(
            dist=dist,
            yaw_err=yaw_err,
            v_cmd=v_cmd,
            w_cmd=w_cmd,
            v_measured=current_measured_v,
            w_measured=current_measured_w,
            collision=obstacle_collision,
            reached=reached,
            timeout=reset_timeout,
            min_dist=min_dist,
            current_act=current_act,
            prev_act=prev_act,
            prev_prev_act=prev_prev_act,
        )

        reward_component_count += len(rewards)
        for k, v in reward_info.items():
            if k not in reward_component_sums:
                reward_component_sums[k] = 0.0
            reward_component_sums[k] += float(np.sum(v))

        avg_reward = np.mean(reward_buffer) if reward_buffer else 0.0
        for i in range(num_envs):
            reward_buffer.append(rewards[i])

        print_phase_status(
            train_mode="rule_based",
            step_count=step_count,
            supervised_steps=0,
            supervised_obs_count=0,
            update_count=0,
            total_epochs=0,
            avg_reward=avg_reward,
        )
        print(f"当前奖励: {rewards[0]:.2f}")
        print(f"yaw:{yaw}")
        print(f"goal local:{goal_local}")
        print(f"robot local:{car_local}")
        print(f"角速度：{w_cmd}")
        print(f"线速度：{v_cmd}")
        print(f"目标距离：{dist}")
        print(f"用户意图：{user_intent_np}")
        print(f"角度差：{yaw_err}")
        print(f"总碰撞次数：{total_obstacle_collision}")
        print_reward_breakdown(reward_info, env_idx=0)

        if np.any(to_reset):
            reset_ids = np.nonzero(to_reset)[0]
            current_base_vels = robots.get_velocities()
            current_base_vels[reset_ids] = 0.0
            robots.set_velocities(current_base_vels)
            current_joint_vels = robots.get_joint_velocities()
            current_joint_vels[reset_ids] = 0.0
            robots.set_joint_velocities(current_joint_vels)

            for i in reset_ids:
                scene_manager.reset_scene_obstacles(np.array([i]), rng=rng)
                spawn_pos, spawn_yaw = scene_manager.get_robot_spawn_config(i, rng=rng)
                spawn_rot = quat_wxyz_from_yaw(np.array([spawn_yaw], dtype=np.float32))
                idx = np.array([i], dtype=np.int32)
                robots.set_world_poses(positions=spawn_pos.reshape(1, 3), orientations=spawn_rot.reshape(1, 4), indices=idx)
                robots.set_velocities(np.zeros((1, 6), dtype=np.float32), indices=idx)
                robots.set_joint_velocities(np.zeros((1, robots.num_dof), dtype=np.float32), indices=idx)
                robots.set_joint_velocity_targets(np.zeros((1, robots.num_dof), dtype=np.float32), indices=idx)
                goal_pos[i] = scene_manager.get_goal_config(i, rng=rng)
                markers[i].set_world_pose(position=goal_pos[i].tolist(), orientation=[1, 0, 0, 0])
                episode_start_time[i] = current_time
                action_history[i] = np.zeros(4, dtype=np.float32)
                prev_act[i] = np.zeros(2)
                prev_prev_act[i] = np.zeros(2)

            for _ in range(8):
                robots.set_joint_velocity_targets(np.zeros((num_envs, robots.num_dof), dtype=np.float32))
                robots.set_velocities(np.zeros((num_envs, 6), dtype=np.float32))
                robots.set_joint_velocities(np.zeros((num_envs, robots.num_dof), dtype=np.float32))
                world.step(render=True)

        prev_prev_act = prev_act.copy()
        prev_act = current_act.copy()

        time.sleep(0.001)

    simulation_app.close()


if __name__ == "__main__":
    main()
