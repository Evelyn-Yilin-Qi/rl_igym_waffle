"""
Stage1 SFT + RL training script.

- Scene: SCENE_EMPTY only
- Observation: 44 dims (LiDAR36 + intent2 + base_vel2 + action_history4)
- Training strategy: supervised warmup (rule policy) -> PPO finetune
- Network: force simple_fc_sft
- Reward: use project reward_config.yaml via RewardComposer
"""
import os
import sys
import time
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from omni.isaac.kit import SimulationApp
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] {device}")
if torch.cuda.is_available():
    print(f"[GPU] {torch.cuda.get_device_name(0)}")


simulation_app = SimulationApp({"headless": False})


from sim.scenes import SCENE_EMPTY, yaw_from_quat_wxyz, quat_wxyz_from_yaw, wrap_to_pi
from sim.robot import MAX_V, MAX_W
from envs.observations import (
    assemble_observations,
    get_base_velocity_from_tensor,
    compute_lidar_ranges,
    check_obstacle_collision,
    get_obs,
)
from envs.user_intent import compute_user_intent_torch
from process_settings.env_setup import EnvironmentSetup
from core_network import create_policy, create_value
from algorithms import create_algorithm
from rewards import RewardComposer
from utils.config_utils import get_enabled_component


def load_config(cfg_path):
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


def save_model(rl_agent, checkpoint_dir, tag):
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = os.path.join(checkpoint_dir, f"ppo_{tag}.pth")
    rl_agent.save(save_path)
    print(f"[Checkpoint] Saved => {save_path}")


def get_network_params_by_name(network_cfg, network_name):
    components = OmegaConf.to_container(network_cfg.networks.components, resolve=True)
    for comp in components:
        if comp.get("name") == network_name:
            return comp.get("params", {}) or {}
    return {}


class RuleBasedController:
    """Rule controller for SFT dataset collection."""

    def __init__(
        self,
        max_v=0.5,
        max_w=1.0,
        safe_dist=1.0,
        collision_dist=0.2,
        v_gain=1.0,
        w_gain=1.5,
        obstacle_v_gain=1.0,
        obstacle_w_gain=2.0,
    ):
        self.max_v = max_v
        self.max_w = max_w
        self.safe_dist = safe_dist
        self.collision_dist = collision_dist
        self.v_gain = v_gain
        self.w_gain = w_gain
        self.obstacle_v_gain = obstacle_v_gain
        self.obstacle_w_gain = obstacle_w_gain
        self.front_sectors = [0]
        self.left_sectors = list(range(1, 9))
        self.right_sectors = list(range(28, 36))
        self.back_sectors = list(range(9, 28))

    def get_obstacle_direction(self, lidar_ranges):
        min_dist = np.min(lidar_ranges)
        min_idx = int(np.argmin(lidar_ranges))
        if min_idx in self.front_sectors and min_dist < self.safe_dist:
            return "front", min_dist
        if min_idx in self.left_sectors and min_dist < self.safe_dist:
            return "left", min_dist
        if min_idx in self.right_sectors and min_dist < self.safe_dist:
            return "right", min_dist
        if min_dist < self.safe_dist:
            return "back", min_dist
        return None, min_dist

    def compute_velocity_commands(self, lidar_ranges, user_intent, base_vel):
        obstacle_side, min_dist = self.get_obstacle_direction(lidar_ranges)
        if np.linalg.norm(user_intent) < 1e-6:
            target_angle = 0.0
        else:
            target_angle = np.arctan2(user_intent[1], user_intent[0])

        if min_dist < self.collision_dist:
            v_cmd = 0.0
            if obstacle_side == "left":
                w_cmd = -self.max_w * 0.8
            elif obstacle_side == "right":
                w_cmd = self.max_w * 0.8
            elif obstacle_side == "front":
                w_cmd = self.max_w * np.sign(np.random.uniform(-1, 1))
            else:
                w_cmd = 0.0
        elif obstacle_side is not None:
            dist_ratio = (min_dist - self.collision_dist) / (self.safe_dist - self.collision_dist)
            if obstacle_side == "left":
                w_cmd = np.clip(self.obstacle_w_gain * (-2.0 + target_angle / 2), -self.max_w, self.max_w)
                v_cmd = self.obstacle_v_gain * self.max_v * dist_ratio + (1 - abs(w_cmd)) + random.uniform(0.1, 0.2)
            elif obstacle_side == "right":
                w_cmd = np.clip(self.obstacle_w_gain * (2.0 + target_angle / 2), -self.max_w, self.max_w)
                v_cmd = self.obstacle_v_gain * self.max_v * dist_ratio + (1 - abs(w_cmd)) + random.uniform(0.1, 0.2)
            elif obstacle_side == "front":
                left = lidar_ranges[1]
                right = lidar_ranges[-1]
                w_cmd = 1.0 if left > right else -1.0
                v_cmd = self.obstacle_v_gain * self.max_v * dist_ratio + (1 - abs(w_cmd)) + random.uniform(0.1, 0.2)
            else:
                w_cmd = np.clip(self.w_gain * target_angle, -self.max_w, self.max_w)
                v_cmd = self.obstacle_v_gain * self.max_v * dist_ratio + (1 - abs(w_cmd)) + random.uniform(0.1, 0.2)
        else:
            intent_magnitude = np.linalg.norm(user_intent)
            v_cmd = np.clip(self.v_gain * intent_magnitude, 0, self.max_v)
            w_cmd = np.clip(self.w_gain * target_angle, -self.max_w, self.max_w)

        v_cmd = 0.8 * v_cmd + 0.2 * base_vel[0]
        w_cmd = 0.8 * w_cmd + 0.2 * base_vel[1]
        return np.clip(v_cmd, 0, self.max_v), np.clip(w_cmd, -self.max_w, self.max_w)


def train_supervised_policy(rl_agent, obs_np, act_np, epochs=50, batch_size=2048):
    if len(obs_np) == 0:
        print("[SFT] No samples collected.")
        return None
    obs_tensor = torch.FloatTensor(obs_np).to(device)
    act_tensor = torch.FloatTensor(act_np).to(device)
    criterion = nn.MSELoss()
    rl_agent.policy.train()
    final_loss = None
    print(f"[SFT] samples={len(obs_np)}, epochs={epochs}")
    for epoch in range(epochs):
        indices = np.arange(len(obs_tensor))
        np.random.shuffle(indices)
        epoch_loss = 0.0
        num_batches = 0
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start + batch_size]
            mean, _ = rl_agent.policy(obs_tensor[batch_idx])
            loss = criterion(mean, act_tensor[batch_idx])
            rl_agent.actor_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rl_agent.policy.parameters(), max_norm=1.0)
            rl_agent.actor_optimizer.step()
            epoch_loss += float(loss.item())
            num_batches += 1
        final_loss = epoch_loss / max(1, num_batches)
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"[SFT] epoch={epoch:03d}, loss={final_loss:.6f}")
    return final_loss


def main():
    network_cfg = load_config("config/network_config.yaml")
    algo_cfg = load_config("config/algorithm_config.yaml")
    reward_cfg = load_config("config/reward_config.yaml")

    num_envs = 16
    reset_dist = 0.2
    TIMEOUT_SECONDS = 60.0
    physics_dt = 0.025
    max_v = float(MAX_V)
    max_w = float(MAX_W)
    lidar_num_rays = 36
    lidar_max_range = 3.0
    DCOL = 0.2
    DCRIT = 0.5

    SUPERVISED_EPOCHS = 50
    SUPERVISED_STEPS = 200000
    TRAIN_MODE = "supervised"

    seed = 0
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    env_cfg = OmegaConf.create({"sim": {"dt": physics_dt}})
    scene_types = [SCENE_EMPTY] * num_envs
    env_setup = EnvironmentSetup(
        env_cfg=env_cfg,
        num_envs=num_envs,
        scene_types=scene_types,
        simulation_app=simulation_app,
        rng=rng,
        show_visual_walls=None,  # let SceneManager decide visibility by scene type
        reset_scene_obstacles=False,
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
    prev_measured_v = np.zeros((num_envs,), dtype=np.float32)
    prev_measured_w = np.zeros((num_envs,), dtype=np.float32)
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

    configured_network_type, _ = get_enabled_component(network_cfg.networks)
    network_params = get_network_params_by_name(network_cfg, "simple_fc_sft")
    if configured_network_type != "simple_fc_sft":
        print(f"[INFO] forcing simple_fc_sft (config enabled={configured_network_type})")
    network_params.update({
        "obs_dim": 44,
        "act_dim": 2,
        "lidar_dim": lidar_num_rays,
        "state_dim": 8,
    })
    policy = create_policy("simple_fc_sft", **network_params)
    value = create_value("simple_fc_sft", **network_params)

    algorithm_type, algorithm_params = get_enabled_component(algo_cfg.algorithms)
    rl_agent = create_algorithm(algorithm_type, policy=policy, value=value, config=algorithm_params, device=device)
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
    )

    for _ in range(8):
        robots.set_joint_velocity_targets(np.zeros((num_envs, robots.num_dof), dtype=np.float32))
        robots.set_velocities(np.zeros((num_envs, 6), dtype=np.float32))
        robots.set_joint_velocities(np.zeros((num_envs, robots.num_dof), dtype=np.float32))
        world.step(render=True)

    update_freq = 2048
    save_every = 10
    total_epochs = 50
    progress_print_every = 1000
    update_count = 0
    step_count = -1
    current_time = 0.0
    reward_buffer = deque(maxlen=100)
    reward_component_sums = {}
    reward_component_count = 0
    supervised_obs = []
    supervised_actions = []
    checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints", "ppo_stage1_sft_rl")
    writer = SummaryWriter(log_dir=os.path.join(PROJECT_ROOT, "runs", "ppo_stage1_sft_rl"))
    temp_cur_act = np.zeros((num_envs, 2), dtype=np.float32)

    print(f"[Stage1-SFT-RL] start, supervised_steps={SUPERVISED_STEPS}, supervised_epochs={SUPERVISED_EPOCHS}")
    while simulation_app.is_running():
        current_time += physics_dt
        step_count += 1

        pos, rot = robots.get_world_poses()
        yaw = yaw_from_quat_wxyz(rot)
        x_local = (pos[:, 0] - env_origins[:, 0]).astype(np.float32)
        y_local = (pos[:, 1] - env_origins[:, 1]).astype(np.float32)
        goal_local = goal_pos[:, :2] - env_origins[:, :2]
        gx_local = goal_local[:, 0].astype(np.float32)
        gy_local = goal_local[:, 1].astype(np.float32)
        ex = gx_local - x_local
        ey = gy_local - y_local
        goal_heading = np.arctan2(ey, ex)

        lidar_ranges = compute_lidar_ranges(pos, rot, env_origins, lidar_num_rays, lidar_max_range, scene_manager)
        robot_velocities_np = robots.get_velocities()
        robot_velocities_torch = torch.from_numpy(robot_velocities_np).float().to(device)
        base_vel_np = get_base_velocity_from_tensor(robot_velocities_torch, max_v=max_v, max_w=max_w)
        current_measured_v = robot_velocities_np[:, 0]
        current_measured_w = robot_velocities_np[:, 5]

        robot_pos_torch = torch.from_numpy(pos).float().to(device)
        goal_pos_torch = torch.from_numpy(goal_pos).float().to(device)
        env_origins_torch = torch.from_numpy(env_origins).float().to(device)
        _, user_intent_ego, _ = compute_user_intent_torch(
            robot_pos_torch, torch.tensor(yaw), goal_pos_torch, env_origins_torch, normalize=True
        )
        user_intent_np = user_intent_ego.cpu().numpy().astype(np.float32)

        obs = assemble_observations(
            robot_positions=pos,
            robot_orientations=rot,
            goal_positions=goal_pos,
            env_origins=env_origins,
            lidar_ranges=lidar_ranges,
            base_vel=base_vel_np,
            action_history=action_history,
            max_v=max_v,
            max_w=max_w,
            lidar_max_range=lidar_max_range,
            user_input=user_intent_np,
        )

        obstacle_collision, critical_flag, min_dist = check_obstacle_collision(lidar_ranges, DCOL, DCRIT)

        v_cmd = np.zeros(num_envs, dtype=np.float32)
        w_cmd = np.zeros(num_envs, dtype=np.float32)
        log_probs = np.zeros(num_envs, dtype=np.float32)
        current_act = np.zeros((num_envs, 2), dtype=np.float32)

        if TRAIN_MODE == "supervised" and step_count < SUPERVISED_STEPS:
            for i in range(num_envs):
                if obstacle_collision[i]:
                    v, w = 0.0, 0.0
                else:
                    v, w = rule_controller.compute_velocity_commands(lidar_ranges[i], user_intent_np[i], base_vel_np[i])
                v_cmd[i], w_cmd[i] = v, w
                current_act[i, 0] = v / max_v
                current_act[i, 1] = w / max_w
                if not obstacle_collision[i]:
                    supervised_obs.append(obs[i])
                    supervised_actions.append(current_act[i])
        else:
            if TRAIN_MODE == "supervised":
                final_loss = train_supervised_policy(
                    rl_agent,
                    np.array(supervised_obs, dtype=np.float32),
                    np.array(supervised_actions, dtype=np.float32),
                    epochs=SUPERVISED_EPOCHS,
                )
                save_model(rl_agent, checkpoint_dir, "supervised_final")
                print(f"[SFT] final_loss={final_loss}")
                TRAIN_MODE = "ppo"

            for i in range(num_envs):
                if obstacle_collision[i]:
                    act = np.array([0.0, 0.0], dtype=np.float32)
                    log_prob = 0.0
                else:
                    if step_count % 1 == 0:
                        act, log_prob = rl_agent.select_action(obs[i])
                        temp_cur_act[i] = act
                    else:
                        act = temp_cur_act[i]
                        obs_cur = torch.FloatTensor(obs[i]).unsqueeze(0).to(device)
                        mean, log_std = rl_agent.policy(obs_cur)
                        std = log_std.exp()
                        dist_t = Normal(mean, std)
                        log_prob = dist_t.log_prob(torch.tensor(act).unsqueeze(0).to(device)).sum().detach().cpu().numpy()

                log_probs[i] = log_prob
                current_act[i] = act
                v_cmd[i] = np.clip(act[0] * max_v, 0.0, max_v)
                w_cmd[i] = np.clip(act[1] * max_w, -max_w, max_w)

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

        next_obs = get_obs(
            robots, env_origins, goal_pos, lidar_num_rays, lidar_max_range, scene_manager, max_v, max_w, action_history, device=device
        )
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
            reward_component_sums[k] = reward_component_sums.get(k, 0.0) + float(np.sum(v))

        if TRAIN_MODE == "ppo":
            for i in range(num_envs):
                reward_buffer.append(rewards[i])
                rl_agent.store_transition(obs[i], current_act[i], rewards[i], next_obs[i], to_reset[i], log_probs[i])

            if step_count % update_freq == 0 and len(rl_agent.buffer["obs"]) >= rl_agent.batch_size:
                train_info = rl_agent.update()
                update_count += 1
                avg_reward = np.mean(reward_buffer) if reward_buffer else 0.0
                print(f"[PPO] step={step_count}, update={update_count}/{total_epochs}, avg_reward={avg_reward:.3f}")
                writer.add_scalar("train/actor_loss", train_info["actor_loss"], update_count)
                writer.add_scalar("train/critic_loss", train_info["critic_loss"], update_count)
                writer.add_scalar("train/entropy", train_info["entropy"], update_count)
                writer.add_scalar("train/approx_kl", train_info["approx_kl"], update_count)
                writer.add_scalar("reward/total_mean", avg_reward, update_count)
                if reward_component_count > 0:
                    for k, v_sum in reward_component_sums.items():
                        writer.add_scalar(f"reward/{k}", v_sum / reward_component_count, update_count)
                reward_component_count = 0
                for k in reward_component_sums:
                    reward_component_sums[k] = 0.0

                if update_count % save_every == 0:
                    save_model(rl_agent, checkpoint_dir, f"epoch_{update_count:03d}")
                if update_count >= total_epochs:
                    from datetime import datetime
                    save_model(rl_agent, checkpoint_dir, f"final_{datetime.now().strftime('%Y%m%d_%H')}")
                    break

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
                prev_measured_v[i] = 0.0
                prev_measured_w[i] = 0.0
                prev_act[i] = np.zeros(2)
                prev_prev_act[i] = np.zeros(2)

            for _ in range(8):
                robots.set_joint_velocity_targets(np.zeros((num_envs, robots.num_dof), dtype=np.float32))
                robots.set_velocities(np.zeros((num_envs, 6), dtype=np.float32))
                robots.set_joint_velocities(np.zeros((num_envs, robots.num_dof), dtype=np.float32))
                world.step(render=True)

        prev_measured_v = current_measured_v.copy()
        prev_measured_w = current_measured_w.copy()
        prev_prev_act = prev_act.copy()
        prev_act = current_act.copy()

        if step_count % progress_print_every == 0:
            avg_reward = np.mean(reward_buffer) if reward_buffer else 0.0
            print(
                f"[Progress] step={step_count}, mode={TRAIN_MODE}, "
                f"updates={update_count}/{total_epochs}, avg_reward={avg_reward:.3f}, "
                f"supervised_samples={len(supervised_obs)}"
            )
        time.sleep(0.001)

    simulation_app.close()
    writer.close()


if __name__ == "__main__":
    main()
