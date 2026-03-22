"""
SFT + PPO training script (engineering V3).

Design goals:
1) Keep the teammate SFT -> PPO two-stage logic.
2) Use project-standard env setup and observation pipeline (44-dim).
3) Keep BOX-only scene setting.
4) Use modular reward composer (with config-enabled components).

Naming notes (for teammates reading against `sft_rl_tb3.py`):
- `cfg` (old) -> `env_cfg`/`network_cfg`/`algo_cfg`/`reward_cfg` (split configs)
- `ppo` (old) -> `rl_agent` (algorithm instance from factory)
- `reward_details` (old) -> `reward_info` (composer output dict)
- `user_intent_env` (old var name) -> `user_intent_ego` (actual model input meaning)
- `CKPT_SAVE_DIR` + `save_ckpt()` (old) -> `checkpoint_dir` + `save_model()`
- `Actor/Critic/PPO` classes (old inline) -> project modules (`core_network` + `algorithms`)
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


# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=== Device ===")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


# Isaac Sim app
simulation_app = SimulationApp({"headless": False})


from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal

from sim.scenes import SCENE_BOX, yaw_from_quat_wxyz, quat_wxyz_from_yaw, wrap_to_pi
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
    """Load config from common candidate paths."""
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
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = os.path.join(checkpoint_dir, f"ppo_{tag}.pth")
    rl_agent.save(save_path)
    print(f"[Checkpoint] Saved => {save_path}")


def get_network_params_by_name(network_cfg, network_name):
    """Get network params by component name from network config."""
    components = OmegaConf.to_container(network_cfg.networks.components, resolve=True)
    for comp in components:
        if comp.get("name") == network_name:
            return comp.get("params", {}) or {}
    return {}


class RuleBasedController:
    """
    Rule-based controller for SFT data collection.
    Inputs:
      - lidar_ranges (36)
      - user_intent (ego frame: forward, left)
      - base_vel
    """

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

        # 36 rays, 10 degrees each
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
        v_cmd = np.clip(v_cmd, 0, self.max_v)
        w_cmd = np.clip(w_cmd, -self.max_w, self.max_w)
        return v_cmd, w_cmd


def train_supervised_policy(rl_agent, obs_np, act_np, epochs=50, batch_size=2048):
    """Supervised pretraining using current policy mean output."""
    if len(obs_np) == 0:
        print("[SFT] No supervised samples. Skip pretraining.")
        return None

    print(f"[SFT] Start supervised training: samples={len(obs_np)}, epochs={epochs}")
    obs_tensor = torch.FloatTensor(obs_np).to(device)
    act_tensor = torch.FloatTensor(act_np).to(device)
    criterion = nn.MSELoss()
    final_loss = None
    rl_agent.policy.train()

    for epoch in range(epochs):
        indices = np.arange(len(obs_tensor))
        np.random.shuffle(indices)
        epoch_loss = 0.0
        num_batches = 0
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start + batch_size]
            batch_obs = obs_tensor[batch_idx]
            batch_act = act_tensor[batch_idx]
            mean, _ = rl_agent.policy(batch_obs)
            loss = criterion(mean, batch_act)
            rl_agent.actor_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rl_agent.policy.parameters(), max_norm=1.0)
            rl_agent.actor_optimizer.step()
            epoch_loss += float(loss.item())
            num_batches += 1
        final_loss = epoch_loss / max(1, num_batches)
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"[SFT] Epoch {epoch:03d} | loss={final_loss:.6f}")
    return final_loss


def main():
    # Config split (old script used single `cfg`)
    env_cfg = OmegaConf.create(
        {
            "env": {
                "numEnvs": 1,
                "resetDist": 0.2,
                "lidar": {"num_rays": 36, "max_range": 3.0},
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
    network_cfg = load_config("config/network_config.yaml")
    algo_cfg = load_config("config/algorithm_config.yaml")
    reward_cfg = load_config("config/reward_config.yaml")

    # Environment/training params
    num_envs = int(env_cfg.env.numEnvs)
    reset_dist = float(env_cfg.env.resetDist)
    show_visual_walls = bool(env_cfg.env.scene.show_visual_walls)
    TIMEOUT_SECONDS = 120.0
    physics_dt = float(env_cfg.sim.dt)
    max_v = float(MAX_V)
    max_w = float(MAX_W)
    lidar_num_rays = int(env_cfg.env.lidar.num_rays)
    lidar_max_range = float(env_cfg.env.lidar.max_range)
    DCOL = 0.2
    DCRIT = 0.5

    # SFT/PPO schedule
    # Toggle supervised warmup:
    # - True  -> Rule data collection + supervised pretrain + PPO
    # - False -> Direct PPO from step 0
    # Whether to enable SFT warmup before PPO.
    ENABLE_SFT = False
    in_supervised_phase = ENABLE_SFT
    SUPERVISED_STEPS = 200000
    SUPERVISED_EPOCHS = 50

    # Reproducibility
    seed = 0
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # BOX-only scene using project-standard setup
    scene_types = [SCENE_BOX] * num_envs
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
    action_history = np.zeros((num_envs, 4), dtype=np.float32)  # v_t-1,w_t-1,v_t-2,w_t-2
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

    # Network: force SFT architecture while keeping config-managed params
    configured_network_type, configured_network_params = get_enabled_component(network_cfg.networks)
    network_params = get_network_params_by_name(network_cfg, "simple_fc_sft")
    if configured_network_type != "simple_fc_sft":
        print(f"[INFO] network '{configured_network_type}' is enabled in config. Forcing 'simple_fc_sft' for this script.")
    network_type = "simple_fc_sft"
    network_params.update({
        "obs_dim": 44,
        "act_dim": 2,
        "lidar_dim": lidar_num_rays,
        "state_dim": 8,  # UserIntent(2)+BaseVel(2)+ActionHistory(4)
    })
    policy = create_policy(network_type, **network_params)
    value = create_value(network_type, **network_params)

    algorithm_type, algorithm_params = get_enabled_component(algo_cfg.algorithms)
    # `rl_agent` corresponds to old variable name `ppo`
    rl_agent = create_algorithm(
        algorithm_type,
        policy=policy,
        value=value,
        config=algorithm_params,
        device=device,
    )
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
    save_every = 60
    total_epochs = 300
    update_count = 0
    step_count = -1
    current_time = 0.0
    reward_buffer = deque(maxlen=100)
    reward_component_sums = {}
    reward_component_count = 0
    supervised_obs = []
    supervised_actions = []
    total_obstacle_collision = 0

    # old `CKPT_SAVE_DIR` equivalent
    checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints", "ppo_sft_rl_tb3_v3")
    writer = SummaryWriter(log_dir=os.path.join(PROJECT_ROOT, "runs", "ppo_sft_rl_tb3_v3"))

    temp_cur_act = np.zeros((num_envs, 2), dtype=np.float32)
    temp_logprob = np.zeros((num_envs,), dtype=np.float32)

    print("\n=== Start SFT RL V3 ===")
    print(
        f"num_envs={num_envs}, scene=BOX only, use_supervised={ENABLE_SFT}, "
        f"supervised_steps={SUPERVISED_STEPS}"
    )

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
        # user_intent_ego is the model input convention:
        # [ux, uy] where ux=forward(+X of robot body), uy=left(+Y of robot body).
        _, user_intent_ego, _ = compute_user_intent_torch(
            robot_pos_torch,
            torch.tensor(yaw),
            goal_pos_torch,
            env_origins_torch,
            normalize=True,
        )
        user_intent_np = user_intent_ego.cpu().numpy().astype(np.float32)  # fed into obs[36:38]

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
        total_obstacle_collision += int(np.sum(obstacle_collision))

        v_cmd = np.zeros(num_envs, dtype=np.float32)
        w_cmd = np.zeros(num_envs, dtype=np.float32)
        log_probs = np.zeros(num_envs, dtype=np.float32)
        current_act = np.zeros((num_envs, 2), dtype=np.float32)

        if in_supervised_phase and step_count < SUPERVISED_STEPS:
            for i in range(num_envs):
                if obstacle_collision[i]:
                    v = 0.0
                    w = 0.0
                else:
                    v, w = rule_controller.compute_velocity_commands(
                        lidar_ranges=lidar_ranges[i],
                        user_intent=user_intent_np[i],
                        base_vel=base_vel_np[i],
                    )
                v_cmd[i] = v
                w_cmd[i] = w
                current_act[i, 0] = v / max_v
                current_act[i, 1] = w / max_w
                log_probs[i] = 0.0
                if not obstacle_collision[i]:
                    supervised_obs.append(obs[i])
                    supervised_actions.append(current_act[i])
        else:
            if in_supervised_phase:
                final_loss = train_supervised_policy(
                    rl_agent,
                    np.array(supervised_obs, dtype=np.float32),
                    np.array(supervised_actions, dtype=np.float32),
                    epochs=SUPERVISED_EPOCHS,
                )
                save_model(rl_agent, checkpoint_dir, "supervised_final")
                print(f"[SFT] done, final loss = {final_loss}")
                in_supervised_phase = False
                print("=== Switched to PPO finetuning ===")

            for i in range(num_envs):
                if obstacle_collision[i]:
                    act = np.array([0.0, 0.0], dtype=np.float32)
                    log_prob = 0.0
                else:
                    if step_count % 1 == 0:
                        act, log_prob = rl_agent.select_action(obs[i])
                        temp_cur_act[i] = act
                        temp_logprob[i] = log_prob
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

        # `reward_info` corresponds to old `reward_details` dictionary.
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

        if not in_supervised_phase:
            for i in range(num_envs):
                reward_buffer.append(rewards[i])
                rl_agent.store_transition(
                    obs=obs[i],
                    act=current_act[i],
                    rew=rewards[i],
                    next_obs=next_obs[i],
                    done=to_reset[i],
                    log_prob=log_probs[i],
                )

            if step_count % update_freq == 0 and len(rl_agent.buffer["obs"]) >= rl_agent.batch_size:
                    train_info = rl_agent.update()
                    update_count += 1
                    avg_reward = np.mean(reward_buffer) if reward_buffer else 0.0
                    writer.add_scalar("train/actor_loss", train_info["actor_loss"], update_count)
                    writer.add_scalar("train/critic_loss", train_info["critic_loss"], update_count)
                    writer.add_scalar("train/entropy", train_info["entropy"], update_count)
                    writer.add_scalar("train/approx_kl", train_info["approx_kl"], update_count)
                    writer.add_scalar("reward/total_mean", avg_reward, update_count)
                    if reward_component_count > 0:
                        for k, v_sum in reward_component_sums.items():
                            writer.add_scalar(f"reward/{k}", v_sum / reward_component_count, update_count)
                    for k in reward_component_sums.keys():
                        reward_component_sums[k] = 0.0
                    reward_component_count = 0

                    if update_count % save_every == 0:
                        save_model(rl_agent, checkpoint_dir, f"epoch_{update_count:03d}")
                    if update_count >= total_epochs:
                        from datetime import datetime
                        save_model(rl_agent, checkpoint_dir, f"final_{datetime.now().strftime('%Y%m%d_%H')}")
                        break

        avg_reward = np.mean(reward_buffer) if reward_buffer else 0.0
        if in_supervised_phase:
            print(f"\n=== 监督学习数据收集 ===")
            print(
                f"当前阶段: SUPERVISED | step={step_count}/{SUPERVISED_STEPS} | "
                f"已收集样本: {len(supervised_obs)}"
            )
        else:
            print(f"\n=== PPO策略更新 ===")
            print(
                f"当前阶段: PPO | step={step_count} | "
                f"update={update_count}/{total_epochs} | 最近100步平均奖励: {avg_reward:.2f}"
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
        # Unified reward logging: print directly from reward_info (no manual recomputation).
        idx = 0
        reward_name_map = {
            "distance_reward": "距离奖励",
            "goal_reward": "目标奖励",
            "static_pen": "静止惩罚",
            "obstacle_pen": "障碍物惩罚",
            "collision_pen": "碰撞惩罚",
            "heading_pen": "航向惩罚",
            "smooth_pen": "动作平滑惩罚",
        }
        preferred_order = [
            "distance_reward",
            "goal_reward",
            "static_pen",
            "obstacle_pen",
            "collision_pen",
            "heading_pen",
            "smooth_pen",
        ]

        def _reward_scalar(v):
            arr = np.asarray(v)
            if arr.size == 0:
                return 0.0
            flat = arr.reshape(-1)
            pick = min(idx, flat.shape[0] - 1)
            return float(flat[pick])

        printed = set()
        for key in preferred_order:
            if key in reward_info:
                zh_name = reward_name_map.get(key, key)
                print(f"{zh_name} ({key}): {_reward_scalar(reward_info[key]):.4f}")
                printed.add(key)

        for key, value in reward_info.items():
            if key in printed:
                continue
            zh_name = reward_name_map.get(key, key)
            print(f"{zh_name} ({key}): {_reward_scalar(value):.4f}")

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

        time.sleep(0.001)

    simulation_app.close()
    writer.close()


if __name__ == "__main__":
    main()
