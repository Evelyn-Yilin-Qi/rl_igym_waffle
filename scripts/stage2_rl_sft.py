"""
Stage2 RL + Optional SFT

Design constraints:
- Do NOT modify scripts/stage2_rl.py.
- Keep stage2 scene loading logic unchanged: 16 envs with mixed scenes.
- Add optional SFT phase for multi-env data collection + supervised pretrain.
- Keep modular pipeline: core_network / algorithms / rewards / env_setup.
- Continue training from latest stage1_rl_sft checkpoint by default.
"""
import os
import sys
import time
import glob
import random
from datetime import datetime
import numpy as np
import torch
from torch.distributions import Normal
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
from omni.isaac.kit import SimulationApp

# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=== 设备配置 ===")
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU可用内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("未检测到GPU，使用CPU运行")

# Isaac Sim
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
    assemble_observations,
    get_base_velocity_from_tensor,
    compute_lidar_ranges,
    check_obstacle_collision,
    get_obs,
)
from envs.user_intent import compute_user_intent_torch
from core_network import create_policy, create_value
from algorithms import create_algorithm
from rewards import RewardComposer
from process_settings.env_setup import EnvironmentSetup
from utils.config_utils import get_enabled_component
from sft.rule_controller import RuleBasedController
from sft.supervised import train_supervised_policy


def load_config(cfg_path):
    candidates = []
    if os.path.isabs(cfg_path):
        candidates.append(cfg_path)
    else:
        candidates.extend(
            [
                os.path.join(PROJECT_ROOT, cfg_path),
                os.path.abspath(cfg_path),
                os.path.join(os.getcwd(), cfg_path),
            ]
        )

    found = None
    for path in candidates:
        if os.path.exists(path):
            found = os.path.abspath(path)
            break
    if found is None:
        raise FileNotFoundError(f"Config file not found. Tried: {candidates}")
    return OmegaConf.load(found)


def find_latest_checkpoint(checkpoint_dir, pattern="ppo_*.pth"):
    if not os.path.exists(checkpoint_dir):
        return None
    files = glob.glob(os.path.join(checkpoint_dir, pattern))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def load_checkpoint(rl_agent, checkpoint_path=None, checkpoint_dir=None):
    if checkpoint_path is None and checkpoint_dir is not None:
        checkpoint_path = find_latest_checkpoint(checkpoint_dir)
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        print(f"[INFO] checkpoint not found: {checkpoint_path}")
        print("[INFO] train from scratch")
        return False
    try:
        rl_agent.load(checkpoint_path)
        print(f"[INFO] loaded checkpoint: {checkpoint_path}")
        return True
    except Exception as e:
        print(f"[WARN] load checkpoint failed: {e}")
        print("[INFO] train from scratch")
        return False


def save_model(rl_agent, checkpoint_dir, tag):
    os.makedirs(checkpoint_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    save_path = os.path.join(checkpoint_dir, f"ppo_{tag}_{timestamp}.pth")
    rl_agent.save(save_path)
    print(f"[Checkpoint] Saved => {save_path}")


def main(checkpoint_path=None):
    # ============================================================
    # User Settings (all tunables in one place)
    # ============================================================
    # Config files
    network_cfg_path = "config/network_config.yaml"
    algo_cfg_path = "config/algorithm_config.yaml"
    reward_cfg_path = "config/reward_config.yaml"


    #❗❗❗我现在SFT,RL的相关参数都设置的很小，但是流程都跑通了，后面肯定需要加大调参
    # stage1 会默认加载stage1中的最新的checkpoint对应的模型，如果有需要可以改成加载特定的checkpoint
    # Stage2 env/sim settings
    num_envs = 16   #❗❗❗可以变，确认没问题了可以变大！
    reset_dist = 0.2
    TIMEOUT_SECONDS = 60.0
    physics_dt = 0.025
    max_v = float(MAX_V)
    max_w = float(MAX_W)
    lidar_num_rays = 36
    lidar_max_range = 3.0
    DCOL = 0.2
    DCRIT = 0.5

    # SFT settings
    enable_sft_warmup = True
    sft_collect_total_steps = 2000
    sft_opt_epochs = 50

    # RL settings
    ppo_update_interval_steps = 124
    ppo_total_update_epochs = 30
    ckpt_save_every_update_epochs = 6
    rl_progress_print_every_steps = max(1, ppo_update_interval_steps // 20)

    # Output/log settings
    checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints", "ppo_stage2_rl_sft")
    run_dir = os.path.join(PROJECT_ROOT, "runs", "ppo_stage2_rl_sft")
    stage1_checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints", "ppo_stage1_rl_sft")

    # Reproducibility
    seed = 0
    # ============================================================

    # Runtime phase state
    sft_collection_active = enable_sft_warmup

    # Load configs
    network_cfg = load_config(network_cfg_path)
    algo_cfg = load_config(algo_cfg_path)
    reward_cfg = load_config(reward_cfg_path)

    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Env setup (stage2 mixed scenes)
    scene_list = [SCENE_EMPTY, SCENE_CYLINDER, SCENE_DOOR, SCENE_BOX]
    scene_types = [scene_list[i % 4] for i in range(num_envs)]
    print("\n=== Stage2场景配置 ===")
    scene_count = {}
    for scene_type in scene_types:
        scene_count[scene_type] = scene_count.get(scene_type, 0) + 1
    for scene_type, count in sorted(scene_count.items()):
        print(f"{scene_type}: {count}个环境")

    env_cfg = OmegaConf.create({"sim": {"dt": physics_dt}})
    env_setup = EnvironmentSetup(
        env_cfg=env_cfg,
        num_envs=num_envs,
        scene_types=scene_types,
        simulation_app=simulation_app,
        rng=rng,
        show_visual_walls=None,
        reset_scene_obstacles=True,
    )
    setup_result = env_setup.setup_all()
    world = setup_result["world"]
    robots = setup_result["robots"]
    scene_manager = setup_result["scene_manager"]
    diff_ctrl = setup_result["diff_ctrl"]
    left_idx = setup_result["left_idx"]
    right_idx = setup_result["right_idx"]
    env_origins = setup_result["env_origins"]
    goal_pos = setup_result["goal_pos"]
    markers = setup_result["markers"]

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

    # Network/Algo/Reward
    network_type, network_params = get_enabled_component(network_cfg.networks)
    network_params = network_params.copy() if network_params else {}
    network_params["obs_dim"] = 44
    network_params["act_dim"] = 2
    network_params["lidar_dim"] = lidar_num_rays
    network_params["state_dim"] = 8
    policy = create_policy(network_type, **network_params)
    value = create_value(network_type, **network_params)

    algorithm_type, algorithm_params = get_enabled_component(algo_cfg.algorithms)
    rl_agent = create_algorithm(
        algorithm_type,
        policy=policy,
        value=value,
        config=algorithm_params,
        device=device,
    )

    # Stage2 continues from stage1_rl_sft by default
    _ = load_checkpoint(rl_agent, checkpoint_path=checkpoint_path, checkpoint_dir=stage1_checkpoint_dir)
    reward_composer = RewardComposer(reward_cfg.rewards)

    # SFT controller
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

    ppo_update_epoch_count = 0
    step_count = -1
    current_time = 0.0

    reward_buffer = deque(maxlen=100)
    reward_component_sums = {}
    reward_component_count = 0
    supervised_obs = []
    supervised_actions = []
    total_obstacle_collision = 0

    writer = SummaryWriter(log_dir=run_dir)
    temp_cur_act = np.zeros((num_envs, 2), dtype=np.float32)
    temp_logprob = np.zeros((num_envs,), dtype=np.float32)

    print("\n=== Start Stage2 RL+SFT ===")
    print(
        f"num_envs={num_envs}, scenes=EMPTY/CYLINDER/DOOR/BOX, use_supervised={enable_sft_warmup}, "
        f"supervised_steps={sft_collect_total_steps}"
    )

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

        obstacle_collision, _, min_dist = check_obstacle_collision(lidar_ranges, DCOL, DCRIT)
        total_obstacle_collision += int(np.sum(obstacle_collision))

        v_cmd = np.zeros(num_envs, dtype=np.float32)
        w_cmd = np.zeros(num_envs, dtype=np.float32)
        log_probs = np.zeros(num_envs, dtype=np.float32)
        current_act = np.zeros((num_envs, 2), dtype=np.float32)

        if sft_collection_active and step_count < sft_collect_total_steps:
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
            if sft_collection_active:
                final_loss = train_supervised_policy(
                    rl_agent,
                    np.array(supervised_obs, dtype=np.float32),
                    np.array(supervised_actions, dtype=np.float32),
                    device=device,
                    epochs=sft_opt_epochs,
                )
                save_model(rl_agent, checkpoint_dir, "supervised_final")
                print(f"[SFT] done, final loss = {final_loss}")
                sft_collection_active = False
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

        action_history[:, 2:] = action_history[:, 0:2]
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
            robots,
            env_origins,
            goal_pos,
            lidar_num_rays,
            lidar_max_range,
            scene_manager,
            max_v,
            max_w,
            action_history,
            device=device,
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
            if k not in reward_component_sums:
                reward_component_sums[k] = 0.0
            reward_component_sums[k] += float(np.sum(v))

        if not sft_collection_active:
            updated_this_step = False
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

            if step_count % ppo_update_interval_steps == 0 and len(rl_agent.buffer["obs"]) >= rl_agent.batch_size:
                buffer_len = len(rl_agent.buffer["obs"])
                train_info = rl_agent.update()
                ppo_update_epoch_count += 1
                updated_this_step = True
                avg_reward = np.mean(reward_buffer) if reward_buffer else 0.0

                print("\n=== PPO策略更新 ===")
                print(
                    f"更新触发步数(step): {step_count}, "
                    f"PPO更新epoch: {ppo_update_epoch_count}/{ppo_total_update_epochs}, "
                    f"最近100步平均奖励: {avg_reward:.2f}"
                )
                print(f'Actor Loss: {train_info["actor_loss"]:.4f}, Critic Loss: {train_info["critic_loss"]:.4f}')
                print(f'Entropy: {train_info["entropy"]:.4f}, Approx KL: {train_info["approx_kl"]:.4f}')
                print(f"总碰撞次数：{total_obstacle_collision}")

                writer.add_scalar("train/actor_loss", train_info["actor_loss"], ppo_update_epoch_count)
                writer.add_scalar("train/critic_loss", train_info["critic_loss"], ppo_update_epoch_count)
                writer.add_scalar("train/entropy", train_info["entropy"], ppo_update_epoch_count)
                writer.add_scalar("train/approx_kl", train_info["approx_kl"], ppo_update_epoch_count)

                if reward_component_count > 0:
                    for k, v_sum in reward_component_sums.items():
                        writer.add_scalar(f"reward/{k}", v_sum / reward_component_count, ppo_update_epoch_count)
                writer.add_scalar("reward/total_mean", avg_reward, ppo_update_epoch_count)
                writer.add_scalar("rollout/steps_per_update", ppo_update_interval_steps, ppo_update_epoch_count)
                writer.add_scalar("rollout/buffer_size", buffer_len, ppo_update_epoch_count)
                writer.add_scalar("rollout/step_count", step_count, ppo_update_epoch_count)

                for k in reward_component_sums.keys():
                    reward_component_sums[k] = 0.0
                reward_component_count = 0

                if ppo_update_epoch_count % ckpt_save_every_update_epochs == 0:
                    save_model(rl_agent, checkpoint_dir, f"epoch_{ppo_update_epoch_count:03d}")

                if ppo_update_epoch_count >= ppo_total_update_epochs:
                    save_model(rl_agent, checkpoint_dir, "final")
                    break

            if (not updated_this_step) and (step_count % rl_progress_print_every_steps == 0):
                step_in_update = step_count % ppo_update_interval_steps
                next_update_in = (
                    ppo_update_interval_steps - step_in_update
                    if step_in_update != 0
                    else ppo_update_interval_steps
                )
                print(
                    "[RL进度] "
                    f"step={step_count}, "
                    f"ppo_epoch={ppo_update_epoch_count}/{ppo_total_update_epochs}, "
                    f"buffer={len(rl_agent.buffer['obs'])}, "
                    f"距下次update约{next_update_in}步"
                )

        if sft_collection_active:
            print("\n=== 监督学习数据收集 ===")
            print(
                f"当前阶段: SUPERVISED | step={step_count}/{sft_collect_total_steps} | "
                f"已收集样本: {len(supervised_obs)}"
            )

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
    import argparse

    parser = argparse.ArgumentParser(description="Stage2 RL+SFT training script")
    parser.add_argument("--checkpoint", type=str, default=None, help="optional checkpoint path")
    args = parser.parse_args()

    main(checkpoint_path=args.checkpoint)
